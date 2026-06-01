"""
InferenceChain full node.

Wires every subsystem together and runs them concurrently:
  • Sequencer        — 1-second block production
  • ShardProtocol    — parallel inference job orchestration
  • P2PNode          — WebSocket gossip (blocks, shard offers, shard results)
  • RPCServer        — JSON-RPC 2.0 (port 8545)
  • RollupPoster     — L1 state-root commitments every 100 blocks
  • HealthServer     — metrics endpoint (port 9095)

Usage:
    python -m chain                         # reads genesis.json + .env
    python -m chain --genesis /path/genesis.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from aiohttp import web

from .genesis import build_genesis, load_config, CHAIN_DEFAULTS
from .sequencer import Sequencer
from .shard.protocol import ShardProtocol
from .shard.benchmark_store import BenchmarkStore
from .shard.thought_store import ThoughtStore
from .benchmark.runner import BenchmarkRunner
from .p2p.node import P2PNode
from .p2p.messages import TOPICS
from .p2p.thought_protocol import (
    handle_thought_broadcast, handle_sync_request, handle_sync_response,
    THOUGHT_SYNC_REQUEST, THOUGHT_SYNC_RESPONSE,
)
from .rpc.server import RPCServer
from .crypto import address_from_key

log = logging.getLogger("chain.node")

# ── Prometheus metrics ────────────────────────────────────────────────────────
BLOCKS_PRODUCED  = Counter("ic_blocks_total",       "Total L2 blocks produced")
JOBS_POSTED      = Counter("ic_jobs_posted_total",   "Total inference jobs posted")
SHARDS_COMPLETED = Counter("ic_shards_completed_total", "Total shards completed")
ACTIVE_JOBS      = Gauge  ("ic_active_jobs",         "Currently in-flight jobs")
CHAIN_HEAD       = Gauge  ("ic_chain_head",          "Current L2 block number")


class InferenceChainNode:
    """
    Full InferenceChain L2 node. Owns all subsystems and their lifecycles.
    """

    def __init__(
        self,
        genesis_path:  str = "genesis.json",
        privkey:       Optional[str] = None,
        rpc_host:      str = "0.0.0.0",
        rpc_port:      int = 8545,
        p2p_host:      str = "0.0.0.0",
        p2p_port:      int = 9000,
        health_port:   int = 9095,
        bootstrap_peers: list[str] = None,
        l1_dep_path:   str = "l1_deployment.json",
        db_dir:        str = "data",
    ):
        self._genesis_path   = genesis_path
        self._privkey        = privkey or os.environ.get("SEQUENCER_PRIVATE_KEY", "")
        self._rpc_host       = rpc_host
        self._rpc_port       = rpc_port
        self._p2p_host       = p2p_host
        self._p2p_port       = p2p_port
        self._health_port    = health_port
        self._bootstrap      = bootstrap_peers or []
        self._l1_dep_path    = l1_dep_path
        self._db_dir         = db_dir
        self._shutdown       = asyncio.Event()

        if not self._privkey:
            raise ValueError("SEQUENCER_PRIVATE_KEY is required (set env var or pass --privkey)")

        self.address = address_from_key(self._privkey)
        Path(db_dir).mkdir(exist_ok=True)

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        cfg = {**CHAIN_DEFAULTS, **load_config(self._genesis_path)}

        # ── Sequencer ─────────────────────────────────────────────────────
        self.sequencer = Sequencer(
            genesis_path=self._genesis_path,
            db_path=str(Path(self._db_dir) / "chain"),
            privkey=self._privkey,
        )
        await self.sequencer.start()

        # ── P2P node ──────────────────────────────────────────────────────
        self.p2p = P2PNode(
            host=self._p2p_host,
            port=self._p2p_port,
            bootstrap_peers=self._bootstrap,
            privkey=self._privkey,
            sender_address=self.address,
        )
        await self.p2p.start()

        # ── Shard protocol ────────────────────────────────────────────────
        self.shards = ShardProtocol(self.sequencer, self.p2p, cfg)

        # Wire P2P shard_results → ShardProtocol.on_shard_result
        self.p2p.subscribe(TOPICS["shard_results"], self.shards.on_shard_result)

        # Wire P2P context_load_results → ShardProtocol.on_context_load_result (Phase 3)
        self.p2p.subscribe(TOPICS["context_load_results"], self.shards.on_context_load_result)

        # Update backend capability registry from miner heartbeats (needed for tensor_parallel routing)
        self.p2p.subscribe(TOPICS["heartbeats"], self.shards.on_miner_heartbeat)

        # When a new miner connects, re-dispatch any pending jobs they missed
        self.p2p._on_peer_connected = self.shards.recover_pending_jobs

        # ── Benchmark runner + pg_inft store ─────────────────────────────
        self.benchmark = BenchmarkRunner(self.sequencer, self.p2p, cfg)
        self.p2p.subscribe(TOPICS["benchmark_responses"], self.benchmark.on_benchmark_response)

        # BenchmarkStore + ThoughtStore: Peerbit distributed store via sidecar.
        # Optional — chain works without it.
        peerbit_url = cfg.get("peerbit_url") or os.environ.get("PEERBIT_URL")
        self.benchmark_store: Optional[BenchmarkStore] = None
        self.thought_store: Optional[ThoughtStore] = None
        if peerbit_url:
            self.benchmark_store = BenchmarkStore(peerbit_url)
            await self.benchmark_store.connect()
            self.thought_store = ThoughtStore(peerbit_url)
            await self.thought_store.connect()
            log.info("thought_store_connected")
            # Give ShardProtocol access so dispatch_job() can proactively pre-fetch
            self.shards._thought_store = self.thought_store

        # Subscribe thought gossip so sequencer's pg_inft mirrors all inferences
        if self.thought_store:
            async def _on_thought_broadcast(payload: dict) -> None:
                peer = payload.get("miner_address", "peer")
                await handle_thought_broadcast(payload, peer, self.thought_store)

            async def _on_thought_sync(payload: dict) -> None:
                msg_type = int(payload.get("type", 0))
                if msg_type == THOUGHT_SYNC_REQUEST:
                    requester = payload.get("requester", "peer")
                    peer = self.p2p.peer_by_address(requester) if hasattr(self.p2p, "peer_by_address") else None
                    if peer:
                        await handle_sync_request(payload, peer, self.address, self.thought_store)
                elif msg_type == THOUGHT_SYNC_RESPONSE:
                    await handle_sync_response(payload, payload.get("responder", "peer"), self.thought_store)

            self.p2p.subscribe(TOPICS["thought_broadcast"], _on_thought_broadcast)
            self.p2p.subscribe(TOPICS["thought_sync"],      _on_thought_sync)

        # Wire sequencer's block hook to P2P broadcast + metrics
        self.sequencer.on_block_produced = self._on_block_produced

        # Wire sequencer's job dispatch to ShardProtocol
        self.sequencer.shard_protocol = self.shards

        # ── Rollup poster (optional — requires l1_deployment.json) ────────
        self._rollup_poster = None
        if Path(self._l1_dep_path).exists() and os.environ.get("L1_RPC_URL"):
            try:
                self._rollup_poster = self._build_rollup_poster(cfg)
                log.info("rollup_poster_wired l1_dep=%s", self._l1_dep_path)
            except Exception as exc:
                log.warning("rollup_poster_skipped reason=%s", exc)
        else:
            log.info("rollup_poster_disabled (no l1_deployment.json or L1_RPC_URL)")

        # ── RPC server ────────────────────────────────────────────────────
        self.rpc = RPCServer(
            sequencer=self.sequencer,
            shard_protocol=self.shards,
            benchmark=self.benchmark,
            thought_store=self.thought_store,
            host=self._rpc_host,
            port=self._rpc_port,
        )

        log.info(
            "node_ready address=%s chain_id=%d rpc=:%d p2p=:%d",
            self.address, self.sequencer.chain_id, self._rpc_port, self._p2p_port,
        )

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.start()

        tasks = [
            asyncio.create_task(self.sequencer.run(), name="sequencer"),
            asyncio.create_task(self.p2p.serve(),     name="p2p-server"),
            asyncio.create_task(self.rpc.run(),       name="rpc-server"),
            asyncio.create_task(self._health_server(),name="health"),
        ]

        # Wait for shutdown signal
        await self._shutdown.wait()

        log.info("node_shutting_down")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("node_stopped")

    def stop(self) -> None:
        self._shutdown.set()
        self.sequencer.stop()

    # ── Block hook ────────────────────────────────────────────────────────────

    async def _on_block_produced(self, block) -> None:
        # Broadcast block to peers
        await self.p2p.broadcast(TOPICS["blocks"], block.to_dict())

        # Update metrics
        BLOCKS_PRODUCED.inc()
        CHAIN_HEAD.set(block.header.block_number)

        # Count jobs in block; persist benchmark scores to pg_inft
        from .types import TxType
        for tx in block.transactions:
            if tx.tx_type == TxType.JOB_POST:
                JOBS_POSTED.inc()
            elif tx.tx_type == TxType.SHARD_COMMIT:
                SHARDS_COMPLETED.inc()
            elif tx.tx_type == TxType.BENCHMARK_COMMIT and self.benchmark_store:
                asyncio.create_task(
                    self.benchmark_store.upsert_from_tx(tx, block.header.block_number)
                )

        # Post state root to L1 if it's time
        if self._rollup_poster:
            try:
                await self._rollup_poster.maybe_post(block, self.sequencer.state())
            except Exception as exc:
                log.warning("rollup_post_error err=%s", exc)

    # ── Health / metrics server ───────────────────────────────────────────────

    async def _health_server(self) -> None:
        async def handle_health(req: web.Request) -> web.Response:
            head = self.sequencer.head()
            active = len(self.shards.active_jobs()) if self.shards else 0
            ACTIVE_JOBS.set(active)
            return web.json_response({
                "status":       "ok",
                "address":      self.address,
                "block_number": head.header.block_number,
                "block_hash":   head.block_hash[:16] + "…",
                "active_jobs":  active,
                "tps":          round(self.sequencer.tps(), 2),
                "validators":   len(self.sequencer.state().active_validators()),
            })

        async def handle_metrics(req: web.Request) -> web.Response:
            return web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)

        app = web.Application()
        app.router.add_get("/health",  handle_health)
        app.router.add_get("/metrics", handle_metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._health_port)
        await site.start()
        log.info("health_server_started port=%d", self._health_port)

    # ── Rollup poster construction ─────────────────────────────────────────────

    def _build_rollup_poster(self, cfg: dict):
        from .rollup_poster import RollupPoster
        dep = json.loads(Path(self._l1_dep_path).read_text())
        return RollupPoster(
            l1_rpc_url=os.environ["L1_RPC_URL"],
            rollup_address=dep["rollup"]["address"],
            rollup_abi=dep["rollup"]["abi"],
            sequencer_privkey=self._privkey,
            cfg=cfg,
            db_path=str(Path(self._db_dir) / "rollup.db"),
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceChain L2 node")
    parser.add_argument("--genesis",   default="genesis.json",    help="Genesis config path")
    parser.add_argument("--privkey",   default="",                help="Sequencer private key (overrides env)")
    parser.add_argument("--rpc-port",  type=int, default=8545,    help="JSON-RPC port")
    parser.add_argument("--p2p-port",  type=int, default=9000,    help="P2P WebSocket port")
    parser.add_argument("--health-port", type=int, default=9095,  help="Health/metrics port")
    parser.add_argument("--peers",     nargs="*", default=[],     help="Bootstrap peer ws://host:port")
    parser.add_argument("--l1-dep",    default="l1_deployment.json", help="L1 deployment JSON path")
    parser.add_argument("--db",        default="data",            help="Data directory")
    parser.add_argument("--log-level", default="INFO",            help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    node = InferenceChainNode(
        genesis_path=args.genesis,
        privkey=args.privkey or None,
        rpc_port=args.rpc_port,
        p2p_port=args.p2p_port,
        health_port=args.health_port,
        bootstrap_peers=args.peers,
        l1_dep_path=args.l1_dep,
        db_dir=args.db,
    )

    loop = asyncio.get_event_loop()

    def _handle_signal(sig_name: str) -> None:
        log.info("signal_received sig=%s — shutting down", sig_name)
        node.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig.name: _handle_signal(s))

    loop.run_until_complete(node.run())
