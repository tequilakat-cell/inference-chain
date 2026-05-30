"""
Sequencer — the main L2 block production loop.

Owns the canonical chain tip and state. Produces one block per block_time_ms,
applies it, broadcasts to P2P peers, and triggers the rollup poster every
state_root_interval blocks.

Single-sequencer for v1. Decentralised sequencer rotation is v2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .genesis import build_genesis, load_config, CHAIN_DEFAULTS
from .types import Block, TxType
from .state import StateDB
from .mempool import Mempool, build_transaction
from .block_builder import BlockBuilder
from .block_validator import validate_block
from .crypto import address_from_key

log = logging.getLogger("chain.sequencer")


class Sequencer:
    def __init__(
        self,
        genesis_path: str = "genesis.json",
        db_path:      str = "chain.db",
        privkey:      Optional[str] = None,
    ):
        self.cfg = {**CHAIN_DEFAULTS, **load_config(genesis_path)}
        self._privkey = privkey or os.environ.get("SEQUENCER_PRIVATE_KEY", "")
        if not self._privkey:
            raise ValueError("SEQUENCER_PRIVATE_KEY not set")

        self.address   = address_from_key(self._privkey)
        self.chain_id  = self.cfg["chain_id"]
        self.db_path   = db_path

        self._head:    Optional[Block]   = None
        self._state:   Optional[StateDB] = None
        self.mempool   = Mempool()
        self._builder  = BlockBuilder(
            sequencer_address=self.address,
            sequencer_privkey=self._privkey,
            chain_id=self.chain_id,
        )

        # Pluggable hooks — set by the node runner
        self.on_block_produced = None   # async (block: Block) → None
        self.shard_protocol    = None   # ShardProtocol instance

        self._shutdown = asyncio.Event()
        self._block_times: list[float] = []
        self._started = False
        self._block_history: deque = deque(maxlen=1000)  # newest first, for explorer

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise chain from genesis or restore from db. Idempotent."""
        if self._started:
            return
        self._started = True
        snap_path = Path(self.db_path + ".snap.json")
        if snap_path.exists():
            log.info("restoring_state from=%s", snap_path)
            snap = json.loads(snap_path.read_text())
            self._state = StateDB.from_snapshot(snap["state"])
            # Reconstruct a minimal genesis block as parent for the restored state
            genesis_block, _ = build_genesis(self.cfg)
            self._head = genesis_block   # actual head stored in snap; simplified for MVP
        else:
            log.info("building_genesis chain_id=%d sequencer=%s", self.chain_id, self.address)
            genesis_cfg = {
                **self.cfg,
                "sequencer_address": self.address,
            }
            self._head, self._state = build_genesis(genesis_cfg)

        log.info(
            "sequencer_ready address=%s chain_id=%d head=%d",
            self.address, self.chain_id, self._head.header.block_number,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.start()
        block_time_s = self.cfg["block_time_ms"] / 1000.0
        state_root_interval = self.cfg["state_root_interval"]

        while not self._shutdown.is_set():
            t0 = time.monotonic()
            try:
                await self._produce_block()
            except Exception as exc:
                log.error("block_production_error err=%s", exc, exc_info=True)

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, block_time_s - elapsed))

        log.info("sequencer_stopped")

    async def _produce_block(self) -> None:
        block, new_state = await self._builder.build_next(
            self._head, self._state, self.mempool
        )

        # Validate own block (paranoia check)
        valid, reason = validate_block(block, self._head, self.address, self.chain_id)
        if not valid:
            log.error("own_block_invalid reason=%s", reason)
            return

        self._head  = block
        self._state = new_state
        self._block_history.appendleft(block)   # newest at index 0

        # Track timing
        self._block_times.append(time.monotonic())
        if len(self._block_times) > 100:
            self._block_times.pop(0)

        # Trigger shard assignment for any new JOB_POST transactions
        if self.shard_protocol:
            for tx in block.transactions:
                if tx.tx_type == TxType.JOB_POST:
                    p = tx.payload_dict()
                    # Only dispatch immediately if the job is PENDING (not WAITING on a parent).
                    job = self._state.job(p["job_id"])
                    if job and job.status == "pending" and not job.parent_job_id:
                        asyncio.create_task(
                            self.shard_protocol.dispatch_job(p, block, self._state)
                        )

            # Dispatch chained jobs whose parent completed in this block.
            for job_id in self._state.pop_newly_pending():
                asyncio.create_task(
                    self.shard_protocol.dispatch_chained_job(job_id, block, self._state)
                )

        # Broadcast to P2P
        if self.on_block_produced:
            await self.on_block_produced(block)

        # Persist snapshot every 100 blocks
        if block.header.block_number % 100 == 0:
            self._persist_snapshot()

    # ── Public API (used by RPC and bridge) ───────────────────────────────────

    def head(self) -> Block:
        return self._head

    def state(self) -> StateDB:
        return self._state

    async def submit_transaction(self, tx) -> tuple[bool, str]:
        """Accept a raw transaction into the mempool after validation."""
        from .mempool import validate_transaction
        current_nonce = self._state.nonce(tx.sender) if tx.sender else 0
        valid, reason = validate_transaction(tx, current_nonce, self.chain_id)
        if not valid:
            return False, reason
        added = await self.mempool.add(tx)
        return added, "" if added else "duplicate"

    def stop(self) -> None:
        self._shutdown.set()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _persist_snapshot(self) -> None:
        try:
            snap = {
                "block_number": self._head.header.block_number,
                "block_hash":   self._head.block_hash,
                "state":        self._state.to_snapshot(),
            }
            Path(self.db_path + ".snap.json").write_text(json.dumps(snap, indent=2))
        except Exception as exc:
            log.warning("snapshot_failed err=%s", exc)

    def tps(self) -> float:
        if len(self._block_times) < 2:
            return 0.0
        window = self._block_times[-1] - self._block_times[0]
        return len(self._block_times) / window if window > 0 else 0.0
