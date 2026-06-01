"""
InferenceChain L2 miner — pure L2, no L1 chain interaction.

Architecture vs the L1 miner:
  L1 miner: claimJob() → submitResult() → finalizeJob()  (all on-chain txs)
  L2 miner: receive ShardOffer (P2P) → run inference → broadcast ShardResult (P2P)

The sequencer handles all on-chain bookkeeping. The miner only needs to:
  1. Connect to the L2 P2P network
  2. Listen for ShardOffer messages assigned to its address
  3. Run inference using the GPU/CPU backend
  4. Broadcast ShardResult back over P2P

No ETH bonds. No L1 transactions. No 10-minute challenge windows.
Shard offers arrive in <100ms; the full round trip is limited only by inference time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import base64

import aiohttp
from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ── GPU backends (reuse L1 miner backends) ────────────────────────────────────
# Append L1 miner dir AFTER the current package so 'miner' resolves to our
# inference_chain/miner/ package, not inference/miner/miner.py
_L1_MINER_DIR = str(Path(__file__).parent.parent.parent / "l1" / "miner")
if _L1_MINER_DIR not in sys.path:
    sys.path.append(_L1_MINER_DIR)
from backends import get_backend, get_available_backends
from backends.base import InferenceBackend
import keys

# ── L2 chain modules ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from chain.p2p.node import P2PNode
from chain.p2p.messages import TOPICS, make_envelope
from chain.types import ShardResult
from chain.crypto import keccak256_hex, sign as crypto_sign, address_from_key, verify_sig
from chain.shard.context_cache import ContextKVCache
from chain.shard.thought_store import ThoughtStore
from chain.p2p.thought_protocol import (
    ThoughtBroadcast, THOUGHT_BROADCAST, THOUGHT_SYNC_REQUEST, THOUGHT_SYNC_RESPONSE,
    handle_thought_broadcast, handle_sync_request, handle_sync_response,
    send_sync_request,
)
from .registry_client import RegistryClient
from .thought_extractor import extract_thinking, build_proof

# pg_inft context injector (lives in inference/miner/memory/)
try:
    _L1_MEMORY_DIR = str(Path(__file__).parent.parent.parent / "l1" / "miner" / "memory")
    if _L1_MEMORY_DIR not in sys.path:
        sys.path.append(_L1_MEMORY_DIR)
    from context_injector import format_prior_context, inject_context  # type: ignore
    _CONTEXT_INJECTOR_AVAILABLE = True
except Exception as _ctx_exc:
    _CONTEXT_INJECTOR_AVAILABLE = False
    log.warning("context_injector_unavailable err=%s — context injection disabled", _ctx_exc)
    def format_prior_context(results, max_tokens=512): return ""  # type: ignore
    def inject_context(prompt, prior_context): return prompt  # type: ignore

log = logging.getLogger("l2_miner")


def _activation_hash(hidden_bytes: bytes) -> str:
    """Short hex hash of a hidden-state tensor, used in placeholder ShardResult outputs."""
    import hashlib
    return hashlib.sha256(hidden_bytes).hexdigest()[:16]

# ── Metrics ───────────────────────────────────────────────────────────────────
SHARDS_OFFERED    = Counter  ("l2_shards_offered_total",    "Shard offers received")
SHARDS_ACCEPTED   = Counter  ("l2_shards_accepted_total",   "Shard offers accepted")
SHARDS_COMPLETED  = Counter  ("l2_shards_completed_total",  "Shards successfully completed")
SHARDS_FAILED     = Counter  ("l2_shards_failed_total",     "Shards that errored")
INFER_SECONDS     = Histogram("l2_inference_seconds",       "Inference latency",
                               buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120])
ACTIVE_SHARDS     = Gauge    ("l2_active_shards",           "Currently processing shards")
INFT_BALANCE      = Gauge    ("l2_inft_balance",            "L2 INFT balance (from RPC)")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class L2MinerConfig:
    private_key:           str
    models:                dict[str, str]   # hf_model_id → /path/to/model.gguf
    l2_rpc_url:            str = "http://127.0.0.1:8545"
    l2_chain_id:           int = 2026
    bootstrap_peers:       list[str] = field(default_factory=list)
    p2p_host:              str = "0.0.0.0"
    p2p_port:              int = 9001
    health_port:           int = 9090
    max_concurrent_shards: int = 4
    key_dir:               str = "~/.inference-miner/keys"
    log_level:             str = "INFO"
    backend:               str = ""         # auto-detect if empty
    encryption_enabled:    bool = False
    min_stake_inft:        int = 100
    # Pipeline parallel (llama.cpp RPC)
    rpc_port:              int = 0          # 0 = disabled; set e.g. 50052 to enable rpc-server
    rpc_advertise_host:    str = ""         # public IP/hostname other miners connect to
    rpc_server_bin:        str = ""         # path to rpc-server binary; auto-detected if empty
    max_memory_gb:         int = 0          # max GPU/RAM budget for this node (0 = unlimited)
    # Phase 4: KV cache
    kv_cache_dir:          str = "/tmp/inft_kv"  # directory for prompt-cache files
    kv_cache_ttl_s:        int = 3600            # evict files older than this
    # Peerbit sidecar URL for distributed memory
    peerbit_url:           Optional[str] = None  # e.g. "http://127.0.0.1:7731"; None disables
    # Semantic memory — embedding model (nomic-embed-text-v1.5 = 768 dims)
    embed_model_path:      str = ""              # path to embedding GGUF; "" disables embeddings


def load_config(path: str) -> L2MinerConfig:
    with open(path) as f:
        raw = json.load(f)

    models = raw.get("models", {})
    models_dir = raw.get("models_dir", "")
    if not models and models_dir and Path(models_dir).is_dir():
        for f_path in Path(models_dir).glob("*.gguf"):
            models[f_path.stem] = str(f_path)

    return L2MinerConfig(
        private_key=os.environ.get("PRIVATE_KEY", raw.get("private_key", "")),
        models=models,
        l2_rpc_url=os.environ.get("L2_RPC_URL", raw.get("l2_rpc_url", "http://127.0.0.1:8545")),
        l2_chain_id=int(raw.get("l2_chain_id", 2026)),
        bootstrap_peers=raw.get("bootstrap_peers", []),
        p2p_host=raw.get("p2p_host", "0.0.0.0"),
        p2p_port=int(raw.get("p2p_port", 9001)),
        health_port=int(raw.get("health_port", 9090)),
        max_concurrent_shards=int(raw.get("max_concurrent_shards", 4)),
        key_dir=raw.get("key_dir", "~/.inference-miner/keys"),
        log_level=raw.get("log_level", "INFO"),
        backend=os.environ.get("BACKEND", raw.get("backend", "")),
        encryption_enabled=raw.get("encryption_enabled", False),
        min_stake_inft=int(raw.get("min_stake_inft", 100)),
        rpc_port=int(raw.get("rpc_port", 0)),
        rpc_advertise_host=raw.get("rpc_advertise_host", ""),
        rpc_server_bin=raw.get("rpc_server_bin", ""),
        max_memory_gb=int(raw.get("max_memory_gb", 0)),
        kv_cache_dir=raw.get("kv_cache_dir", "/tmp/inft_kv"),
        kv_cache_ttl_s=int(raw.get("kv_cache_ttl_s", 3600)),
        peerbit_url=os.environ.get("PEERBIT_URL", raw.get("peerbit_url", None)) or None,
        embed_model_path=os.environ.get("EMBED_MODEL_PATH", raw.get("embed_model_path", "")),
    )


# ── GPU model pool (same as L1 miner) ────────────────────────────────────────

class GPUModelPool:
    def __init__(self, model_map: dict[str, str], preferred_backend: str = ""):
        self._map     = model_map
        self._backend: InferenceBackend = get_backend(
            model_map, preferred=preferred_backend or None
        )
        log.info("gpu_backend_selected backend=%s info=%s",
                 self._backend.name, self._backend.info)

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    async def run(self, model_id: str, prompt: str, max_tokens: int) -> str:
        if not self.supports(model_id):
            raise RuntimeError(f"Model not supported: {model_id}")
        return await self._backend.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=min(max_tokens, 2048),
            temperature=0.7,
        )

    @property
    def name(self) -> str:
        return self._backend.name


# ── Embedder (semantic memory) ───────────────────────────────────────────────

class Embedder:
    """
    Wraps a llama.cpp embedding model (nomic-embed-text-v1.5, 768 dims) to
    vectorize thoughts for semantic memory. Loads lazily on first use; all
    methods degrade to [] if the model path is unset or llama_cpp is missing.
    """

    DIM = 768

    def __init__(self, model_path: str):
        self._path = os.path.expanduser(model_path) if model_path else ""
        self._llm = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._path) and os.path.isfile(self._path)

    def _load(self) -> None:
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self._path, embedding=True, n_ctx=2048, verbose=False,
        )
        log.info("embedder_loaded path=%s dim=%d", self._path, self.DIM)

    async def embed(self, text: str) -> list:
        """Return a 768-dim embedding for text, or [] on any failure."""
        if not self.enabled or not text:
            return []
        try:
            async with self._lock:
                if self._llm is None:
                    await asyncio.get_event_loop().run_in_executor(None, self._load)

            def _run() -> list:
                out = self._llm.create_embedding(text)
                vec = out["data"][0]["embedding"]
                # nomic in llama.cpp may return a nested [[...]] for pooled output
                if vec and isinstance(vec[0], list):
                    vec = vec[0]
                return [float(x) for x in vec]

            vec = await asyncio.get_event_loop().run_in_executor(None, _run)
            if len(vec) != self.DIM:
                log.warning("embed_dim_mismatch got=%d want=%d", len(vec), self.DIM)
                return []
            return vec
        except Exception as exc:
            log.debug("embed_failed err=%s", exc)
            return []


# ── L2 Miner ─────────────────────────────────────────────────────────────────

class L2Miner:
    def __init__(self, cfg: L2MinerConfig, priv_pem: bytes, pub_der: bytes):
        self.cfg         = cfg
        self.address     = address_from_key(cfg.private_key)
        self.models      = GPUModelPool(cfg.models, cfg.backend)
        self._priv_pem   = priv_pem
        self._pub_der    = pub_der

        self._p2p = P2PNode(
            host=cfg.p2p_host,
            port=cfg.p2p_port,
            bootstrap_peers=cfg.bootstrap_peers,
            privkey=cfg.private_key,
            sender_address=self.address,
        )

        # Queue of offers declined due to capacity — retried when shards finish
        self._pending_offers: list[dict] = []

        # Miner registry — writes to Peerbit sidecar, degrades gracefully if peerbit_url absent
        self._registry = RegistryClient(
            address=self.address,
            models=list(cfg.models.keys()),
            backend=cfg.backend or "cpu",
            url=cfg.peerbit_url or "",
            p2p_addr=f"ws://{cfg.p2p_host}:{cfg.p2p_port}",
            l2_chain_id=cfg.l2_chain_id,
            max_shards=cfg.max_concurrent_shards,
        )

        self._active_shards: dict[str, asyncio.Task] = {}   # key → Task
        self._shutdown    = asyncio.Event()
        self._mine_miner  = None   # CpuMiner, set in start()

        # Pipeline parallel (llama.cpp RPC)
        self._rpc_proc:    Optional[subprocess.Popen] = None   # rpc-server subprocess
        self._rpc_addr:    str = ""                             # advertised "host:port"

        # Legacy tensor-parallel P2P activation state (kept for backward compat)
        self._tp_activations: dict[str, dict] = {}
        self._tp_events: dict[str, asyncio.Event] = {}

        # Phase 3: context load pre-phase.
        # job_id → {shard_index, context_slice, context_hash, model_id}
        # Populated by _on_context_load_offer, consumed by shard execution.
        self._context_chunks: dict[str, dict] = {}

        # Phase 4: persistent KV cache manager.
        # Wraps the prompt-cache files produced by llama-cli --prompt-cache.
        self._kv_cache = ContextKVCache(
            cache_dir=cfg.kv_cache_dir,
            ttl_s=cfg.kv_cache_ttl_s,
        )

        # Peerbit distributed memory rollup via sidecar.
        # None when peerbit_url is not configured — all ops degrade to no-ops.
        self._thought_store: Optional[ThoughtStore] = (
            ThoughtStore(cfg.peerbit_url) if cfg.peerbit_url else None
        )

        # Semantic memory embedder (nomic-embed-text-v1.5). No-op if path unset.
        self._embedder = Embedder(cfg.embed_model_path)

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        available = get_available_backends()
        log.info("available_backends list=%s", available)

        # Connect pg_inft thought store (non-blocking; degrades if unavailable)
        if self._thought_store is not None:
            await self._thought_store.connect()

        # Start MINE CPU miner (runs when inference queue is idle)
        try:
            from mining.mine_cpu import CpuMiner
            self._mine_miner = CpuMiner(
                l2_rpc=self.cfg.l2_rpc_url,
                private_key=self.cfg.private_key,
            )
            asyncio.create_task(self._mine_miner.run(), name="mine-cpu")
            log.info("mine_cpu_started address=%s", self._mine_miner.address)
        except Exception as exc:
            log.warning("mine_cpu_unavailable err=%s", exc)
            self._mine_miner = None

        # Start P2P and subscribe to shard offers + tensor activations
        await self._p2p.start()
        await self._p2p.serve()
        self._p2p.subscribe(TOPICS["shard_offers"],          self._on_shard_offer)
        self._p2p.subscribe(TOPICS["tensor_activations"],   self._on_tensor_activation)
        self._p2p.subscribe(TOPICS["context_load_offers"],  self._on_context_load_offer)
        self._p2p.subscribe(TOPICS["benchmark_challenges"], self._on_benchmark_challenge)
        self._p2p.subscribe(TOPICS["thought_broadcast"],    self._on_thought_broadcast)
        self._p2p.subscribe(TOPICS["thought_sync"],         self._on_thought_sync)
        self._p2p.subscribe(TOPICS["rollup_broadcast"],     self._on_rollup_broadcast)

        # Start Peerbit registry (non-blocking, degrades gracefully if peerbit_url absent)
        asyncio.create_task(self._registry.start())

        # Start rpc-server sidecar for pipeline parallel worker role
        self._rpc_addr = self._start_rpc_server()

        log.info(
            "l2_miner_ready address=%s models=%s backend=%s p2p_port=%d rpc_addr=%s",
            self.address, list(self.cfg.models.keys()), self.models.name,
            self.cfg.p2p_port, self._rpc_addr or "disabled",
        )

        asyncio.create_task(self._register_model_roots())

        # Announce to network — include rpc_addr + live_tps so sequencer can
        # build pipeline jobs with score-proportional layer allocation.
        startup_live_tps: dict[str, float] = {}
        if self._thought_store is not None:
            for m_id in self.cfg.models:
                row = await self._thought_store.get_live_tps(self.address, m_id)
                if row and row.get("live_tps") is not None:
                    startup_live_tps[m_id] = round(float(row["live_tps"]), 4)

        await self._p2p.broadcast(TOPICS["heartbeats"], {
            "type":            "MinerHeartbeat",
            "address":         self.address,
            "models":          list(self.cfg.models.keys()),
            "backend":         self.models.name,
            "tensor_parallel": bool(self._rpc_addr),
            "rpc_addr":        self._rpc_addr,
            "max_memory_gb":   self.cfg.max_memory_gb,
            "live_tps":        startup_live_tps,
        })

        # Bootstrap thought store from peers (cold-start sync)
        if self._thought_store is not None:
            asyncio.create_task(self._bootstrap_thought_sync())

    async def _bootstrap_thought_sync(self) -> None:
        """Request recent thoughts from connected peers to bootstrap local pg_inft."""
        await asyncio.sleep(5)  # wait for peer connections to settle
        peers = list(self._p2p._peers.values()) if hasattr(self._p2p, "_peers") else []
        if not peers:
            return
        for peer in peers[:3]:  # sync from up to 3 peers
            try:
                await send_sync_request(self.address, peer, since_timestamp=0.0, limit=100)
            except Exception as exc:
                log.debug("thought_sync_bootstrap_err peer=%r err=%s", peer, exc)

    async def run(self) -> None:
        await self.start()
        await asyncio.gather(
            self._health_server(),
            self._stats_loop(),
            self._shutdown.wait(),
        )

    # ── Context load pre-phase (Phase 3 / Option B) ──────────────────────────

    async def _on_context_load_offer(self, payload: dict) -> None:
        """
        Handle a ContextLoadOffer from the sequencer.

        Option B (parallel): all miners receive their chunk simultaneously.
        Each miner independently confirms receipt and KV cache status, then
        waits for the ShardOffer which triggers actual inference.

        Phase 4: if a prompt-cache file already exists for (context_hash,
        model_id), report cache_hit=True so the sequencer knows this miner
        will skip context token re-processing during generation.
        """
        assigned = payload.get("assigned_miner", "").lower()
        if assigned != self.address.lower():
            return   # not for us

        job_id        = payload.get("job_id", "")
        shard_index   = int(payload.get("shard_index", 0))
        context_slice = payload.get("context_slice", "")
        context_hash  = payload.get("context_hash", "")
        model_id      = payload.get("model_id", "")

        # Store chunk for use during shard execution
        self._context_chunks[job_id] = {
            "shard_index":   shard_index,
            "context_slice": context_slice,
            "context_hash":  context_hash,
            "model_id":      model_id,
        }

        # Persist full context to local postgres so _run_inference() can pull it
        # via get_job_context() without a blocking search on the hot path.
        if context_slice and self._thought_store is not None:
            asyncio.create_task(
                self._thought_store.set_job_context(
                    job_id=job_id,
                    query_text="",
                    context_text=context_slice,
                    context_hash=context_hash,
                    model_id=model_id,
                    n_entries=len(context_slice.split("\n\n")),
                )
            )

        chunk_hash = keccak256_hex(context_slice.encode("utf-8")) if context_slice else ("0x" + "00" * 32)

        # Phase 4: check if KV cache file already exists for this context
        cache_hit = False
        if context_hash and model_id:
            existing = self._kv_cache.lookup(context_hash, model_id)
            cache_hit = existing is not None
            if not cache_hit:
                # Register the path where we *will* save the cache after first generation
                expected_path = self._kv_cache.cache_path(context_hash, model_id)
                self._context_chunks[job_id]["kv_cache_path"] = expected_path
            else:
                self._context_chunks[job_id]["kv_cache_path"] = existing

        t0 = time.monotonic()
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Sign: keccak(job_id || shard_index || chunk_hash)
        preimage  = (job_id + str(shard_index) + chunk_hash).encode("utf-8")
        signature = crypto_sign(self.cfg.private_key, preimage)

        result_msg = {
            "type":        "ContextLoadResult",
            "job_id":      job_id,
            "shard_index": shard_index,
            "miner":       self.address,
            "chunk_hash":  chunk_hash,
            "cache_hit":   cache_hit,
            "latency_ms":  latency_ms,
            "signature":   signature,
        }
        await self._p2p.broadcast(TOPICS["context_load_results"], result_msg)

        log.info(
            "ctx_load_confirmed job=%s shard=%d cache_hit=%s chunk_chars=%d",
            job_id[:12], shard_index, cache_hit, len(context_slice),
        )

    def _pop_context_chunk(self, job_id: str) -> dict:
        """Retrieve and remove the stored context chunk for a job."""
        return self._context_chunks.pop(job_id, {})

    # ── Hardware benchmark ────────────────────────────────────────────────────

    async def _on_benchmark_challenge(self, payload: dict) -> None:
        """
        Handle a BenchmarkChallenge from the sequencer.

        The sequencer measures wall-clock time from when it sends this challenge
        to when it receives our BenchmarkResponse.  We do NOT self-report time —
        the sequencer owns the clock.

        The nonce is appended to the prompt so the response cannot be precomputed
        or replayed from a previous benchmark session.
        """
        target   = payload.get("miner", "").lower()
        if target != self.address.lower():
            return  # not addressed to us

        model_id = payload.get("model_id", "")
        nonce    = payload.get("nonce", "")
        prompt   = payload.get("prompt", "")
        n_tokens = int(payload.get("n_tokens", 64))

        if not self.models.supports(model_id):
            log.warning(
                "benchmark_declined model=%s reason=not_supported nonce=%s",
                model_id, nonce,
            )
            return

        # Nonce seeding prevents precomputed / cached responses
        seeded_prompt = f"{prompt}\n\n[benchmark_nonce:{nonce}]"

        log.info(
            "benchmark_started model=%s nonce=%s n_tokens=%d",
            model_id, nonce, n_tokens,
        )
        try:
            output = await self.models.run(model_id, seeded_prompt, n_tokens)
        except Exception as exc:
            log.error("benchmark_failed model=%s nonce=%s err=%s", model_id, nonce, exc)
            return

        response = {
            "type":     "BenchmarkResponse",
            "miner":    self.address,
            "model_id": model_id,
            "nonce":    nonce,
            "output":   output[:200],
        }
        await self._p2p.broadcast(TOPICS["benchmark_responses"], response)
        log.info(
            "benchmark_responded model=%s nonce=%s output_chars=%d",
            model_id, nonce, len(output),
        )

    # ── Thought gossip handlers ───────────────────────────────────────────────

    async def _on_thought_broadcast(self, payload: dict) -> None:
        """Receive a ThoughtBroadcast from a peer and ingest into local pg_inft."""
        if self._thought_store is None:
            return
        peer_addr = payload.get("miner_address", "peer")
        await handle_thought_broadcast(payload, peer_addr, self._thought_store)

    async def _on_rollup_broadcast(self, payload: dict) -> None:
        """Receive a consolidated rollup memory from the sequencer; store it locally
        so this miner can inject it at inference time (distributed memory)."""
        if self._thought_store is None:
            return
        try:
            await self._thought_store.upsert_rollup(
                rollup_id=payload.get("rollup_id", ""),
                topic=payload.get("topic", ""),
                model_id=payload.get("model_id", ""),
                summary=payload.get("summary", ""),
                source_count=int(payload.get("source_count", 0)),
                source_job_ids=payload.get("source_job_ids", []),
                embedding=payload.get("embedding", []),
                content_hash=b"",
            )
            log.info(
                "rollup_received id=%s topic=%s sources=%d",
                payload.get("rollup_id", "")[:12], payload.get("topic", ""),
                payload.get("source_count", 0),
            )
        except Exception as exc:
            log.debug("rollup_broadcast_handler_err err=%s", exc)

    async def _on_thought_sync(self, payload: dict) -> None:
        """Handle THOUGHT_SYNC_REQUEST or THOUGHT_SYNC_RESPONSE from a peer."""
        if self._thought_store is None:
            return
        msg_type = int(payload.get("type", 0))
        if msg_type == THOUGHT_SYNC_REQUEST:
            # Another miner is bootstrapping — send our recent thoughts back
            requester = payload.get("requester", "peer")
            peer = self._p2p.peer_by_address(requester) if hasattr(self._p2p, "peer_by_address") else None
            if peer:
                await handle_sync_request(payload, peer, self.address, self._thought_store)
        elif msg_type == THOUGHT_SYNC_RESPONSE:
            peer_addr = payload.get("responder", "peer")
            await handle_sync_response(payload, peer_addr, self._thought_store)

    # ── Shard offer handling ──────────────────────────────────────────────────

    async def _on_shard_offer(self, payload: dict) -> None:
        SHARDS_OFFERED.inc()

        spec      = payload.get("spec", {})
        job_id    = payload.get("job_id", "")
        model_id  = payload.get("model_id", "")
        shard_idx = int(spec.get("shard_index", 0))
        mode      = spec.get("mode", "parallel_sample")

        # Check if this offer is for us
        assigned = spec.get("assigned_miner", "").lower()
        if assigned != self.address.lower():
            return

        # Check capacity — queue the offer rather than dropping it
        if len(self._active_shards) >= self.cfg.max_concurrent_shards:
            log.info("shard_queued job=%s shard=%d (at capacity, will retry)", job_id, shard_idx)
            self._pending_offers.append(payload)
            return

        # Check model support
        if not self.models.supports(model_id):
            log.warning("shard_declined job=%s model=%s reason=not_supported", job_id, model_id)
            return

        shard_key = f"{job_id}:{shard_idx}"
        if shard_key in self._active_shards:
            return   # duplicate offer

        # ── Pipeline parallel branch (llama.cpp RPC) ─────────────────────────
        if mode in ("pipeline_parallel", "tensor_parallel"):
            role = spec.get("role", "coordinator")
            if role == "worker" and not self._rpc_addr:
                log.warning(
                    "pp_worker_declined job=%s shard=%d — rpc-server not running",
                    job_id, shard_idx,
                )
                return
            log.info(
                "pp_accepted job=%s shard=%d role=%s model=%s",
                job_id, shard_idx, role, model_id,
            )
            SHARDS_ACCEPTED.inc()
            ACTIVE_SHARDS.inc()
            if self._mine_miner:
                self._mine_miner.pause()
            asyncio.create_task(self._registry.announce_job_accepted(
                job_id, model_id, mode, spec.get("total_shards", 1)
            ))
            task = asyncio.create_task(
                self._execute_pipeline_shard(payload, shard_key), name=shard_key
            )
            self._active_shards[shard_key] = task
            return

        # ── Normal (non-pipeline) branch ──────────────────────────────────────
        log.info("shard_accepted job=%s shard=%d model=%s", job_id, shard_idx, model_id)
        SHARDS_ACCEPTED.inc()
        ACTIVE_SHARDS.inc()

        # Pause MINE mining while running inference (inference takes priority)
        if self._mine_miner:
            self._mine_miner.pause()

        # Announce to pg_inft job board
        asyncio.create_task(self._registry.announce_job_accepted(
            job_id, model_id, spec.get("mode", "parallel_sample"), spec.get("total_shards", 1)
        ))

        task = asyncio.create_task(
            self._execute_shard(payload, shard_key),
            name=shard_key,
        )
        self._active_shards[shard_key] = task

    # ── Pipeline parallel (llama.cpp RPC) ────────────────────────────────────

    def _start_rpc_server(self) -> str:
        """
        Start rpc-server as an always-on sidecar subprocess.
        Returns the advertised "host:port" string, or "" if disabled/unavailable.
        """
        if not self.cfg.rpc_port:
            return ""

        # Find rpc-server binary — check PATH, brew, and source build dir
        bin_path = self.cfg.rpc_server_bin or ""
        if not bin_path:
            for candidate in ["rpc-server", "llama-rpc-server"]:
                found = shutil.which(candidate)
                if found:
                    bin_path = found
                    break
            if not bin_path:
                extra = [
                    "/opt/homebrew/bin/rpc-server",
                    "/usr/local/bin/rpc-server",
                    "/tmp/llama_cpp_src/build/bin/rpc-server",
                ]
                for p in extra:
                    if os.path.isfile(p) and os.access(p, os.X_OK):
                        bin_path = p
                        break

        if not bin_path:
            log.warning(
                "rpc_server_not_found — install llama.cpp and ensure rpc-server is in PATH. "
                "Pipeline parallel worker role disabled."
            )
            return ""

        host = self.cfg.p2p_host if self.cfg.p2p_host != "0.0.0.0" else "0.0.0.0"
        port = self.cfg.rpc_port

        try:
            self._rpc_proc = subprocess.Popen(
                [bin_path, "--host", "0.0.0.0", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            advertise_host = self.cfg.rpc_advertise_host or "127.0.0.1"
            rpc_addr = f"{advertise_host}:{port}"
            log.info("rpc_server_started pid=%d addr=%s", self._rpc_proc.pid, rpc_addr)
            return rpc_addr
        except Exception as exc:
            log.warning("rpc_server_start_failed err=%s — pipeline parallel worker disabled", exc)
            return ""

    async def _execute_pipeline_shard(self, offer: dict, shard_key: str) -> None:
        """
        Execute one shard of a pipeline parallel job.

        Coordinator (shard 0): load model with --rpc pointing to worker RPC servers,
            run full generation, broadcast ShardResult with real output.
        Worker (shard 1..N-1): rpc-server is already running as sidecar and will
            receive layer weights from coordinator automatically. Submit a placeholder
            ShardResult immediately so the sequencer can track participation.
        """
        spec      = offer.get("spec", {})
        job_id    = offer.get("job_id", "")
        model_id  = offer.get("model_id", "")
        role      = spec.get("role", "coordinator")
        shard_idx = int(spec.get("shard_index", 0))
        max_tok   = int(spec.get("max_tokens", 256))
        prompt    = spec.get("prompt_slice", "")
        rpc_peers = spec.get("rpc_peers", "[]")

        try:
            if role == "worker":
                # Worker: rpc-server is already running. Submit placeholder immediately.
                # The actual compute contribution happens transparently via the RPC protocol.
                placeholder = f"[pipeline_worker:{shard_idx}:{self._rpc_addr}]"
                await self._broadcast_shard_result(job_id, shard_idx, placeholder, 0)
                log.info(
                    "pp_worker_ready job=%s shard=%d rpc_addr=%s",
                    job_id[:12], shard_idx, self._rpc_addr,
                )
                SHARDS_COMPLETED.inc()
                return

            # Coordinator path
            try:
                peer_list: list[str] = json.loads(rpc_peers) if rpc_peers else []
            except Exception:
                peer_list = []

            rpc_servers_str = ",".join(peer_list)

            # Phase 3: prepend coordinator's context chunk to the prompt.
            # In Option B, the coordinator received its own context slice during
            # the pre-phase.  For pipeline_parallel the coordinator gets the full
            # context (see ShardProtocol._split_context for pipeline mode).
            ctx_info = self._pop_context_chunk(job_id)
            ctx_slice = ctx_info.get("context_slice", "")
            kv_cache_path = ctx_info.get("kv_cache_path", "")
            if ctx_slice:
                prompt = ctx_slice + prompt

            log.info(
                "pp_coordinator_start job=%s model=%s rpc_peers=%s ctx_chars=%d kv_cache=%s",
                job_id[:12], model_id, peer_list, len(ctx_slice), kv_cache_path or "none",
            )

            t0 = time.monotonic()

            # Parse per-worker memory budgets and score-proportional tensor split from spec.
            import json as _json
            try:
                worker_memory_gb: list[int] = _json.loads(spec.get("rpc_memory_gb", "[]") or "[]")
            except Exception:
                worker_memory_gb = []

            # tensor_split: score-proportional layer fractions computed by the sequencer.
            # Overrides the memory-budget heuristic when present — benchmark scores are
            # more accurate than estimating from memory_gb.
            try:
                tensor_split_fracs: list[float] = _json.loads(spec.get("tensor_split", "[]") or "[]")
            except Exception:
                tensor_split_fracs = []

            # Access the LlamaCppBackend directly for RPC-aware generation
            backend = self.models._backend
            if hasattr(backend, "generate_with_rpc") and rpc_servers_str:
                # Phase 4: pass prompt_cache_path if we have a KV cache slot.
                # Phase 5: pass tensor_split_fracs for score-proportional layer allocation.
                output = await backend.generate_with_rpc(
                    model_id, prompt, max_tok, 0.7, rpc_servers_str,
                    prompt_cache_path=kv_cache_path,
                    max_memory_gb=self.cfg.max_memory_gb,
                    worker_memory_gb=worker_memory_gb,
                    tensor_split_fracs=tensor_split_fracs,
                )
                # Register the cache file as written (file may now exist on disk)
                if kv_cache_path and ctx_info.get("context_hash") and ctx_info.get("model_id"):
                    self._kv_cache.register(ctx_info["context_hash"], ctx_info["model_id"], kv_cache_path)
            else:
                # Fallback: regular inference without distributing layers
                if rpc_servers_str:
                    log.warning(
                        "pp_rpc_unsupported backend=%s — running local inference only",
                        backend.name,
                    )
                output = await self.models.run(model_id, prompt, max_tok)

            elapsed_ms = int((time.monotonic() - t0) * 1000)

            log.info(
                "pp_coordinator_done job=%s elapsed=%dms output_len=%d",
                job_id[:12], elapsed_ms, len(output),
            )

            # pg_inft: record pipeline coordinator throughput.
            if self._thought_store is not None and elapsed_ms > 0:
                est_tokens = max(1, len(output.split()))
                actual_tps = round(est_tokens / (elapsed_ms / 1000.0), 4)
                asyncio.create_task(
                    self._thought_store.update_live_tps(self.address, model_id, actual_tps)
                )

            await self._broadcast_shard_result(job_id, shard_idx, output, elapsed_ms)
            SHARDS_COMPLETED.inc()
            asyncio.create_task(self._drain_pending_offers())

        except Exception as exc:
            SHARDS_FAILED.inc()
            log.error("pp_shard_failed key=%s role=%s err=%s", shard_key, role, exc, exc_info=True)
        finally:
            self._active_shards.pop(shard_key, None)
            ACTIVE_SHARDS.dec()
            if self._mine_miner and len(self._active_shards) == 0:
                self._mine_miner.resume()

    async def _on_tensor_activation(self, msg: dict) -> None:
        """Receive a hidden-state tensor from an upstream pipeline stage."""
        job_id = msg.get("job_id", "")
        stage  = int(msg.get("stage", -1))
        if not job_id or stage < 0:
            return
        key = f"{job_id}:{stage}"
        self._tp_activations[key] = msg
        event = self._tp_events.pop(key, None)
        if event:
            event.set()

    async def _execute_tensor_shard(self, offer: dict, shard_key: str) -> None:
        """
        Execute one pipeline stage of a TENSOR_PARALLEL job.

        Stage 0   : embed prompt → forward layers[0:layer_end] → broadcast TensorActivation
                    + broadcast placeholder ShardResult so the sequencer can track liveness.
        Stage 1..N-2: wait for upstream activation → forward layers[start:end]
                    → broadcast TensorActivation + placeholder ShardResult.
        Stage N-1 : wait for upstream activation → forward remaining layers + decode
                    → broadcast the real ShardResult with actual text output.
        """
        spec      = offer.get("spec", {})
        job_id    = offer.get("job_id", "")
        model_id  = offer.get("model_id", "")
        stage     = int(spec.get("shard_index", 0))
        n_stages  = int(spec.get("total_shards", 1))
        max_tok   = int(spec.get("max_tokens", 256))
        prompt    = spec.get("prompt_slice", "")
        timeout_s = int(spec.get("timeout_ms", 35_000 * n_stages)) / 1000

        backend = self.models._backend

        try:
            # Determine this stage's layer range from the model's total layer count.
            n_layers   = await backend.num_layers(model_id)
            stage_size = max(1, n_layers // n_stages)
            layer_start = stage * stage_size
            layer_end   = layer_start + stage_size if stage < n_stages - 1 else n_layers

            t0 = time.monotonic()

            if stage == 0:
                hidden_bytes, dtype, shape = await backend.embed_and_forward(
                    model_id, prompt, layer_end
                )
                await self._broadcast_activation(job_id, stage, n_stages, hidden_bytes, dtype, shape)
                placeholder = f"[tp_stage:{stage}:{_activation_hash(hidden_bytes)}]"
                await self._broadcast_shard_result(
                    job_id, stage, placeholder, int((time.monotonic() - t0) * 1000)
                )
                log.info(
                    "tp_stage0_done job=%s layers=[0:%d] shape=%s elapsed=%dms",
                    job_id[:12], layer_end, shape, int((time.monotonic() - t0) * 1000),
                )
                return

            # Wait for the upstream stage's activation.
            prev_key = f"{job_id}:{stage - 1}"
            prev_activation = self._tp_activations.pop(prev_key, None)
            if prev_activation is None:
                event = asyncio.Event()
                self._tp_events[prev_key] = event
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout_s)
                    prev_activation = self._tp_activations.pop(prev_key, None)
                except asyncio.TimeoutError:
                    log.error(
                        "tp_activation_timeout job=%s stage=%d waiting_for=%d",
                        job_id[:12], stage, stage - 1,
                    )
                    SHARDS_FAILED.inc()
                    return

            if prev_activation is None:
                log.error("tp_activation_missing job=%s stage=%d", job_id[:12], stage)
                return

            hidden_bytes = base64.b64decode(prev_activation["data_b64"])
            shape        = prev_activation["shape"]
            dtype        = prev_activation["dtype"]

            if stage == n_stages - 1:
                # Final stage: forward remaining layers and decode to text.
                output = await backend.forward_and_decode(
                    model_id, hidden_bytes, shape, dtype, layer_start, max_tok
                )
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                await self._broadcast_shard_result(job_id, stage, output, elapsed_ms)
                SHARDS_COMPLETED.inc()
                asyncio.create_task(self._drain_pending_offers())
                log.info(
                    "tp_final_done job=%s layers=[%d:%d] elapsed=%dms output_len=%d",
                    job_id[:12], layer_start, n_layers, elapsed_ms, len(output),
                )
            else:
                # Middle stage: forward and pass on.
                new_hidden, new_dtype, new_shape = await backend.forward_layers(
                    model_id, hidden_bytes, shape, dtype, layer_start, layer_end
                )
                await self._broadcast_activation(
                    job_id, stage, n_stages, new_hidden, new_dtype, new_shape
                )
                placeholder = f"[tp_stage:{stage}:{_activation_hash(new_hidden)}]"
                await self._broadcast_shard_result(
                    job_id, stage, placeholder, int((time.monotonic() - t0) * 1000)
                )
                log.info(
                    "tp_middle_done job=%s stage=%d layers=[%d:%d] elapsed=%dms",
                    job_id[:12], stage, layer_start, layer_end,
                    int((time.monotonic() - t0) * 1000),
                )

        except Exception as exc:
            SHARDS_FAILED.inc()
            log.error(
                "tp_stage_failed job=%s stage=%d err=%s", job_id[:12], stage, exc, exc_info=True
            )
        finally:
            self._active_shards.pop(shard_key, None)
            ACTIVE_SHARDS.dec()
            if self._mine_miner and len(self._active_shards) == 0:
                self._mine_miner.resume()

    async def _broadcast_activation(
        self,
        job_id: str,
        stage: int,
        n_stages: int,
        hidden_bytes: bytes,
        dtype: str,
        shape: list[int],
    ) -> None:
        preimage  = (job_id + str(stage) + base64.b64encode(hidden_bytes).decode()).encode()
        signature = crypto_sign(self.cfg.private_key, preimage)
        msg = {
            "type":     "TensorActivation",
            "job_id":   job_id,
            "stage":    stage,
            "n_stages": n_stages,
            "shape":    shape,
            "dtype":    dtype,
            "data_b64": base64.b64encode(hidden_bytes).decode(),
            "producer": self.address,
            "signature": signature,
        }
        await self._p2p.broadcast(TOPICS["tensor_activations"], msg)

    async def _broadcast_shard_result(
        self,
        job_id: str,
        shard_idx: int,
        output: str,
        latency_ms: int,
    ) -> None:
        preimage  = (str(shard_idx) + job_id + output).encode("utf-8")
        signature = crypto_sign(self.cfg.private_key, preimage)
        msg = {
            "type":        "ShardResult",
            "job_id":      job_id,
            "shard_index": shard_idx,
            "miner":       self.address,
            "output":      output,
            "latency_ms":  latency_ms,
            "signature":   signature,
        }
        await self._p2p.broadcast(TOPICS["shard_results"], msg)

    async def _execute_shard(self, offer: dict, shard_key: str) -> None:
        try:
            result = await self._run_inference(offer)
            if result:
                await self._p2p.broadcast(TOPICS["shard_results"], result)
                SHARDS_COMPLETED.inc()
                # When capacity frees, try any queued offers
                asyncio.create_task(self._drain_pending_offers())
                # Log to pg_inft reputation ledger
                job_id    = offer.get("job_id", "")
                shard_idx = int(offer.get("spec", {}).get("shard_index", 0))
                output_hash = keccak256_hex(result["output"].encode())
                asyncio.create_task(self._registry.log_shard_complete(job_id, shard_idx, result["latency_ms"]))
                asyncio.create_task(self._registry.announce_job_complete(job_id, output_hash, result["latency_ms"]))
        except Exception as exc:
            SHARDS_FAILED.inc()
            job_id    = offer.get("job_id", "")
            shard_idx = int(offer.get("spec", {}).get("shard_index", 0))
            asyncio.create_task(self._registry.log_shard_failed(job_id, shard_idx))
            log.error("shard_failed key=%s err=%s", shard_key, exc, exc_info=True)
        finally:
            self._active_shards.pop(shard_key, None)
            ACTIVE_SHARDS.dec()
            # Resume MINE mining when inference queue is empty
            if self._mine_miner and len(self._active_shards) == 0:
                self._mine_miner.resume()

    async def _run_inference(self, offer: dict) -> Optional[dict]:
        spec        = offer.get("spec", {})
        job_id      = offer.get("job_id", "")
        model_id    = offer.get("model_id", "")
        shard_idx   = int(spec.get("shard_index", 0))
        max_tokens  = int(spec.get("max_tokens", 256))
        prompt      = spec.get("prompt_slice", "")

        # Decrypt if enabled
        if self.cfg.encryption_enabled and self._priv_pem and prompt:
            try:
                prompt_bytes = bytes.fromhex(prompt) if prompt.startswith("0x") \
                               else prompt.encode("utf-8")
                prompt = keys.decrypt(self._priv_pem, prompt_bytes).decode("utf-8")
            except Exception as exc:
                log.warning("decrypt_failed job=%s shard=%d err=%s", job_id, shard_idx, exc)

        # Phase 3: prepend this miner's context chunk (Option B — already pre-loaded
        # during the CONTEXT_LOAD pre-phase).  In parallel_sample / context_split
        # modes each miner independently prepends its own context slice so the full
        # conversation history is distributed across miners.
        ctx_info = self._pop_context_chunk(job_id)
        ctx_slice = ctx_info.get("context_slice", "")
        if ctx_slice:
            prompt = ctx_slice + prompt

        # ── Proactive pre-fetch: inject pre-assembled context ─────────────────
        # The sequencer wrote this context into postgres at job-dispatch time via
        # set_job_context().  Reading it here costs one indexed point-lookup instead
        # of a full BM25+trigram search on the hot path.
        if self._thought_store is not None:
            try:
                ctx_row = await self._thought_store.get_job_context(job_id)
                if ctx_row and ctx_row.get("context_text"):
                    prior_ctx = ctx_row["context_text"]
                    prompt = inject_context(prompt, prior_ctx)
                    log.debug(
                        "prefetch_ctx_hit job=%s entries=%d ctx_chars=%d",
                        job_id, ctx_row.get("n_entries", 0), len(prior_ctx),
                    )
                else:
                    # Fallback: sequencer didn't pre-fetch (no pg_inft configured there)
                    prior_results = await self._thought_store.search(prompt, model_id, 5)
                    prior_dicts = [
                        {
                            "id":            r.id,
                            "job_id":        r.job_id,
                            "miner_address": r.miner_address,
                            "model_id":      r.model_id,
                            "question_text": r.question_text,
                            "thinking_text": r.thinking_text,
                            "answer_text":   r.answer_text,
                            "score":         r.score,
                        }
                        for r in prior_results
                    ]
                    prior_ctx = format_prior_context(prior_dicts, max_tokens=512)
                    prompt = inject_context(prompt, prior_ctx)
                    if prior_ctx:
                        log.debug(
                            "prior_context_injected job=%s entries=%d ctx_chars=%d",
                            job_id, len(prior_results), len(prior_ctx),
                        )
            except Exception as exc:
                log.debug("thought_store_pre_hook_err job=%s err=%s", job_id, exc)

        # ── Inject consolidated rollup memory (compact, high-signal) ──────────
        # Embed the query and pull the single most-relevant rollup from the local
        # pg_inft replica, prepending its distilled summary so this inference
        # benefits from a whole cluster of prior knowledge within the token budget.
        if self._embedder.enabled and self._thought_store is not None:
            try:
                q_emb = await self._embedder.embed(spec.get("prompt_slice", "") or prompt)
                rolls = await self._thought_store.search_rollups(q_emb, model_id, 1)
                if rolls and float(rolls[0].get("score", 0)) >= 0.5:
                    prompt = ("Relevant background memory:\n"
                              + rolls[0]["summary_text"] + "\n\n" + prompt)
                    log.info(
                        "rollup_injected job=%s topic=%s score=%.3f",
                        job_id, rolls[0].get("topic", ""), float(rolls[0]["score"]),
                    )
            except Exception as exc:
                log.debug("rollup_inject_err job=%s err=%s", job_id, exc)

        log.info(
            "inference_start job=%s shard=%d model=%s max_tokens=%d backend=%s ctx_chars=%d",
            job_id, shard_idx, model_id, max_tokens, self.models.name, len(ctx_slice),
        )

        t0 = time.monotonic()
        output = await self.models.run(model_id, prompt, max_tokens)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        INFER_SECONDS.observe(elapsed_ms / 1000)

        log.info(
            "inference_done job=%s shard=%d elapsed=%dms output_len=%d",
            job_id, shard_idx, elapsed_ms, len(output),
        )

        # pg_inft: record production throughput for benchmark score refinement.
        # Estimate tokens from output word count (proportional to real token count).
        if self._thought_store is not None and elapsed_ms > 0:
            est_tokens = max(1, len(output.split()))
            actual_tps = round(est_tokens / (elapsed_ms / 1000.0), 4)
            asyncio.create_task(
                self._thought_store.update_live_tps(self.address, model_id, actual_tps)
            )

        # ── pg_inft post-inference hook: extract thinking and ingest ──────────
        # Extract <think>...</think> blocks, build Ethereum ECDSA proof, and
        # ingest into the distributed thought store.  Wrapped in try/except.
        if self._thought_store is not None and self.cfg.private_key:
            try:
                question_text = spec.get("prompt_slice", "")
                thinking_text, answer_text = extract_thinking(output)
                proof_sig_hex = build_proof(
                    job_id, question_text, thinking_text, answer_text,
                    self.cfg.private_key,
                )
                # Embed + ingest + gossip off the hot path so the ShardResult
                # returns immediately (embedding adds tens of ms otherwise).
                asyncio.create_task(self._ingest_and_gossip(
                    job_id, model_id, question_text, thinking_text,
                    answer_text, proof_sig_hex,
                ))
            except Exception as exc:
                log.debug("thought_store_post_hook_err job=%s err=%s", job_id, exc)

        # Sign: keccak(shard_index || job_id || output)
        preimage  = (str(shard_idx) + job_id + output).encode("utf-8")
        signature = crypto_sign(self.cfg.private_key, preimage)

        return {
            "type":        "ShardResult",
            "job_id":      job_id,
            "shard_index": shard_idx,
            "miner":       self.address,
            "output":      output,
            "latency_ms":  elapsed_ms,
            "signature":   signature,
        }

    async def _ingest_and_gossip(
        self, job_id: str, model_id: str, question: str,
        thinking: str, answer: str, proof_sig_hex: str,
    ) -> None:
        """
        Embed the question, ingest the thought locally with its vector, and gossip
        it (vector included) so every replica's semantic index matches. Runs as a
        background task — never on the inference hot path.
        """
        try:
            embedding = await self._embedder.embed(question)
            await self._thought_store.ingest(
                job_id=job_id, miner_address=self.address, model_id=model_id,
                question=question, thinking=thinking, answer=answer,
                proof_sig_hex=proof_sig_hex, block_number=None, tx_hash=None,
                peer_origin=None,
            )
            if embedding:
                await self._thought_store.set_embedding(job_id, embedding)
            await self._p2p.broadcast(TOPICS["thought_broadcast"], {
                "type":          THOUGHT_BROADCAST,
                "job_id":        job_id,
                "miner_address": self.address,
                "model_id":      model_id,
                "question_text": question,
                "thinking_text": thinking,
                "answer_text":   answer,
                "proof_sig":     proof_sig_hex,
                "block_number":  0,
                "tx_hash":       "",
                "embedding":     embedding,
            })
            log.debug(
                "thought_ingest_gossip job=%s emb_dim=%d ans_chars=%d",
                job_id[:12], len(embedding), len(answer),
            )
        except Exception as exc:
            log.debug("ingest_and_gossip_err job=%s err=%s", job_id[:12], exc)

    # ── Model root registration ───────────────────────────────────────────────

    async def _register_model_roots(self) -> None:
        """
        Compute and register the Merkle root for each configured model.
        Retries every 30 s until all models are registered (handles chain restarts
        and transient RPC failures at startup).
        """
        from .model_registry import compute_model_root

        # Pre-compute roots once (CPU-bound, done outside the retry loop)
        roots: dict[str, tuple[str, int]] = {}
        for model_id, model_path in self.cfg.models.items():
            try:
                root, leaf_count = await asyncio.get_event_loop().run_in_executor(
                    None, compute_model_root, model_id, model_path
                )
                roots[model_id] = (root, leaf_count)
                log.info("model_root_computed model=%s root=%s leaves=%d",
                         model_id, root[:18] + "…", leaf_count)
            except Exception as exc:
                log.warning("model_root_compute_failed model=%s err=%s", model_id, exc)

        if not roots:
            return

        pending = set(roots.keys())
        while pending:
            for model_id in list(pending):
                root, leaf_count = roots[model_id]
                try:
                    tx_hash = await self._rpc_call("inft_registerModel", [
                        model_id, root, leaf_count, self.cfg.private_key,
                    ])
                    log.info("model_root_registered model=%s tx=%s",
                             model_id, str(tx_hash)[:18])
                    pending.discard(model_id)
                except Exception as exc:
                    log.warning("model_root_registration_failed model=%s err=%s — retrying in 30s",
                                model_id, exc)
            if pending:
                await asyncio.sleep(30)

    async def _rpc_call(self, method: str, params: list):
        """POST a JSON-RPC call to the L2 node and return the result."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.cfg.l2_rpc_url,
                json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                body = await resp.json()
        if "error" in body:
            raise RuntimeError(body["error"].get("message", str(body["error"])))
        return body.get("result")

    # ── Health server ─────────────────────────────────────────────────────────

    async def _health_server(self) -> None:
        async def handle_health(req: web.Request) -> web.Response:
            return web.json_response({
                "status":         "ok",
                "address":        self.address,
                "backend":        self.models.name,
                "models":         list(self.cfg.models.keys()),
                "active_shards":  len(self._active_shards),
                "max_shards":     self.cfg.max_concurrent_shards,
            })

        async def handle_metrics(req: web.Request) -> web.Response:
            return web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)

        app = web.Application()
        app.router.add_get("/health",  handle_health)
        app.router.add_get("/metrics", handle_metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.cfg.health_port)
        await site.start()
        log.info("health_server_started port=%d", self.cfg.health_port)
        # Keep running until shutdown
        await self._shutdown.wait()

    async def _stats_loop(self) -> None:
        tick = 0
        while not self._shutdown.is_set():
            try:
                balance = await self._l2_balance()
                INFT_BALANCE.set(balance)
                log.info(
                    "stats address=%s l2_inft=%.2f active_shards=%d backend=%s kv_cache=%s",
                    self.address, balance, len(self._active_shards),
                    self.models.name, self._kv_cache.stats(),
                )
                # Collect live_tps from pg_inft for each supported model.
                # The sequencer uses this to refine tensor-split allocations.
                live_tps: dict[str, float] = {}
                if self._thought_store is not None:
                    for m_id in self.cfg.models:
                        row = await self._thought_store.get_live_tps(self.address, m_id)
                        if row and row.get("live_tps") is not None:
                            live_tps[m_id] = round(float(row["live_tps"]), 4)

                await self._p2p.broadcast(TOPICS["heartbeats"], {
                    "type":            "MinerHeartbeat",
                    "address":         self.address,
                    "models":          list(self.cfg.models.keys()),
                    "backend":         self.models.name,
                    "tensor_parallel": bool(self._rpc_addr),
                    "rpc_addr":        self._rpc_addr,
                    "live_tps":        live_tps,
                })
                # Registry heartbeat every 60s
                if tick % 1 == 0:
                    await self._registry.heartbeat()
                # Phase 4: evict expired KV cache files every 10 minutes
                if tick % 10 == 0:
                    self._kv_cache.evict_expired()
            except Exception as exc:
                log.debug("stats_error err=%s", exc)
            tick += 1
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass

    async def _l2_balance(self) -> float:
        try:
            async with aiohttp.ClientSession() as session:
                body = {"jsonrpc": "2.0", "method": "inft_getAccount",
                        "params": [self.address], "id": 1}
                async with session.post(
                    self.cfg.l2_rpc_url, json=body,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    acc  = data.get("result", {})
                    # Return balance in INFT units (divide by 1e18)
                    return int(acc.get("balance_inft", 0)) / 1e18
        except Exception:
            return 0.0

    async def _drain_pending_offers(self) -> None:
        """Process queued offers now that capacity has freed up."""
        while self._pending_offers and len(self._active_shards) < self.cfg.max_concurrent_shards:
            offer = self._pending_offers.pop(0)
            log.info("shard_retry job=%s shard=%d (from queue)",
                     offer.get("job_id","?"), offer.get("spec",{}).get("shard_index",0))
            await self._on_shard_offer(offer)

    def stop(self) -> None:
        self._shutdown.set()
        asyncio.create_task(self._registry.stop())
        if self._rpc_proc and self._rpc_proc.poll() is None:
            self._rpc_proc.terminate()
            log.info("rpc_server_stopped pid=%d", self._rpc_proc.pid)
        if self._thought_store is not None:
            asyncio.create_task(self._thought_store.close())


# ── Entry point ───────────────────────────────────────────────────────────────

async def async_main(config_path: str) -> None:
    cfg = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not cfg.private_key:
        log.error("No private_key in config or PRIVATE_KEY env var")
        sys.exit(1)
    if not cfg.models:
        log.error("No models configured")
        sys.exit(1)

    key_dir  = Path(cfg.key_dir).expanduser()
    priv_pem, pub_der = keys.load_or_generate(key_dir)

    miner = L2Miner(cfg, priv_pem, pub_der)

    loop = asyncio.get_event_loop()

    def _handle(sig_name: str) -> None:
        log.info("signal_received sig=%s", sig_name)
        miner.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig.name: _handle(s))

    await miner.run()
    log.info("l2_miner_stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceChain L2 miner")
    parser.add_argument("--config", default="config_l2.json")
    args = parser.parse_args()
    asyncio.run(async_main(args.config))


if __name__ == "__main__":
    main()
