"""
Shard protocol — the core parallel inference orchestrator.

Called by the sequencer when a TX_JOB_POST is confirmed. Coordinates the
full shard lifecycle:

  1. VRF shard assignment using parent block hash
  2. ShardOffer dispatch over P2P
  3. ShardResult collection from P2P
  4. Mode-specific assembly
  5. TX_SHARD_COMMIT injection into mempool for each completed shard
  6. Timeout detection and slash + reassignment

The ShardProtocol is stateful but not persistent — on restart, in-flight jobs
are lost. Production v2 would checkpoint job_state to SQLite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional, List

from ..types import (
    ShardSpec, ShardResult, JobState, JobStatus, ShardStatus, ShardMode,
    TxType, Transaction,
)
from ..crypto import keccak256_hex
from ..genesis import CHAIN_DEFAULTS
from .vrf import select_miners, select_fallback
from .assembler import assemble
from .slash import build_slash_tx
from .modes.context_split import split_prompt as cs_split
from .modes.parallel_sample import split_prompt as ps_split
from .modes.speculative import split_prompt as spec_split
from .modes.pipeline_parallel import split_prompt as pp_split
from .context_cache import ContextKVCache

# Backend capability rank — higher = better suited for coordinator role
_BACKEND_RANK: dict[str, int] = {
    "cuda": 4, "metal": 3, "mlx": 3, "vulkan": 2, "llama_cpp": 1, "cpu": 0, "mock": -1,
}

log = logging.getLogger("chain.shard.protocol")


class ShardProtocol:
    def __init__(
        self,
        sequencer,          # Sequencer instance (for mempool access + state)
        p2p_node,           # P2PNode instance
        cfg: dict = None,
    ):
        self._seq  = sequencer
        self._p2p  = p2p_node
        self._cfg  = {**CHAIN_DEFAULTS, **(cfg or {})}

        # job_id → JobState  (in-memory; lost on restart)
        self._jobs: dict[str, JobState] = {}

        # job_id → set of timers (asyncio.Task handles for cleanup)
        self._timers: dict[str, list[asyncio.Task]] = {}

        # Miner backend capability registry — updated from heartbeat messages.
        # address.lower() → backend name ("metal", "vulkan", "cuda", …)
        self._miner_backends: dict[str, str] = {}
        # Set of miner addresses that advertise tensor_parallel support.
        self._tp_capable: set[str] = set()
        # address.lower() → "host:port" RPC address for pipeline parallel
        self._miner_rpc_addrs: dict[str, str] = {}
        # address.lower() → list of model IDs advertised in heartbeat
        self._miner_models: dict[str, list[str]] = {}
        # address.lower() → epoch-ms of last heartbeat
        self._miner_last_seen: dict[str, int] = {}
        # address.lower() → max_memory_gb advertised in heartbeat (0 = unlimited)
        self._miner_memory_gb: dict[str, int] = {}
        # address.lower() → {model_id: live_tps} — production EWMA from pg_inft,
        # reported in heartbeats.  Preferred over benchmark score when >=5 samples.
        self._miner_live_tps: dict[str, dict[str, float]] = {}

        # Context load pre-phase (Phase 3 / Option B).
        # job_id → {
        #   "miners": [addr, ...],           # miners assigned to context chunks
        #   "chunks": {addr: text},          # context slice per miner
        #   "context_hash": str,             # merkle root from assemble_context()
        #   "model_id": str,
        #   "results": {addr: result_dict},  # ContextLoadResult messages received
        #   "event": asyncio.Event,          # fires when all results arrive or timeout
        #   "all_cache_hit": bool,           # True once we know all miners hit cache
        # }
        self._ctx_load: dict[str, dict] = {}

        # Phase 4: KV cache manager (sequencer side tracks which jobs had cache hits)
        kv_dir = self._cfg.get("kv_cache_dir", "/tmp/inft_kv")
        kv_ttl = int(self._cfg.get("kv_cache_ttl_s", 3600))
        self._kv_cache = ContextKVCache(cache_dir=kv_dir, ttl_s=kv_ttl)

        # Proactive pre-fetch: set by node.py after thought_store is initialised.
        # Used in dispatch_job() to search pg_inft and stage context before
        # ContextLoadOffers are sent so miners find it in local postgres.
        self._thought_store = None

    # ── Context load pre-phase (Phase 3 / Option B) ───────────────────────────

    @staticmethod
    def _split_context(context_text: str, n: int) -> list[str]:
        """
        Split context_text into n chunks at Q&A boundaries (double-newline).

        Unlike prompt splitting, we do NOT overlap — each Q&A pair is whole
        and belongs to exactly one miner. This is the Option B design: miners
        independently process their chunk in parallel, no sequential chaining.

        For pipeline_parallel mode the coordinator (index 0) receives the full
        context so it can build the complete KV state; remaining shards get "".
        """
        if n <= 1 or not context_text:
            return [context_text] + [""] * (n - 1)

        # Split on double-newline (Q&A pair boundaries from assemble_context)
        pairs = [p for p in context_text.split("\n\n") if p.strip()]
        if not pairs:
            return [context_text] + [""] * (n - 1)

        # Distribute pairs round-robin across shards
        chunks: list[list[str]] = [[] for _ in range(n)]
        for i, pair in enumerate(pairs):
            chunks[i % n].append(pair)

        return ["\n\n".join(c) + ("\n\n" if c else "") for c in chunks]

    async def _context_load_phase(
        self,
        job_id:       str,
        miners:       list[str],
        context_text: str,
        context_hash: str,
        model_id:     str,
        mode:         str,
    ) -> dict[str, dict]:
        """
        Option B: broadcast ContextLoadOffer to all miners in parallel.
        Each miner gets its own context chunk and independently pre-loads it.
        Returns {miner_addr: ContextLoadResult} for all respondents.

        For pipeline_parallel: coordinator (miners[0]) gets the full context;
        workers get empty string (their contribution is layer compute, not context).
        """
        if not self._cfg.get("context_load_enabled", True):
            return {}

        n = len(miners)
        if mode == ShardMode.PIPELINE_PARALLEL:
            # Coordinator owns the full context KV; workers confirm with empty chunk
            chunks = [context_text] + [""] * (n - 1)
        else:
            chunks = self._split_context(context_text, n)

        state_entry: dict = {
            "miners":       miners,
            "chunks":       {m.lower(): chunks[i] for i, m in enumerate(miners)},
            "context_hash": context_hash,
            "model_id":     model_id,
            "results":      {},
            "event":        asyncio.Event(),
            "all_cache_hit":False,
        }
        self._ctx_load[job_id] = state_entry

        # Broadcast all offers simultaneously (Option B — fully parallel)
        for i, (miner, chunk) in enumerate(zip(miners, chunks)):
            offer = {
                "type":           "ContextLoadOffer",
                "job_id":         job_id,
                "context_hash":   context_hash,
                "shard_index":    i,
                "total_shards":   n,
                "assigned_miner": miner,
                "context_slice":  chunk,
                "model_id":       model_id,
            }
            await self._p2p.broadcast("context_load_offers", offer)

        log.info(
            "ctx_load_offered job=%s mode=%s miners=%d context_chars=%d",
            job_id[:12], mode, n, len(context_text),
        )

        timeout_s = self._cfg.get("context_load_timeout_ms", 8_000) / 1000.0
        try:
            await asyncio.wait_for(state_entry["event"].wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            got = len(state_entry["results"])
            log.warning(
                "ctx_load_timeout job=%s got=%d/%d — proceeding with partial confirmation",
                job_id[:12], got, n,
            )

        results = state_entry["results"]
        state_entry["all_cache_hit"] = (
            len(results) == n and all(r.get("cache_hit") for r in results.values())
        )
        log.info(
            "ctx_load_complete job=%s confirmed=%d/%d all_cache_hit=%s",
            job_id[:12], len(results), n, state_entry["all_cache_hit"],
        )

        # Inject CONTEXT_LOAD_COMMIT txs for each respondent
        for miner_addr, result in results.items():
            tx = self._build_context_load_commit_tx(job_id, miner_addr, result)
            await self._seq.mempool.add(tx)

        return results

    async def on_context_load_result(self, msg: dict) -> None:
        """
        Called by P2P node when a ContextLoadResult arrives from a miner.
        Records the result and fires the event when all miners have responded.
        """
        job_id     = msg.get("job_id", "")
        miner      = msg.get("miner", "").lower()
        shard_idx  = int(msg.get("shard_index", -1))
        chunk_hash = msg.get("chunk_hash", "")
        cache_hit  = bool(msg.get("cache_hit", False))
        latency_ms = int(msg.get("latency_ms", 0))
        signature  = msg.get("signature", "")

        state_entry = self._ctx_load.get(job_id)
        if state_entry is None:
            return  # result arrived after timeout cleanup — ignore

        expected_miners = [m.lower() for m in state_entry["miners"]]
        if miner not in expected_miners:
            log.warning("ctx_load_unexpected_miner job=%s miner=%s", job_id[:12], miner[:10])
            return

        # Verify signature over (job_id || shard_index || chunk_hash)
        from ..crypto import verify_sig
        preimage = (job_id + str(shard_idx) + chunk_hash).encode("utf-8")
        if signature and not verify_sig(preimage, signature, miner):
            log.warning("ctx_load_bad_sig job=%s miner=%s", job_id[:12], miner[:10])
            return

        state_entry["results"][miner] = {
            "shard_index": shard_idx,
            "chunk_hash":  chunk_hash,
            "cache_hit":   cache_hit,
            "latency_ms":  latency_ms,
        }

        if cache_hit:
            # Phase 4: record which (context_hash, model_id) are warm on this miner
            self._kv_cache.register(
                state_entry["context_hash"],
                state_entry["model_id"],
                f"miner:{miner}",   # symbolic — actual path lives on miner side
            )

        log.info(
            "ctx_load_result job=%s shard=%d miner=%s cache_hit=%s chunk_hash=%s",
            job_id[:12], shard_idx, miner[:10], cache_hit, chunk_hash[:12],
        )

        # Fire event when all assigned miners have responded
        if len(state_entry["results"]) >= len(state_entry["miners"]):
            state_entry["event"].set()

    def _build_context_load_commit_tx(
        self,
        job_id:  str,
        miner:   str,
        result:  dict,
    ) -> "Transaction":
        payload = json.dumps({
            "job_id":       job_id,
            "shard_index":  result.get("shard_index", 0),
            "miner":        miner,
            "chunk_hash":   result.get("chunk_hash", ""),
            "cache_hit":    result.get("cache_hit", False),
            "latency_ms":   result.get("latency_ms", 0),
            "block_number": 0,  # filled by state on apply
        }, separators=(",", ":"))

        tx_hash = keccak256_hex(
            ("CONTEXT_LOAD_COMMIT" + job_id + miner + str(result.get("shard_index", 0))).encode()
        )
        return Transaction(
            tx_type=TxType.CONTEXT_LOAD_COMMIT,
            sender="",
            nonce=0,
            payload=payload,
            gas_price=0,
            signature="",
            tx_hash=tx_hash,
        )

    def get_context_load_state(self, job_id: str) -> dict:
        """Return current context load state for a job (for RPC introspection)."""
        entry = self._ctx_load.get(job_id, {})
        if not entry:
            return {}
        return {
            "miners":        entry["miners"],
            "confirmed":     list(entry["results"].keys()),
            "all_cache_hit": entry["all_cache_hit"],
            "context_hash":  entry.get("context_hash", ""),
        }

    # ── Phase 0 — Job dispatch ────────────────────────────────────────────────

    async def dispatch_job(self, job_payload: dict, block, state) -> None:
        """
        Called by sequencer when a TX_JOB_POST transaction lands in a block.
        Assigns shards via VRF and broadcasts ShardOffer messages.
        """
        job_id   = job_payload["job_id"]
        mode     = job_payload["shard_mode"]
        n_shards = int(job_payload["n_shards"])
        prompt   = job_payload.get("prompt", "")
        max_tok  = int(job_payload["max_tokens"])

        if n_shards == 1 and mode not in (ShardMode.TENSOR_PARALLEL, ShardMode.PIPELINE_PARALLEL):
            mode = ShardMode.PARALLEL_SAMPLE  # single-shard falls back to simplest mode

        worker_rpc_addrs: list[str] = []   # populated only for PIPELINE_PARALLEL
        ts_fracs:         list[float] = [] # populated only for PIPELINE_PARALLEL

        # Split prompt based on mode
        if mode == ShardMode.CONTEXT_SPLIT:
            slices = cs_split(prompt, n_shards)
        elif mode == ShardMode.SPECULATIVE:
            slices = spec_split(prompt, 2)
            n_shards = 2
        elif mode in (ShardMode.TENSOR_PARALLEL, ShardMode.PIPELINE_PARALLEL):
            # Handled after VRF miner selection.
            slices = None   # filled below
        else:
            slices = ps_split(prompt, n_shards)

        # Prefer miners that have registered the requested model on-chain.
        # Fall back to all staked validators when fewer registered miners exist
        # than shards requested — otherwise a partially-registered network would
        # silently reduce n_shards to 1.
        model_id   = job_payload["model_id"]
        all_validators = state.active_validators()
        registered = [
            (addr, stake) for addr, stake in all_validators
            if state.get_model_root(addr, model_id) is not None
        ]

        if len(registered) >= n_shards:
            validators = registered
        else:
            validators = all_validators
            if registered:
                log.info(
                    "model_root_partial job=%s model=%s registered=%d need=%d "
                    "— using all %d staked validators",
                    job_id, model_id, len(registered), n_shards, len(validators),
                )
            else:
                log.info(
                    "model_root_unregistered job=%s model=%s — using all %d validators",
                    job_id, model_id, len(validators),
                )

        miners = select_miners(job_id, n_shards, block.header.parent_hash, validators)

        if not miners:
            log.warning("no_miners_available job=%s", job_id)
            return

        # Pipeline parallel (also handles legacy tensor_parallel mode name):
        # Use llama.cpp RPC to distribute model layers across miners.
        # Shard 0 = coordinator (loads GGUF + calls llama-cli --rpc; does NOT need
        #   its own rpc-server running — only workers do).
        # Shard 1..N-1 = workers (rpc-server sidecar, expose GPU/CPU memory).
        if mode in (ShardMode.PIPELINE_PARALLEL, ShardMode.TENSOR_PARALLEL):
            mode = ShardMode.PIPELINE_PARALLEL   # normalise legacy name

            # Workers must advertise rpc_addr. Coordinator can be any miner.
            worker_capable = [m for m in miners if self._miner_rpc_addrs.get(m.lower())]
            if not worker_capable:
                log.warning(
                    "pp_fallback job=%s rpc_workers=0/%d — using parallel_sample "
                    "(at least one miner needs rpc_addr in heartbeat)",
                    job_id, len(miners),
                )
                mode   = ShardMode.PARALLEL_SAMPLE
                slices = ps_split(prompt, len(miners))
            else:
                # Sort by benchmark score (primary) then backend rank (secondary).
                # Highest-scoring miner becomes coordinator — it does the most work.
                all_sorted = sorted(
                    miners,
                    key=lambda m: self._miner_sort_key(m.lower(), model_id, state),
                    reverse=True,
                )
                # Prefer a non-worker as coordinator so we don't "waste" a worker
                # slot on a miner that could be contributing RPC layer compute.
                non_workers = [m for m in all_sorted
                               if not self._miner_rpc_addrs.get(m.lower())]
                if non_workers:
                    coordinator = non_workers[0]
                    workers = [m for m in all_sorted
                               if m != coordinator
                               and self._miner_rpc_addrs.get(m.lower())]
                else:
                    # All miners are worker-capable; highest score becomes coordinator.
                    coordinator = all_sorted[0]
                    workers = [m for m in all_sorted[1:]
                               if self._miner_rpc_addrs.get(m.lower())]

                miners = [coordinator] + workers[: n_shards - 1]
                n_shards = len(miners)
                slices = pp_split(prompt, n_shards)

                # Compute score-proportional layer fractions for tensor split.
                # coordinator index 0 gets the largest fraction (it's the fastest).
                ts_fracs = self._compute_tensor_split(miners, model_id, state)

                # Build the coordinator's rpc_peers list (all workers' RPC addresses).
                worker_rpc_addrs = [
                    self._miner_rpc_addrs[m.lower()]
                    for m in miners[1:]
                    if self._miner_rpc_addrs.get(m.lower())
                ]
                log.info(
                    "pp_pipeline job=%s stages=%d coordinator=%s workers=%s "
                    "rpc_peers=%s tensor_split=%s",
                    job_id, n_shards, miners[0][:10],
                    [m[:10] for m in miners[1:]],
                    worker_rpc_addrs,
                    [f"{f:.2f}" for f in ts_fracs],
                )

        if slices is None:
            slices = ps_split(prompt, n_shards)   # safety fallback

        # ── Proactive Parallel Pre-fetch ───────────────────────────────────────
        # Search pg_inft for thoughts relevant to this job's prompt NOW — before
        # the ContextLoadOffer is sent — so miners find the assembled context in
        # their LOCAL postgres replica instead of running a sequential search on
        # the inference hot path.
        #
        # Priority: user-supplied context_text > proactively fetched context.
        context_text = job_payload.get("context_text", "")
        context_hash = job_payload.get("context_hash", "")
        if not context_text and self._thought_store is not None and prompt:
            try:
                prior = await self._thought_store.search(prompt, model_id, 10)
                if prior:
                    parts = []
                    for r in prior:
                        q = (r.question_text or "").strip()
                        a = (r.answer_text or "").strip()
                        if q or a:
                            parts.append(f"Q: {q}\nA: {a}")
                    context_text = "\n\n".join(parts)
                    context_hash = keccak256_hex(context_text.encode("utf-8")) if context_text else ""
                    log.info(
                        "prefetch_context job=%s entries=%d ctx_chars=%d",
                        job_id[:12], len(prior), len(context_text),
                    )
            except Exception as _pf_exc:
                log.debug("prefetch_context_err job=%s err=%s", job_id[:12], _pf_exc)

        # Stage the assembled context in postgres so every miner can read it
        # from their local replica via get_job_context(job_id) during inference.
        if context_text and self._thought_store is not None:
            try:
                await self._thought_store.set_job_context(
                    job_id=job_id,
                    query_text=prompt[:512],
                    context_text=context_text,
                    context_hash=context_hash,
                    model_id=model_id,
                    n_entries=len(context_text.split("\n\n")),
                )
            except Exception as _sc_exc:
                log.debug("set_job_context_err job=%s err=%s", job_id[:12], _sc_exc)

        # ── Phase 3: Context load pre-phase (Option B — parallel) ─────────────
        # Miners receive their context chunk via ContextLoadOffer; they also write
        # the full context to their local postgres so _run_inference() can use it
        # without a blocking DB search.
        from ..crypto import ZERO_HASH
        if (
            context_text
            and context_hash
            and context_hash != ZERO_HASH
            and miners
        ):
            await self._context_load_phase(
                job_id=job_id,
                miners=miners,
                context_text=context_text,
                context_hash=context_hash,
                model_id=model_id,
                mode=mode,
            )

        # Build job state
        job = JobState(
            job_id=job_id,
            requester=job_payload.get("sender", ""),
            model_id=job_payload["model_id"],
            prompt=prompt,
            mode=mode,
            n_shards=len(miners),
            max_tokens=max_tok,
            fee_inft=int(job_payload.get("fee_inft", 0)),
            block_number=block.header.block_number,
            deadline_ms=int(time.time() * 1000) + self._cfg["assembly_timeout_ms"],
            original_prompt=job_payload.get("original_prompt", prompt),
            context_hash=job_payload.get("context_hash") or None,
            context_entries=int(job_payload.get("context_entries", 0)),
        )

        # Pipeline parallel is sequential through stages — give extra time.
        shard_timeout = self._cfg["shard_result_timeout_ms"]
        if mode == ShardMode.PIPELINE_PARALLEL:
            shard_timeout = shard_timeout * n_shards

        # Build specs and initial status
        import json as _json
        for i, (miner, prompt_slice) in enumerate(zip(miners, slices)):
            if mode == ShardMode.PIPELINE_PARALLEL:
                role = "coordinator" if i == 0 else "worker"
                # Coordinator gets the list of worker RPC addresses; workers get empty.
                rpc_peers_val = _json.dumps(worker_rpc_addrs) if i == 0 else "[]"
                rpc_addr_val  = self._miner_rpc_addrs.get(miner.lower(), "")
                # Include each worker's memory budget so coordinator can set tensor-split.
                if i == 0:
                    worker_mem = [
                        self._miner_memory_gb.get(m.lower(), 0)
                        for m in miners[1:]
                    ]
                    rpc_memory_val   = _json.dumps(worker_mem)
                    # Tensor split: coordinator gets fracs[0], workers get fracs[1..].
                    # Only the coordinator needs this — it passes -ts to llama-cli.
                    tensor_split_val = _json.dumps(ts_fracs) if ts_fracs else ""
                else:
                    rpc_memory_val   = ""
                    tensor_split_val = ""
            else:
                role             = ""
                rpc_peers_val    = ""
                rpc_addr_val     = ""
                rpc_memory_val   = ""
                tensor_split_val = ""

            spec = ShardSpec(
                shard_index=i,
                total_shards=len(miners),
                mode=mode,
                assigned_miner=miner,
                prompt_slice=prompt_slice,
                max_tokens=max_tok if mode != ShardMode.CONTEXT_SPLIT else max_tok // len(miners),
                timeout_ms=shard_timeout,
                backend_hint=self._miner_backends.get(miner.lower(), ""),
                role=role,
                rpc_peers=rpc_peers_val,
                rpc_addr=rpc_addr_val,
                rpc_memory_gb=rpc_memory_val,
                tensor_split=tensor_split_val,
            )
            job.specs[i]        = spec
            job.shard_status[i] = ShardStatus.OFFERED

        self._jobs[job_id] = job
        self._timers[job_id] = []

        # Broadcast ShardOffer for each shard
        for i, spec in job.specs.items():
            offer = {
                "type":       "ShardOffer",
                "job_id":     job_id,
                "model_id":   job.model_id,
                "spec":       spec.to_dict(),
                "fee_share":  job.fee_inft // job.n_shards,
                "requester":  job.requester,
            }
            await self._p2p.broadcast("shard_offers", offer)

            # Schedule timeout for offer acceptance
            t = asyncio.create_task(
                self._offer_timeout(job_id, i, spec.assigned_miner,
                                    self._cfg["shard_offer_timeout_ms"] / 1000)
            )
            self._timers[job_id].append(t)

        log.info(
            "job_dispatched job=%s mode=%s n_shards=%d miners=%s",
            job_id, mode, len(miners), [m[:10] for m in miners],
        )

    # ── Phase 1 — Miner result arrival ────────────────────────────────────────

    async def on_shard_result(self, msg: dict) -> None:
        """
        Called by p2p.node when a ShardResultMsg arrives.
        Validates the result, records it, and triggers assembly when ready.
        """
        job_id     = msg.get("job_id", "")
        shard_idx  = int(msg.get("shard_index", -1))
        miner      = msg.get("miner", "")
        output     = msg.get("output", "")
        latency_ms = int(msg.get("latency_ms", 0))
        signature  = msg.get("signature", "")

        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.COMPLETE:
            return

        spec = job.specs.get(shard_idx)
        if spec is None:
            log.warning("unknown_shard job=%s shard=%d", job_id, shard_idx)
            return

        # Verify miner assignment
        if miner.lower() != spec.assigned_miner.lower():
            log.warning(
                "wrong_miner job=%s shard=%d expected=%s got=%s",
                job_id, shard_idx, spec.assigned_miner[:10], miner[:10],
            )
            return

        # Verify signature over (shard_index || job_id || output)
        from ..crypto import verify_sig
        preimage = (str(shard_idx) + job_id + output).encode("utf-8")
        if signature and not verify_sig(preimage, signature, miner):
            log.warning("bad_shard_sig job=%s shard=%d miner=%s", job_id, shard_idx, miner[:10])
            return

        result = ShardResult(
            shard_index=shard_idx,
            miner=miner,
            output=output,
            latency_ms=latency_ms,
            signature=signature,
        )
        job.results[shard_idx]      = result
        job.shard_status[shard_idx] = ShardStatus.SUBMITTED

        log.info(
            "shard_received job=%s shard=%d miner=%s latency=%dms mode=%s",
            job_id, shard_idx, miner[:10], latency_ms, job.mode,
        )

        if job.ready_to_assemble():
            await self._assemble(job_id)

    # ── Phase 2 — Assembly ────────────────────────────────────────────────────

    async def _assemble(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.COMPLETE:
            return
        job.status = JobStatus.ASSEMBLING

        results_ordered = [
            job.results[i]
            for i in sorted(job.results.keys())
        ]

        final_output, output_hash = assemble(job.mode, results_ordered)
        job.final_output = final_output
        job.output_hash  = output_hash
        job.status       = JobStatus.COMPLETE

        # Cancel pending timeouts for this job
        for t in self._timers.get(job_id, []):
            t.cancel()

        # Inject TX_SHARD_COMMIT for each shard into the mempool
        for shard_result in results_ordered:
            tx = self._build_shard_commit_tx(job_id, shard_result, output_hash, job)
            await self._seq.mempool.add(tx)

        # Inject TX_HISTORY_COMMIT to record this Q&A in per-wallet history
        if job.requester:
            history_tx = self._build_history_commit_tx(job_id, job, output_hash, final_output)
            await self._seq.mempool.add(history_tx)

        log.info(
            "job_assembled job=%s mode=%s output_len=%d hash=%s",
            job_id, job.mode, len(final_output), output_hash[:16],
        )

    def _build_shard_commit_tx(
        self,
        job_id:      str,
        result:      ShardResult,
        output_hash: str,
        job:         JobState,
    ) -> Transaction:
        payload = json.dumps({
            "job_id":         job_id,
            "shard_index":    result.shard_index,
            "output_hash":    output_hash,
            "miner":          result.miner,
            "latency_ms":     result.latency_ms,
            "miner_sig":      result.signature,
            "output_preview": result.output[:200],   # stored on-chain for quick reads
            "fee_share":      job.fee_inft // job.n_shards,
        }, separators=(",", ":"))

        tx_hash = keccak256_hex(
            ("SHARD_COMMIT" + job_id + str(result.shard_index) + result.miner).encode()
        )
        return Transaction(
            tx_type=TxType.SHARD_COMMIT,
            sender="",
            nonce=0,
            payload=payload,
            gas_price=0,
            signature="",
            tx_hash=tx_hash,
        )

    def _build_history_commit_tx(
        self,
        job_id:       str,
        job:          JobState,
        output_hash:  str,
        final_output: str,
    ) -> Transaction:
        # Use the original user prompt (without context prefix) so history entries
        # don't accumulate an ever-growing context snowball across restarts.
        stored_prompt = job.original_prompt or job.prompt
        prompt_hash = keccak256_hex(stored_prompt.encode("utf-8"))
        payload = json.dumps({
            "job_id":       job_id,
            "wallet":       job.requester,
            "model_id":     job.model_id,
            "prompt_hash":  prompt_hash,
            "output_hash":  output_hash,
            "prompt":       stored_prompt,
            "output":       final_output,
            "timestamp":    int(time.time() * 1000),
            "block_number": job.block_number,
        }, separators=(",", ":"))

        tx_hash = keccak256_hex(
            ("HISTORY_COMMIT" + job_id + job.requester).encode()
        )
        return Transaction(
            tx_type=TxType.HISTORY_COMMIT,
            sender="",
            nonce=0,
            payload=payload,
            gas_price=0,
            signature="",
            tx_hash=tx_hash,
        )

    # ── Phase 3 — Timeouts and slash ─────────────────────────────────────────

    async def _offer_timeout(
        self, job_id: str, shard_idx: int, miner: str, delay_s: float
    ) -> None:
        await asyncio.sleep(delay_s)
        job = self._jobs.get(job_id)
        if job is None or shard_idx in job.results:
            return  # Result arrived in time

        if job.shard_status.get(shard_idx) == ShardStatus.OFFERED:
            log.warning("offer_timeout job=%s shard=%d miner=%s — retrying with all available miners",
                        job_id, shard_idx, miner[:10])
            job.shard_status[shard_idx] = ShardStatus.TIMEOUT

            # Re-broadcast to ALL active miners (not just VRF winner) so a miner
            # that has since freed capacity can pick it up. No slash — the miner
            # may have been legitimately at capacity.
            await self._rebroadcast_shard(job_id, shard_idx)

            # Schedule a follow-up result timeout so the job eventually fails
            # if the re-broadcast also goes unanswered.
            t = asyncio.create_task(
                self._result_timeout(
                    job_id, shard_idx, miner,
                    self._cfg["shard_result_timeout_ms"] / 1000,
                )
            )
            self._timers.setdefault(job_id, []).append(t)

    async def _result_timeout(
        self, job_id: str, shard_idx: int, miner: str, delay_s: float
    ) -> None:
        await asyncio.sleep(delay_s)
        job = self._jobs.get(job_id)
        if job is None or shard_idx in job.results:
            return

        log.warning("result_timeout job=%s shard=%d miner=%s", job_id, shard_idx, miner[:10])
        job.shard_status[shard_idx] = ShardStatus.TIMEOUT

        # Hard slash (30%) — miner accepted but never returned a result
        slash_tx = build_slash_tx(miner, job_id, shard_idx,
                                  self._seq.chain_id, hard=True)
        await self._seq.mempool.add(slash_tx)
        await self._reassign_shard(job_id, shard_idx, miner)

    async def _rebroadcast_shard(self, job_id: str, shard_idx: int) -> None:
        """Re-broadcast an offered shard to give the assigned miner another chance."""
        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.COMPLETE:
            return
        spec = job.specs.get(shard_idx)
        if spec is None:
            return
        offer = {
            "type":        "ShardOffer",
            "job_id":      job_id,
            "model_id":    job.model_id,
            "spec":        spec.to_dict(),
            "fee_share":   job.fee_inft // job.n_shards,
            "requester":   job.requester,
            "rebroadcast": True,
        }
        await self._p2p.broadcast("shard_offers", offer)
        log.info("shard_rebroadcast job=%s shard=%d miner=%s",
                 job_id, shard_idx, spec.assigned_miner[:10])

    async def _reassign_shard(
        self, job_id: str, failed_shard_idx: int, failed_miner: str
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.status == JobStatus.COMPLETE:
            return

        used = {
            spec.assigned_miner
            for spec in job.specs.values()
        }
        validators = self._seq.state().active_validators()
        head_hash  = self._seq.head().block_hash

        fallback = select_fallback(
            job_id=job_id,
            shard_idx=failed_shard_idx,
            block_hash=head_hash,
            validators=validators,
            previously_used=used,
        )
        if fallback is None:
            log.error("no_fallback_miner job=%s shard=%d — marking job failed", job_id, failed_shard_idx)
            job.status = JobStatus.FAILED
            return

        old_spec = job.specs[failed_shard_idx]
        new_spec = ShardSpec(
            shard_index=old_spec.shard_index,
            total_shards=old_spec.total_shards,
            mode=old_spec.mode,
            assigned_miner=fallback,
            prompt_slice=old_spec.prompt_slice,
            max_tokens=old_spec.max_tokens,
            timeout_ms=old_spec.timeout_ms,
        )
        job.specs[failed_shard_idx]        = new_spec
        job.shard_status[failed_shard_idx] = ShardStatus.OFFERED

        offer = {
            "type":      "ShardOffer",
            "job_id":    job_id,
            "model_id":  job.model_id,
            "spec":      new_spec.to_dict(),
            "fee_share": job.fee_inft // job.n_shards,
            "requester": job.requester,
            "reassign":  True,
        }
        await self._p2p.broadcast("shard_offers", offer)

        t = asyncio.create_task(
            self._result_timeout(
                job_id, failed_shard_idx, fallback,
                self._cfg["shard_result_timeout_ms"] / 1000,
            )
        )
        self._timers.setdefault(job_id, []).append(t)

        log.info(
            "shard_reassigned job=%s shard=%d old=%s new=%s",
            job_id, failed_shard_idx, failed_miner[:10], fallback[:10],
        )

    # ── Benchmark-score helpers ───────────────────────────────────────────────

    _MIN_LIVE_SAMPLES = 5   # samples needed before trusting live_tps over bench_tps

    def _effective_tps(self, addr: str, model_id: str, state) -> Optional[float]:
        """
        Return the most reliable TPS estimate for (addr, model_id).

        Priority:
          1. live_tps from recent heartbeat (production EWMA, >=_MIN_LIVE_SAMPLES)
          2. benchmark score from chain state (StateDB)
          3. None (miner gets minimum layer fraction)

        Live TPS is preferred because it reflects actual production conditions
        rather than an isolated benchmark that could differ from real workloads.
        """
        addr_lower = addr.lower()
        live_map   = self._miner_live_tps.get(addr_lower, {})
        live_entry = live_map.get(model_id)
        if live_entry is not None:
            return float(live_entry)

        score = state.get_miner_score(addr, model_id) if state else None
        if score is not None:
            return float(score["tokens_per_sec"])

        return None

    def _miner_sort_key(self, addr: str, model_id: str, state) -> tuple:
        """
        Return (effective_tps, backend_rank) for sorting miners.
        Miners with no score get (0, backend_rank) — eligible as workers but
        deprioritised for the coordinator role.
        """
        tps  = self._effective_tps(addr, model_id, state) or 0.0
        rank = _BACKEND_RANK.get(self._miner_backends.get(addr.lower(), ""), 0)
        return (tps, rank)

    def _compute_tensor_split(
        self, miners: list[str], model_id: str, state
    ) -> list[float]:
        """
        Compute tensor-split layer fractions proportional to each miner's
        effective TPS (live production EWMA when available, else benchmark score).

        Miners with no score receive min_layer_frac.
        Result is clamped to [min_layer_frac, max_layer_frac] and renormalised
        so the fractions sum to 1.0.

        Example with Mac (22 t/s live) + Khadas (2 t/s bench):
          raw  → [0.917, 0.083]
          clamp→ [0.80,  0.083]   (Mac capped at 80%)
          norm → [0.906, 0.094]   (Khadas gets ~9%)
        """
        min_frac = float(self._cfg.get("min_layer_frac", 0.05))
        max_frac = float(self._cfg.get("max_layer_frac", 0.80))
        n        = len(miners)

        scores: list[Optional[float]] = [
            self._effective_tps(addr, model_id, state) for addr in miners
        ]

        known = [s for s in scores if s is not None]
        if not known:
            return [1.0 / n] * n

        avg_known = sum(known) / len(known)
        filled    = [s if s is not None else avg_known * min_frac for s in scores]

        total  = sum(filled) or 1.0
        raw    = [s / total for s in filled]
        clamped = [max(min_frac, min(max_frac, f)) for f in raw]
        ct      = sum(clamped) or 1.0
        return [f / ct for f in clamped]

    # ── Miner capability registry ─────────────────────────────────────────────

    async def on_miner_heartbeat(self, msg: dict) -> None:
        """
        Called for every MinerHeartbeat P2P message.
        Tracks backend type, RPC address, pipeline-parallel capability, and supported models.
        """
        addr        = msg.get("address", "").lower()
        backend     = msg.get("backend", "")
        rpc_addr    = msg.get("rpc_addr", "")   # "host:port" of this miner's rpc-server
        models      = msg.get("models", [])
        tp_ok       = bool(msg.get("tensor_parallel", False)) or bool(rpc_addr)
        memory_gb   = int(msg.get("max_memory_gb", 0))
        # live_tps: {model_id: tps} dict reported from pg_inft production tracking
        live_tps    = msg.get("live_tps", {})
        if not addr:
            return
        if backend:
            self._miner_backends[addr] = backend
        if rpc_addr:
            self._miner_rpc_addrs[addr] = rpc_addr
        if isinstance(models, list) and models:
            self._miner_models[addr] = models
        if memory_gb > 0:
            self._miner_memory_gb[addr] = memory_gb
        if isinstance(live_tps, dict) and live_tps:
            self._miner_live_tps[addr] = {
                k: float(v) for k, v in live_tps.items() if v is not None
            }
        if tp_ok:
            self._tp_capable.add(addr)
        else:
            self._tp_capable.discard(addr)
        self._miner_last_seen[addr] = int(time.time() * 1000)
        log.info(
            "heartbeat addr=%s backend=%s models=%d rpc_addr=%s mem_gb=%d live_tps=%s",
            addr[:10], backend, len(models), rpc_addr or "none", memory_gb,
            {k: f"{v:.1f}" for k, v in (live_tps or {}).items()},
        )

    def active_miners(self, max_age_ms: int = 300_000) -> list[dict]:
        """
        Return all miners seen via heartbeat within max_age_ms (default 5 min).
        Each entry includes address, backend, models list, and last_seen timestamp.
        """
        cutoff = int(time.time() * 1000) - max_age_ms
        result = []
        for addr, last_seen in self._miner_last_seen.items():
            if last_seen < cutoff:
                continue
            result.append({
                "address":   addr,
                "backend":   self._miner_backends.get(addr, ""),
                "models":    self._miner_models.get(addr, []),
                "rpc_addr":  self._miner_rpc_addrs.get(addr, ""),
                "last_seen": last_seen,
                "active":    True,
                "maxShards": 4,
            })
        return result

    # ── Compounding inference — chained job dispatch ──────────────────────────

    async def dispatch_chained_job(self, job_id: str, block, state) -> None:
        """
        Dispatch a job that was WAITING on a parent and just became PENDING.

        The prompt template is resolved here using the parent's full assembled
        output, which lives in the shard protocol's in-memory _jobs dict.  If
        the parent entry is missing (e.g. after a node restart), the output
        previews stored on-chain are concatenated as a fallback.
        """
        job = state.job(job_id)
        if job is None or job.status != JobStatus.PENDING:
            return

        # Resolve {prev_output} from the parent's full assembled output.
        parent_output = ""
        if job.parent_job_id:
            parent_proto = self._jobs.get(job.parent_job_id)
            if parent_proto and parent_proto.final_output is not None:
                parent_output = parent_proto.final_output
            else:
                # Fallback: stitch together the on-chain output previews.
                parent_state = state.job(job.parent_job_id)
                if parent_state:
                    previews = [
                        r.output
                        for r in sorted(parent_state.results.values(),
                                        key=lambda r: r.shard_index)
                    ]
                    parent_output = " ".join(previews)
                log.warning(
                    "dispatch_chained_job: using preview fallback for parent=%s",
                    job.parent_job_id[:12],
                )

        template = job.prompt_template or job.prompt
        resolved_prompt = template.replace("{prev_output}", parent_output)

        payload = {
            "job_id":        job.job_id,
            "model_id":      job.model_id,
            "prompt":        resolved_prompt,
            "max_tokens":    job.max_tokens,
            "shard_mode":    job.mode,
            "n_shards":      job.n_shards,
            "fee_inft":      job.fee_inft,
            "timeout_ms":    35_000,
            "sender":        job.requester,
            "parent_job_id": job.parent_job_id,
        }
        log.info(
            "chained_job_dispatch job=%s step=%d parent=%s prompt_len=%d",
            job_id[:12], job.chain_step,
            (job.parent_job_id or "")[:12], len(resolved_prompt),
        )
        await self.dispatch_job(payload, block, state)

    # ── Public state queries ──────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def active_jobs(self) -> list[str]:
        return [
            jid for jid, j in self._jobs.items()
            if j.status not in (JobStatus.COMPLETE, JobStatus.FAILED)
        ]

    async def recover_pending_jobs(self) -> None:
        """
        Re-dispatch any jobs the state knows about that the shard protocol lost
        (e.g. after a miner reconnect or node restart). Checks the sequencer state
        for jobs marked 'pending' that are not currently tracked by the protocol.
        """
        state_jobs = self._seq.state().all_jobs()
        recovered = 0
        for job_id, job_rec in state_jobs.items():
            if job_rec.status != "pending" or job_id in self._jobs:
                continue
            log.info("recovery_redispatch job=%s chained=%s",
                     job_id[:12], bool(job_rec.parent_job_id))
            if job_rec.parent_job_id:
                # Chained job — let dispatch_chained_job resolve the prompt.
                asyncio.create_task(
                    self.dispatch_chained_job(job_id, self._seq.head(), self._seq.state())
                )
            else:
                payload = {
                    "job_id":     job_id,
                    "model_id":   job_rec.model_id,
                    "prompt":     job_rec.prompt,
                    "max_tokens": job_rec.max_tokens,
                    "shard_mode": job_rec.mode,
                    "n_shards":   job_rec.n_shards,
                    "fee_inft":   job_rec.fee_inft,
                    "timeout_ms": 35_000,
                }
                asyncio.create_task(
                    self.dispatch_job(payload, self._seq.head(), self._seq.state())
                )
            recovered += 1
        if recovered:
            log.info("recovery_complete redispatched=%d", recovered)
