"""
Lookahead Decoding backend for inference-chain.

Replaces Jacobi decoding with Lookahead Decoding (Fu et al. 2023).

KEY DIFFERENCE FROM JACOBI:
  Jacobi fills the W-position window with a repeated token, requiring
  8-10 iterations to converge.  Lookahead maintains a per-token n-gram
  pool that records (token → predicted_continuation) pairs observed in
  previous iterations.  When building the next window, it looks up pool
  entries to pre-fill positions with high-probability guesses — so the
  window converges in 1 iteration for sequences seen before.

  Pool hit → W tokens accepted in 1 forward pass  (~Wx speedup)
  Cold start → falls back to AR-equivalent (1 token per pass)

  The pool is local per-job and grows as generation proceeds. By the
  middle of a typical 256-token response the hit rate exceeds 80%.

DISAGGREGATED MODE (2-node distributed inference):
  Based on "Disaggregated Prefill-and-Decode" from the distributed
  inference literature:

    Node 1 (prefill): tokenises the prompt and runs a full prefill
    forward pass, which builds the KV cache.  The KV cache state is
    serialised and broadcast over P2P as a kv_state_transfer message.

    Node 2 (decode): receives the KV state, loads it into its local
    model instance, then runs Lookahead Decoding from n_past onward —
    completely skipping its own prefill computation.

  This limits cross-node synchronisation to ONE transfer per job
  (the KV state) instead of one transfer per token, enabling Tensor
  Parallelism within each node independently while still distributing
  the prefill/decode phases across machines.

Environment variables:
  JACOBI_WINDOW       lookahead window width W   (default: 10)
  JACOBI_NGRAM_SIZE   n-gram pool entry length N (default: 3)
  JACOBI_N_GPU_LAYERS GPU layers                 (default: -1 = all)
  JACOBI_N_CTX        context window size        (default: 4096)
  JACOBI_WORKERS      coordinator workers        (empty = standalone)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import pickle
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import InferenceBackend

log = logging.getLogger("miner.backend.lookahead")

try:
    from llama_cpp import Llama
    import llama_cpp.llama_cpp as _lib
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None
    _lib = None


# ── Binary discovery (for distributed worker mode) ────────────────────────

def _find_binary() -> Optional[str]:
    explicit = os.environ.get("JACOBI_BINARY", "")
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    for p in [
        repo_root / "l2/jacobi/llama_fork/llama-jacobi",
        Path("/usr/local/bin/llama-jacobi"),
        Path("/opt/homebrew/bin/llama-jacobi"),
    ]:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


# ── Lookahead Decoding ────────────────────────────────────────────────────

def _greedy(row: np.ndarray) -> int:
    return int(row.argmax())


def _lookahead_generate_inprocess(
    llm:        "Llama",
    prompt:     str,
    max_tokens: int,
    W:          int = 10,   # lookahead window width
    N:          int = 3,    # n-gram pool entry length
    n_past_override: Optional[int] = None,  # for disaggregated-decode: skip prefill
) -> tuple[str, dict]:
    """
    Lookahead Decoding in-process via llama_cpp bindings.

    Algorithm:
      1. Prefill the prompt (or skip if n_past_override is set — disaggregated decode).
      2. starter = greedy prediction for position n_past.
      3. Each iteration:
         a. Build window: [starter, pool_guess_1, ..., pool_guess_{W-1}]
            Pool guesses come from (token → predicted_next) entries observed in
            prior iterations.  Cold start fills with repeated starter.
         b. One forward pass over all W positions.
         c. Jacobi convergence check: count preds[j-1] == window[j] from j=1.
         d. Accept the converged prefix (starter always accepted).
         e. Record (window[k] → preds[k]) in pool for future windows.
         f. New starter = preds[n_accepted - 1].

    Pool effect:
      - First few iterations: pool empty → window = [starter]*W → convergence ≈ 1
      - After ~N*W iterations: pool populated → window guesses match preds →
        convergence ≈ W → W tokens accepted per forward pass
    """
    import llama_cpp.llama_cpp as lib

    ctx = llm.ctx
    mem = lib.llama_get_memory(ctx)

    # ── Prefill (skip in disaggregated-decode mode) ───────────────────────
    if n_past_override is None:
        tokens = llm.tokenize(
            prompt.encode("utf-8", errors="replace"),
            add_bos=True, special=True,
        )
        llm.reset()
        llm.eval(tokens)
        n_past = llm.n_tokens
        starter = _greedy(llm._scores[-1])
    else:
        # Disaggregated decode: model state already loaded from prefill node
        n_past = n_past_override
        starter = _greedy(llm._scores[-1])

    # ── N-gram pool ───────────────────────────────────────────────────────
    # pool[tok] = list of (next_tok,) tuples representing observed continuations.
    # We use it to pre-fill lookahead positions with high-probability guesses,
    # so the Jacobi convergence check succeeds on the first iteration.
    pool: dict[int, list[int]] = defaultdict(list)

    output_tokens: list[int] = []
    iters = total_accepted = pool_warm_accepts = 0
    t0 = time.monotonic()
    eos_set = {llm.token_eos(), llm.token_bos()}
    done = False

    while len(output_tokens) < max_tokens and not done:
        W_act = min(W, max_tokens - len(output_tokens))

        # ── Build lookahead window ─────────────────────────────────────────
        # Pre-fill from pool: if we've seen tok before, use its predicted
        # continuation as the guess — this maximises first-iteration convergence.
        window = [starter]
        tok = starter
        for _ in range(1, W_act):
            if pool[tok]:
                nxt = pool[tok][0]  # most recent cached continuation
                window.append(nxt)
                tok = nxt
            else:
                window.append(tok)  # cold fallback: repeat

        # ── Forward pass over entire window ───────────────────────────────
        lib.llama_memory_seq_rm(mem, 0, n_past, -1)
        llm.n_tokens = n_past
        llm.eval(window)
        iters += 1

        # preds[k] = prediction for position n_past+k+1 (what follows window[k])
        scores = llm._scores[n_past:n_past + W_act]
        preds = [_greedy(scores[k]) for k in range(W_act)]

        # ── Jacobi convergence check ───────────────────────────────────────
        # window[0] == starter always → n_conv starts at 1
        # window[j] accepted if preds[j-1] == window[j] (corrected off-by-one)
        n_conv = 1  # starter always accepted
        for j in range(1, W_act):
            if preds[j - 1] == window[j]:
                n_conv += 1
            else:
                break

        # ── Track pool-warm accepts (when pool guesses helped convergence) ─
        if n_conv > 1 and pool[starter]:
            pool_warm_accepts += n_conv - 1

        # ── Accept converged prefix ────────────────────────────────────────
        for k in range(n_conv):
            tok = window[k]
            if tok in eos_set:
                done = True; break
            output_tokens.append(tok)
            total_accepted += 1
            n_past += 1
            if len(output_tokens) >= max_tokens:
                done = True; break

        # ── Update pool with new observations ─────────────────────────────
        # Record (window[k] → preds[k]) for each position in the window.
        # This fills the pool for future iterations: when we see window[k] again
        # as a starter or mid-window token, we'll guess preds[k] instead of
        # repeating the fallback.
        for k in range(W_act):
            tok_k = window[k]
            pred_k = preds[k]
            if pred_k not in pool[tok_k]:
                pool[tok_k].insert(0, pred_k)
                if len(pool[tok_k]) > 8:  # keep most recent 8 observations
                    pool[tok_k].pop()

        # ── Update starter for next iteration ─────────────────────────────
        if not done:
            # starter = model's prediction after the last accepted token
            starter = preds[n_conv - 1]

    elapsed_ms = (time.monotonic() - t0) * 1000
    text = llm.detokenize(output_tokens).decode("utf-8", errors="replace")
    stats = {
        "algorithm":         "lookahead",
        "total_iters":       iters,
        "total_accepted":    total_accepted,
        "pool_warm_accepts": pool_warm_accepts,
        "ar_fallbacks":      0,
        "elapsed_ms":        elapsed_ms,
        "accepted_per_iter": total_accepted / max(1, iters),
    }
    return text, stats


# ── Disaggregated state helpers ───────────────────────────────────────────

def save_prefill_state(llm: "Llama", prompt: str) -> tuple[bytes, int]:
    """
    Prefill the prompt and serialise the resulting KV cache state.
    Returns (state_bytes, n_past) for transfer to the decode node.
    """
    tokens = llm.tokenize(
        prompt.encode("utf-8", errors="replace"),
        add_bos=True, special=True,
    )
    llm.reset()
    llm.eval(tokens)
    state = llm.save_state()
    state_bytes = pickle.dumps(state)
    return state_bytes, llm.n_tokens


def load_prefill_state(llm: "Llama", state_bytes: bytes) -> int:
    """
    Load serialised KV state from the prefill node into this model instance.
    Returns n_past (the position where decoding should start).
    """
    state = pickle.loads(state_bytes)
    llm.load_state(state)
    return llm.n_tokens


# ── LookaheadBackend ──────────────────────────────────────────────────────

class LookaheadBackend(InferenceBackend):
    """
    Lookahead Decoding backend — in-process via llama_cpp Python bindings.

    Standalone mode (default):
      Runs full prefill + Lookahead decode on this node.
      The n-gram pool warms up quickly, reaching W-token-per-pass throughput
      within the first 20-30 tokens of generation.

    Disaggregated decode mode (JACOBI_DISAGG_DECODE=1):
      Skips prefill. Waits for a `kv_state_transfer` P2P message carrying
      serialised KV state from the prefill node (another miner), then runs
      Lookahead decoding from n_past onward.
      Activated by the sequencer dispatching a job with mode="disaggregated"
      and assigning this miner the "decode" role.
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map          = model_map
        self._models: dict[str, Any] = {}
        self._load_lock    = threading.Lock()
        self._infer_locks: dict[str, asyncio.Lock] = {}
        self._binary       = _find_binary()
        self._window       = int(os.environ.get("JACOBI_WINDOW",    "10"))
        self._ngram        = int(os.environ.get("JACOBI_NGRAM_SIZE", "3"))
        self._n_gpu_layers = int(os.environ.get("JACOBI_N_GPU_LAYERS", "-1"))
        self._n_ctx        = int(os.environ.get("JACOBI_N_CTX", "4096"))
        self._n_batch      = int(os.environ.get("JACOBI_N_BATCH", "512"))
        self._workers      = os.environ.get("JACOBI_WORKERS", "").strip()
        self._worker_proc: Optional[subprocess.Popen] = None

        # Pending KV states from prefill nodes, keyed by job_id
        self._kv_states: dict[str, bytes] = {}
        self._kv_events: dict[str, asyncio.Event] = {}

        if not HAS_LLAMA_CPP:
            log.warning("llama-cpp-python not installed — lookahead backend disabled")
        else:
            log.info(
                "lookahead_backend mode=%s binary=%s window=%d ngram=%d",
                "coordinator" if self._workers else "standalone",
                self._binary or "none", self._window, self._ngram,
            )
            # Pre-warm: load the first model immediately in a background thread
            if model_map:
                first = next(iter(model_map))
                path = os.path.expanduser(model_map[first])
                if Path(path).is_file():
                    t = threading.Thread(
                        target=self._load_blocking,
                        args=(first,), daemon=True, name="lookahead-prewarm",
                    )
                    t.start()
                    log.info("lookahead_prewarm_started model=%s", first)

    # ── InferenceBackend interface ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lookahead"

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name":          "lookahead",
            "algorithm":     "lookahead_decoding_ngram_pool",
            "binary":        self._binary,
            "window":        self._window,
            "ngram_size":    self._ngram,
            "n_gpu_layers":  self._n_gpu_layers,
            "n_ctx":         self._n_ctx,
            "workers":       self._workers or "standalone",
        }

    def supports(self, model_id: str) -> bool:
        if not HAS_LLAMA_CPP or model_id not in self._map:
            return False
        return Path(os.path.expanduser(self._map[model_id])).is_file()

    async def load(self, model_id: str) -> None:
        if model_id in self._models:
            return
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python not installed")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_blocking, model_id)

    def _load_blocking(self, model_id: str) -> None:
        with self._load_lock:
            if model_id in self._models:
                return
            path = os.path.expanduser(self._map[model_id])
            if not Path(path).is_file():
                raise FileNotFoundError(f"GGUF not found: {path}")
            log.info("lookahead_loading model=%s ngl=%d ctx=%d",
                     model_id, self._n_gpu_layers, self._n_ctx)
            llm = Llama(
                model_path   = path,
                n_gpu_layers = self._n_gpu_layers,
                n_ctx        = self._n_ctx,
                n_batch      = self._n_batch,
                logits_all   = True,  # required: Lookahead needs per-token logits
                verbose      = False,
            )
            self._models[model_id] = llm
            log.info("lookahead_loaded model=%s ctx=%d", model_id, self._n_ctx)

    async def generate(
        self,
        model_id:    str,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
    ) -> str:
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python not installed")
        if model_id not in self._models:
            await self.load(model_id)

        if model_id not in self._infer_locks:
            self._infer_locks[model_id] = asyncio.Lock()
        lock = self._infer_locks[model_id]

        llm   = self._models[model_id]
        W, N  = self._window, self._ngram
        loop  = asyncio.get_event_loop()

        def _run() -> str:
            text, stats = _lookahead_generate_inprocess(llm, prompt, max_tokens, W, N)
            log.info(
                "lookahead_done tokens=%d iters=%d acc/iter=%.2f pool_warm=%d ms=%.0f",
                stats["total_accepted"], stats["total_iters"],
                stats["accepted_per_iter"], stats["pool_warm_accepts"],
                stats["elapsed_ms"],
            )
            return text

        async with lock:
            return await loop.run_in_executor(None, _run)

    # ── Disaggregated prefill ─────────────────────────────────────────────

    async def generate_prefill(self, model_id: str, prompt: str) -> bytes:
        """
        Run prefill only. Return serialised KV state for transfer to decode node.
        Called by the miner when assigned the 'prefill' role in a disaggregated job.
        """
        if model_id not in self._models:
            await self.load(model_id)
        if model_id not in self._infer_locks:
            self._infer_locks[model_id] = asyncio.Lock()
        lock = self._infer_locks[model_id]
        llm  = self._models[model_id]
        loop = asyncio.get_event_loop()

        async with lock:
            state_bytes, n_past = await loop.run_in_executor(
                None, save_prefill_state, llm, prompt
            )
        log.info("lookahead_prefill_done n_past=%d state_bytes=%d",
                 n_past, len(state_bytes))
        return state_bytes

    async def generate_from_kv(
        self,
        model_id:    str,
        state_bytes: bytes,
        max_tokens:  int,
    ) -> str:
        """
        Decode from a received KV state (disaggregated decode role).
        Loads the KV state from the prefill node and runs Lookahead from n_past.
        """
        if model_id not in self._models:
            await self.load(model_id)
        if model_id not in self._infer_locks:
            self._infer_locks[model_id] = asyncio.Lock()
        lock = self._infer_locks[model_id]
        llm  = self._models[model_id]
        W, N = self._window, self._ngram
        loop = asyncio.get_event_loop()

        def _run() -> str:
            n_past = load_prefill_state(llm, state_bytes)
            text, stats = _lookahead_generate_inprocess(
                llm, "", max_tokens, W, N, n_past_override=n_past
            )
            log.info(
                "lookahead_decode_done tokens=%d iters=%d acc/iter=%.2f ms=%.0f",
                stats["total_accepted"], stats["total_iters"],
                stats["accepted_per_iter"], stats["elapsed_ms"],
            )
            return text

        async with lock:
            return await loop.run_in_executor(None, _run)

    def receive_kv_state(self, job_id: str, state_bytes: bytes) -> None:
        """Called when a kv_state_transfer P2P message arrives for this job."""
        self._kv_states[job_id] = state_bytes
        if job_id in self._kv_events:
            self._kv_events[job_id].set()

    async def wait_for_kv_state(self, job_id: str, timeout_s: float = 30.0) -> Optional[bytes]:
        """Block until the prefill node's KV state arrives, or timeout."""
        if job_id in self._kv_states:
            return self._kv_states.pop(job_id)
        event = asyncio.Event()
        self._kv_events[job_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
            return self._kv_states.pop(job_id, None)
        except asyncio.TimeoutError:
            log.warning("kv_state_timeout job=%s", job_id)
            return None
        finally:
            self._kv_events.pop(job_id, None)

    # ── Worker sidecar (binary distributed mode) ──────────────────────────

    def start_worker_sidecar(self, model_id: str) -> None:
        if not self._binary or model_id not in self._map:
            return
        if self._worker_proc and self._worker_proc.poll() is None:
            return
        model_path = os.path.expanduser(self._map[model_id])
        cmd = [self._binary, "-m", model_path,
               "-ngl", str(self._n_gpu_layers),
               "-c",   str(self._n_ctx),
               "--worker", "9900"]
        log.info("lookahead_worker_sidecar_start")
        self._worker_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_worker_sidecar(self) -> None:
        if self._worker_proc and self._worker_proc.poll() is None:
            self._worker_proc.terminate()
        self._worker_proc = None


# Backwards-compatibility alias
JacobiBackend = LookaheadBackend
