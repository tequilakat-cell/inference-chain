"""
Jacobi parallel-decoding backend for inference-chain.

Implements Jacobi iterative decoding **in-process** using the llama_cpp
Python bindings.  The model is loaded once on first use and kept in memory
between jobs — identical lifecycle to LlamaCppBackend.

Algorithm (corrected position semantics):
  After eval([g_0..g_{W-1}]) at positions [n_past..n_past+W-1]:
    scores[i] = logits for position n_past+i, predicting token n_past+i+1

  Convergence check:
    pos 0:   accepted if cur[0] == starter_pred
    pos j≥1: accepted if scores[j-1].argmax() == cur[j]

  After accepting n tokens, starter_pred = scores[n-1].argmax()

Speedup comes from W tokens being processed in ONE forward pass per
iteration, while standard AR does 1.  On large models with high
prediction confidence, most windows converge in 1-2 iterations → ~W/2x.

Build the llama-jacobi binary for distributed mode:
    cd l2/jacobi/llama_fork/examples/jacobi && bash build.sh

Environment variables:
  JACOBI_WINDOW       guess window size W           (default: 10)
  JACOBI_MAX_ITER     max iters before AR fallback  (default: 8)
  JACOBI_N_GPU_LAYERS GPU layers (default: -1 = all)
  JACOBI_N_CTX        context window size           (default: 4096)
  JACOBI_N_BATCH      batch size                    (default: 512)
  JACOBI_WORKERS      coordinator worker addresses  (empty = standalone)
  JACOBI_WORKER_PORT  worker TCP port               (default: 9900)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import InferenceBackend

log = logging.getLogger("miner.backend.jacobi")

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


# ── In-process Jacobi core ────────────────────────────────────────────────

def _greedy(scores_row: np.ndarray) -> int:
    """Argmax of a logit row → token id."""
    return int(scores_row.argmax())


def _jacobi_generate_inprocess(
    llm:        "Llama",
    prompt:     str,
    max_tokens: int,
    window:     int,
    max_iter:   int,
) -> tuple[str, dict]:
    """
    Run Jacobi decoding entirely in-process using llama_cpp._scores.

    Requirements:
      - llm must be created with logits_all=True so _scores contains
        per-token logits for every position in the batch (not just the last).
      - llm._scores[i] = logits at position i, predicting position i+1.
      - Window scores after eval(guess) at positions n_past..n_past+W-1
        are in _scores[n_past : n_past+W], NOT _scores[:W].

    Returns (generated_text, stats_dict).
    """
    import llama_cpp.llama_cpp as lib

    ctx = llm.ctx
    mem = lib.llama_get_memory(ctx)

    # ── Step 1: prefill the prompt ────────────────────────────────────────
    prompt_tokens = llm.tokenize(
        prompt.encode("utf-8", errors="replace"),
        add_bos=True,
        special=True,
    )
    llm.reset()
    llm.eval(prompt_tokens)
    n_past = llm.n_tokens  # position of first generated token

    # ── Step 2: starter_pred = greedy from last prefill logit ─────────────
    # _scores[-1] = logits at position n_past-1, predicting position n_past.
    starter_pred = _greedy(llm._scores[-1])

    # ── Step 3: lookahead initialisation ─────────────────────────────────
    # W sequential AR steps give a coherent seed for the Jacobi window.
    # Prevents the "self-fulfilling prophecy" trap where all positions
    # collapse to the same repeated token.
    guess: list[int] = []
    next_tok = starter_pred
    for i in range(window):
        guess.append(next_tok)
        if i < window - 1:
            llm.eval([next_tok])
            next_tok = _greedy(llm._scores[-1])
    # KV cache: positions 0..n_past+W-2 populated (W-1 AR steps decoded).
    # Jacobi loop clears from n_past before each iteration.

    # ── Step 4: Jacobi iteration loop ────────────────────────────────────
    output_tokens: list[int] = []
    total_iters   = 0
    total_accepted = 0
    ar_fallbacks  = 0
    t0 = time.monotonic()
    done = False

    eos_set = {llm.token_eos(), llm.token_bos()}

    while len(output_tokens) < max_tokens and not done:
        W_act = min(window, max_tokens - len(output_tokens))
        window_done = False

        for _it in range(max_iter):
            # ── Clear KV for guess positions, keep accepted prefix ────────
            lib.llama_memory_seq_rm(mem, 0, n_past, -1)
            llm.n_tokens = n_past  # sync Python position tracker (eval reads self.n_tokens)

            # ── One forward pass over the entire guess window ─────────────
            cur = guess[:W_act]
            llm.eval(cur)
            total_iters += 1

            # ── Window logits ──────────────────────────────────────────────
            # _scores has shape (n_past + W_act, vocab) with logits_all=True.
            # The guess window logits are in the LAST W_act rows (not first!).
            #   scores[k] = logits at position n_past+k, predicting n_past+k+1
            scores = llm._scores[n_past : n_past + W_act]  # shape (W_act, vocab)

            # ── Convergence (corrected off-by-one) ─────────────────────────
            # pos 0:   accepted if cur[0] == starter_pred
            # pos j≥1: accepted if scores[j-1].argmax() == cur[j]
            n_conv = 0
            if cur[0] == starter_pred:
                n_conv = 1
                for j in range(1, W_act):
                    if _greedy(scores[j - 1]) == cur[j]:
                        n_conv += 1
                    else:
                        break

            if n_conv > 0:
                for tok in cur[:n_conv]:
                    if tok in eos_set:
                        done = True
                        break
                    output_tokens.append(tok)
                    total_accepted += 1
                    if len(output_tokens) >= max_tokens:
                        done = True
                        break

                n_past += n_conv
                starter_pred = _greedy(scores[n_conv - 1])

                # Next window: slot 0 = starter_pred, tail from scores
                new_guess = [starter_pred] * window
                for i in range(1, window):
                    src = n_conv + i - 1
                    if src < W_act:
                        new_guess[i] = _greedy(scores[src])
                    else:
                        break
                guess = new_guess
                window_done = True
                break

            # No convergence: realign guess with correct semantics
            # guess[0] = starter_pred (guaranteed correct)
            # guess[j+1] = model's prediction for j+1 given corrected prefix
            guess[0] = starter_pred
            for j in range(W_act - 1):
                guess[j + 1] = _greedy(scores[j])

        if not window_done and not done:
            # AR fallback: starter_pred is always correct by definition
            ar_fallbacks += 1
            if starter_pred in eos_set:
                break
            output_tokens.append(starter_pred)
            total_accepted += 1
            lib.llama_memory_seq_rm(mem, 0, n_past, -1)
            llm.n_tokens = n_past
            llm.eval([starter_pred])
            n_past += 1
            starter_pred = _greedy(llm._scores[-1])
            guess = [starter_pred] * window

    elapsed_ms = (time.monotonic() - t0) * 1000
    text = llm.detokenize(output_tokens).decode("utf-8", errors="replace")
    stats = {
        "total_iters":     total_iters,
        "total_accepted":  total_accepted,
        "ar_fallbacks":    ar_fallbacks,
        "elapsed_ms":      elapsed_ms,
        "accepted_per_iter": total_accepted / max(1, total_iters),
    }
    return text, stats


# ── JacobiBackend ──────────────────────────────────────────────────────────

class JacobiBackend(InferenceBackend):
    """
    Jacobi parallel decoding — in-process via llama_cpp Python bindings.

    Model is loaded once and cached in memory between jobs.
    Zero cold-start overhead on repeated inference calls.
    Falls back to the llama-jacobi binary for distributed coordinator mode.
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map          = model_map
        self._models: dict[str, Any] = {}   # model_id → Llama instance
        self._load_lock    = threading.Lock()
        self._binary       = _find_binary()
        self._window       = int(os.environ.get("JACOBI_WINDOW",    "10"))
        self._max_iter     = int(os.environ.get("JACOBI_MAX_ITER",  "8"))
        self._n_gpu_layers = int(os.environ.get("JACOBI_N_GPU_LAYERS", "-1"))
        self._n_ctx        = int(os.environ.get("JACOBI_N_CTX", "4096"))
        self._n_batch      = int(os.environ.get("JACOBI_N_BATCH", "512"))
        self._workers      = os.environ.get("JACOBI_WORKERS", "").strip()
        self._worker_port  = int(os.environ.get("JACOBI_WORKER_PORT", "9900"))
        self._worker_proc: Optional[subprocess.Popen] = None

        if not HAS_LLAMA_CPP:
            log.warning("llama-cpp-python not installed — jacobi backend disabled")
        else:
            mode = "coordinator" if self._workers else "standalone"
            log.info(
                "jacobi_backend mode=%s binary=%s window=%d",
                mode, self._binary or "none", self._window,
            )
            # Pre-warm: load the first model immediately in a background thread
            # so it's in memory before the first shard arrives.
            if model_map:
                first_model = next(iter(model_map))
                model_path = os.path.expanduser(model_map[first_model])
                if Path(model_path).is_file():
                    t = threading.Thread(
                        target=self._load_blocking,
                        args=(first_model,),
                        daemon=True,
                        name="jacobi-prewarm",
                    )
                    t.start()
                    log.info("jacobi_prewarm_started model=%s", first_model)

    # ── InferenceBackend interface ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "jacobi"

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name":          "jacobi",
            "binary":        self._binary,
            "window":        self._window,
            "max_iter":      self._max_iter,
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
        # _load_blocking holds _load_lock while loading; calling it via
        # run_in_executor will block until the pre-warm thread finishes if it's
        # already loading the same model (lock serialises both paths).
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_blocking, model_id)

    def _load_blocking(self, model_id: str) -> None:
        with self._load_lock:
            if model_id in self._models:
                return  # already loaded by another thread
            model_path = os.path.expanduser(self._map[model_id])
            if not Path(model_path).is_file():
                raise FileNotFoundError(f"GGUF not found: {model_path}")
            log.info("jacobi_loading model=%s ngl=%d ctx=%d",
                     model_id, self._n_gpu_layers, self._n_ctx)
            llm = Llama(
                model_path   = model_path,
                n_gpu_layers = self._n_gpu_layers,
                n_ctx        = self._n_ctx,
                n_batch      = self._n_batch,
                logits_all   = True,   # required: Jacobi needs per-token logits
                verbose      = False,
            )
            self._models[model_id] = llm
            log.info("jacobi_loaded model=%s ctx=%d", model_id, self._n_ctx)

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

        llm   = self._models[model_id]
        W     = self._window
        iters = self._max_iter
        loop  = asyncio.get_event_loop()

        def _run() -> str:
            text, stats = _jacobi_generate_inprocess(llm, prompt, max_tokens, W, iters)
            log.info(
                "jacobi_done tokens=%d iters=%d acc/iter=%.2f fallbacks=%d ms=%.0f",
                stats["total_accepted"], stats["total_iters"],
                stats["accepted_per_iter"], stats["ar_fallbacks"], stats["elapsed_ms"],
            )
            return text

        return await loop.run_in_executor(None, _run)

    # ── Worker sidecar (distributed mode) ─────────────────────────────────

    def start_worker_sidecar(self, model_id: str) -> None:
        if not self._binary or model_id not in self._map:
            return
        if self._worker_proc and self._worker_proc.poll() is None:
            return
        model_path = os.path.expanduser(self._map[model_id])
        cmd = [self._binary, "-m", model_path,
               "-ngl", str(self._n_gpu_layers),
               "-c", str(self._n_ctx),
               "--worker", str(self._worker_port)]
        log.info("jacobi_worker_sidecar_start port=%d", self._worker_port)
        self._worker_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_worker_sidecar(self) -> None:
        if self._worker_proc and self._worker_proc.poll() is None:
            self._worker_proc.terminate()
        self._worker_proc = None
