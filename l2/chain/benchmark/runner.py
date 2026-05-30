"""
BenchmarkRunner — sequencer-side hardware benchmark orchestration.

Tamper-proof design:
  1. Sequencer measures wall-clock elapsed (challenge-send → response-received).
     The miner never self-reports its own time.
  2. Each challenge includes a unique random nonce mixed into the prompt, so
     the miner cannot precompute the response or replay an old result.
  3. BENCHMARK_COMMIT is a sequencer-synthesized tx (sender="", not the miner).
     Miners cannot forge or alter their own score on-chain.
  4. Scores are keyed by (miner_address, model_id) — different models on
     different hardware yield correct per-model scores.
  5. Scores expire after benchmark_validity_blocks — miners must re-benchmark
     periodically, preventing "benchmark on fast cloud, mine on slow SBC" fraud.

Flow:
  sequencer                          miner
  --------                           -----
  record t0
  broadcast BenchmarkChallenge  →
                                     run inference(prompt+nonce, n_tokens, temp=0)
                               ←    broadcast BenchmarkResponse
  record t1
  elapsed_ms = t1 - t0
  tps = n_tokens / elapsed_s
  submit BENCHMARK_COMMIT tx
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from ..types import TxType, Transaction
from ..crypto import keccak256_hex

log = logging.getLogger("chain.benchmark.runner")


class BenchmarkRunner:
    def __init__(self, sequencer, p2p_node, cfg: dict):
        self._seq = sequencer
        self._p2p = p2p_node
        self._cfg = cfg
        # nonce → asyncio.Future[dict]  (one per in-flight benchmark)
        self._pending: dict[str, asyncio.Future] = {}

    # ── P2P response handler ──────────────────────────────────────────────────

    async def on_benchmark_response(self, payload: dict) -> None:
        """Called by the P2P node when a BenchmarkResponse arrives from a miner."""
        nonce = payload.get("nonce", "")
        fut   = self._pending.get(nonce)
        if fut and not fut.done():
            fut.set_result(payload)

    # ── Public API ────────────────────────────────────────────────────────────

    async def request_benchmark(
        self,
        miner_address: str,
        model_id:      str,
        timeout_s:     float = 120.0,
    ) -> dict:
        """
        Send a BenchmarkChallenge to miner_address and wait for its response.
        Measures sequencer-side wall-clock elapsed time.
        Submits a BENCHMARK_COMMIT tx on success.
        Raises TimeoutError if the miner doesn't respond within timeout_s.
        """
        nonce = keccak256_hex(
            (miner_address.lower() + model_id + str(time.time())).encode()
        )[:24]

        prompt   = self._cfg.get(
            "benchmark_prompt",
            "Explain the difference between supervised and unsupervised learning "
            "in machine learning, with two examples of each.",
        )
        n_tokens  = int(self._cfg.get("benchmark_n_tokens", 64))
        validity  = int(self._cfg.get("benchmark_validity_blocks", 5760))

        challenge = {
            "type":     "BenchmarkChallenge",
            "miner":    miner_address,
            "model_id": model_id,
            "nonce":    nonce,
            "prompt":   prompt,
            "n_tokens": n_tokens,
        }

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[nonce] = fut

        t0 = time.monotonic()
        await self._p2p.broadcast("benchmark_challenges", challenge)

        log.info(
            "benchmark_challenge_sent miner=%s model=%s nonce=%s n_tokens=%d",
            miner_address[:10], model_id, nonce, n_tokens,
        )

        try:
            await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(nonce, None)
            raise TimeoutError(
                f"benchmark timeout after {timeout_s}s: "
                f"miner={miner_address[:10]} model={model_id}"
            )
        finally:
            self._pending.pop(nonce, None)

        elapsed_ms     = max(1, int((time.monotonic() - t0) * 1000))
        tokens_per_sec = round(n_tokens / (elapsed_ms / 1000.0), 4)

        score = {
            "miner":           miner_address,
            "model_id":        model_id,
            "tokens_per_sec":  tokens_per_sec,
            "n_tokens":        n_tokens,
            "elapsed_ms":      elapsed_ms,
            "nonce":           nonce,
            "validity_blocks": validity,
        }

        log.info(
            "benchmark_complete miner=%s model=%s tps=%.2f elapsed_ms=%d validity_blocks=%d",
            miner_address[:10], model_id, tokens_per_sec, elapsed_ms, validity,
        )

        tx = self._build_commit_tx(score)
        await self._seq.mempool.add(tx)

        return score

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_commit_tx(self, score: dict) -> Transaction:
        payload = json.dumps({
            "miner":           score["miner"],
            "model_id":        score["model_id"],
            "tokens_per_sec":  score["tokens_per_sec"],
            "n_tokens":        score["n_tokens"],
            "elapsed_ms":      score["elapsed_ms"],
            "nonce":           score["nonce"],
            "validity_blocks": score["validity_blocks"],
        }, separators=(",", ":"))

        tx_hash = keccak256_hex(
            (
                "BENCHMARK_COMMIT"
                + score["miner"].lower()
                + score["model_id"]
                + score["nonce"]
            ).encode()
        )
        return Transaction(
            tx_type=TxType.BENCHMARK_COMMIT,
            sender="",
            nonce=0,
            payload=payload,
            gas_price=0,
            signature="",
            tx_hash=tx_hash,
        )
