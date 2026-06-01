"""
BenchmarkStore — aiohttp client for the Peerbit sidecar's /benchmarks endpoint.

Degrades gracefully: if the sidecar is unavailable, all operations are no-ops
and the chain falls back to StateDB-only scoring.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

log = logging.getLogger("chain.shard.benchmark_store")

_MIN_LIVE_SAMPLES = 5


class BenchmarkStore:
    """
    HTTP client wrapping the Peerbit sidecar's /benchmarks endpoints.

    url — base URL of the sidecar, e.g. "http://127.0.0.1:7731".
    """

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")
        self._session = None  # aiohttp.ClientSession

    async def connect(self) -> None:
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            log.info("benchmark_store_connected url=%s", self._url)
        except Exception as exc:
            log.warning(
                "benchmark_store_unavailable url=%s err=%s — scoring disabled",
                self._url, exc,
            )
            self._session = None

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get(self, path: str) -> dict | list | None:
        if self._session is None:
            return None
        try:
            async with self._session.get(f"{self._url}{path}") as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("benchmark_store_get_err path=%s err=%s", path, exc)
            return None

    async def _post(self, path: str, payload: dict) -> dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.post(f"{self._url}{path}", json=payload) as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("benchmark_store_post_err path=%s err=%s", path, exc)
            return None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_benchmark(
        self,
        miner_address: str,
        model_id: str,
        tokens_per_sec: float,
        n_tokens: int,
        elapsed_ms: int,
        nonce: str,
        block_number: int,
        expires_at_block: int,
    ) -> None:
        await self._post("/benchmarks", {
            "miner_address": miner_address,
            "model_id": model_id,
            "tokens_per_sec": float(tokens_per_sec),
            "n_tokens": int(n_tokens),
            "elapsed_ms": int(elapsed_ms),
            "nonce": nonce,
            "block_number": int(block_number),
            "expires_at_block": int(expires_at_block),
        })

    async def upsert_from_tx(self, tx, block_number: int) -> None:
        try:
            p = tx.payload_dict()
            await self.upsert_benchmark(
                miner_address=p["miner"],
                model_id=p["model_id"],
                tokens_per_sec=float(p["tokens_per_sec"]),
                n_tokens=int(p["n_tokens"]),
                elapsed_ms=int(p["elapsed_ms"]),
                nonce=p.get("nonce", ""),
                block_number=block_number,
                expires_at_block=block_number + int(p.get("validity_blocks", 5760)),
            )
            log.info(
                "benchmark_store_written miner=%s model=%s tps=%.2f block=%d",
                p["miner"][:10], p["model_id"],
                float(p["tokens_per_sec"]), block_number,
            )
        except Exception as exc:
            log.debug("benchmark_store_tx_parse_err err=%s", exc)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_benchmark(
        self, miner_address: str, model_id: str
    ) -> Optional[dict]:
        return await self._get(
            f"/benchmarks/{miner_address}/{quote(model_id, safe='')}"
        )

    async def get_all_benchmarks(self) -> list[dict]:
        data = await self._get("/benchmarks")
        return data if isinstance(data, list) else []

    def effective_tps(self, row: dict) -> float:
        live = row.get("live_tps")
        samples = row.get("live_sample_count", 0)
        bench = float(row.get("tokens_per_sec", 0))
        if live is not None and samples >= _MIN_LIVE_SAMPLES:
            return float(live)
        return bench
