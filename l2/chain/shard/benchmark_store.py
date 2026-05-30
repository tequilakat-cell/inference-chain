"""
BenchmarkStore — asyncpg interface to inft.inft_miner_benchmarks.

Chain-side pg_inft client.  Writes benchmark scores when BENCHMARK_COMMIT
transactions land in a block.  Provides bulk score loading for startup
reconciliation with StateDB.

Degrades gracefully: if pg_inft is unavailable, all operations are no-ops
and the chain falls back to StateDB-only scoring.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("chain.shard.benchmark_store")

_MIN_LIVE_SAMPLES = 5   # require this many production samples before trusting live_tps


class BenchmarkStore:
    """
    Async interface to inft.inft_miner_benchmarks (pg_inft v1.1+).

    Methods mirror the SQL functions added in pg_inft--1.0--1.1.sql:
      inft_upsert_benchmark()   — write on BENCHMARK_COMMIT tx
      inft_get_benchmark()      — point lookup
      get_all_benchmarks()      — bulk load on startup
    """

    def __init__(self, dsn: str) -> None:
        self._dsn  = dsn
        self._pool = None  # asyncpg.Pool

    async def connect(self) -> None:
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=3,
                command_timeout=10,
            )
            log.info("benchmark_store_connected dsn=%s", self._dsn[:40])
        except Exception as exc:
            log.warning(
                "benchmark_store_unavailable dsn=%s err=%s — pg_inft scoring disabled",
                self._dsn[:40], exc,
            )
            self._pool = None

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_benchmark(
        self,
        miner_address:    str,
        model_id:         str,
        tokens_per_sec:   float,
        n_tokens:         int,
        elapsed_ms:       int,
        nonce:            str,
        block_number:     int,
        expires_at_block: int,
    ) -> None:
        """Write a sequencer-measured benchmark score to pg_inft."""
        if self._pool is None:
            return
        try:
            await self._pool.execute(
                "SELECT inft.inft_upsert_benchmark($1,$2,$3,$4,$5,$6,$7,$8)",
                miner_address,
                model_id,
                float(tokens_per_sec),
                int(n_tokens),
                int(elapsed_ms),
                nonce,
                int(block_number),
                int(expires_at_block),
            )
        except Exception as exc:
            log.debug(
                "benchmark_store_upsert_err miner=%s model=%s err=%s",
                miner_address[:10], model_id, exc,
            )

    async def upsert_from_tx(self, tx, block_number: int) -> None:
        """
        Parse a BENCHMARK_COMMIT transaction and write it to pg_inft.
        Called from node._on_block_produced for every BENCHMARK_COMMIT tx.
        """
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
        """
        Return score dict or None.
        Includes both bench_tps and live_tps so callers can pick the best signal.
        """
        if self._pool is None:
            return None
        try:
            row = await self._pool.fetchrow(
                "SELECT tokens_per_sec, live_tps, live_sample_count, "
                "       block_number, expires_at_block, nonce, last_updated "
                "FROM inft.inft_get_benchmark($1, $2)",
                miner_address, model_id,
            )
            if row is None:
                return None
            return dict(row)
        except Exception as exc:
            log.debug(
                "benchmark_store_get_err miner=%s model=%s err=%s",
                miner_address[:10], model_id, exc,
            )
            return None

    async def get_all_benchmarks(self) -> list[dict]:
        """
        Bulk-load all benchmark scores for startup reconciliation with StateDB.
        Returns list of {miner_address, model_id, tokens_per_sec, live_tps, ...}.
        """
        if self._pool is None:
            return []
        try:
            rows = await self._pool.fetch(
                "SELECT miner_address, model_id, tokens_per_sec, live_tps, "
                "       live_sample_count, block_number, expires_at_block, nonce "
                "FROM inft.inft_miner_benchmarks "
                "ORDER BY last_updated DESC"
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            log.debug("benchmark_store_bulk_err err=%s", exc)
            return []

    def effective_tps(self, row: dict) -> float:
        """
        Return the most reliable TPS estimate from a benchmark row.
        Prefer live_tps once enough production samples have accumulated.
        """
        live = row.get("live_tps")
        samples = row.get("live_sample_count", 0)
        bench = float(row.get("tokens_per_sec", 0))
        if live is not None and samples >= _MIN_LIVE_SAMPLES:
            return float(live)
        return bench
