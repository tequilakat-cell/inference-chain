"""
RegistryClient — asyncpg interface to the pg_inft miner registry.

Replaces the Peerbit/OrbitDB Node.js sidecar. All three OrbitDB collections
(miners, jobs, reputation_events) now live in the pg_inft PostgreSQL schema.

Drop-in replacement for PeerbitClient: same public method signatures, same
graceful-degradation contract (all methods are no-ops when pg_dsn is absent
or the connection pool is unavailable).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

log = logging.getLogger("l2_miner.registry")


class RegistryClient:
    """
    Async PostgreSQL-backed miner registry.

    All public methods degrade gracefully: if pg_dsn is empty or the pool
    fails to connect, every operation is a silent no-op.
    """

    def __init__(
        self,
        address: str,
        models: list[str],
        backend: str,
        dsn: str = "",
        p2p_addr: str = "",
        l2_chain_id: int = 2026,
        max_shards: int = 4,
    ) -> None:
        self._address    = address
        self._models     = models
        self._backend    = backend
        self._dsn        = dsn
        self._p2p_addr   = p2p_addr
        self._chain_id   = str(l2_chain_id)
        self._max_shards = max_shards
        self._pool       = None  # asyncpg.Pool, set by start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open connection pool and register this miner."""
        if not self._dsn:
            log.info("registry_disabled no pg_dsn configured")
            return
        try:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=4,
                command_timeout=10,
            )
            log.info("registry_connected dsn=%s", self._dsn[:40])
        except Exception as exc:
            log.warning("registry_unavailable dsn=%s err=%s — registry disabled", self._dsn[:40], exc)
            self._pool = None
            return
        await self.register_miner()

    async def stop(self) -> None:
        """Deregister and close the connection pool."""
        await self._deregister()
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:
                log.debug("registry_close_err err=%s", exc)
            self._pool = None

    # ── Miner registry ────────────────────────────────────────────────────────

    async def register_miner(self) -> None:
        await self._exec(
            "SELECT inft.inft_upsert_miner($1,$2,$3,$4,$5,$6,true)",
            self._address, self._models, self._backend,
            self._p2p_addr, self._chain_id, self._max_shards,
        )
        log.info("registry_registered address=%s models=%s", self._address[:12], self._models)

    async def heartbeat(self) -> None:
        await self._exec(
            "SELECT inft.inft_upsert_miner($1,$2,$3,$4,$5,$6,true)",
            self._address, self._models, self._backend,
            self._p2p_addr, self._chain_id, self._max_shards,
        )

    async def get_miners_for_model(self, model_id: str) -> list[dict]:
        rows = await self._fetch(
            "SELECT address, models, backend, p2p_addr, max_shards, reputation, last_seen"
            " FROM inft.inft_get_miners_for_model($1)",
            model_id,
        )
        return [dict(r) for r in rows] if rows else []

    async def list_miners(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT address, models, backend, p2p_addr, max_shards, reputation, last_seen"
            " FROM inft.inft_miners WHERE active = true ORDER BY reputation DESC"
        )
        return [dict(r) for r in rows] if rows else []

    async def _deregister(self) -> None:
        await self._exec("SELECT inft.inft_deregister_miner($1)", self._address)
        log.info("registry_deregistered address=%s", self._address[:12])

    # ── Job board ─────────────────────────────────────────────────────────────

    async def announce_job_accepted(
        self, job_id: str, model_id: str, mode: str, n_shards: int
    ) -> None:
        await self._exec(
            "SELECT inft.inft_upsert_job($1,$2,$3,$4,$5,$6,$7,$8)",
            job_id, model_id, mode, n_shards, "partial", "", "", 0,
        )

    async def announce_job_complete(
        self, job_id: str, output_hash: str, latency_ms: int
    ) -> None:
        await self._exec(
            "SELECT inft.inft_upsert_job($1,$2,$3,$4,$5,$6,$7,$8)",
            job_id, "", "", 0, "complete", output_hash, "", latency_ms,
        )

    # ── Reputation events ─────────────────────────────────────────────────────

    async def log_shard_complete(
        self, job_id: str, shard_idx: int, latency_ms: int
    ) -> None:
        await self._exec(
            "SELECT inft.inft_log_reputation_event($1,$2,$3,$4,$5,$6)",
            str(uuid.uuid4()), self._address, "shard_complete", 1, job_id, shard_idx,
        )

    async def log_shard_failed(self, job_id: str, shard_idx: int) -> None:
        await self._exec(
            "SELECT inft.inft_log_reputation_event($1,$2,$3,$4,$5,$6)",
            str(uuid.uuid4()), self._address, "shard_failed", -5, job_id, shard_idx,
        )

    async def get_reputation_events(self, address: str) -> list[dict]:
        rows = await self._fetch(
            "SELECT event_id, miner, event_type, delta, job_id, shard_idx, created_at"
            " FROM inft.inft_reputation_events WHERE miner = $1 ORDER BY created_at DESC",
            address,
        )
        return [dict(r) for r in rows] if rows else []

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _exec(self, sql: str, *args) -> None:
        if self._pool is None:
            return
        try:
            await self._pool.execute(sql, *args)
        except Exception as exc:
            log.debug("registry_exec_err sql=%.40s err=%s", sql, exc)

    async def _fetch(self, sql: str, *args) -> list:
        if self._pool is None:
            return []
        try:
            return await self._pool.fetch(sql, *args)
        except Exception as exc:
            log.debug("registry_fetch_err sql=%.40s err=%s", sql, exc)
            return []
