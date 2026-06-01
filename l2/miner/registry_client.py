"""
RegistryClient — aiohttp client for the Peerbit sidecar's miner registry API.

Replaces the asyncpg/pg_inft implementation. All three collections
(miners, jobs, reputation_events) now live in the Peerbit distributed store.

Same public method signatures and graceful-degradation contract as before.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

log = logging.getLogger("l2_miner.registry")


class RegistryClient:
    """
    Peerbit-backed miner registry via sidecar HTTP API.

    All public methods degrade gracefully: if peerbit_url is empty or the
    sidecar is unavailable, every operation is a silent no-op.
    """

    def __init__(
        self,
        address: str,
        models: list[str],
        backend: str,
        url: str = "",
        p2p_addr: str = "",
        l2_chain_id: int = 2026,
        max_shards: int = 4,
    ) -> None:
        self._address = address
        self._models = models
        self._backend = backend
        self._url = url.rstrip("/") if url else ""
        self._p2p_addr = p2p_addr
        self._chain_id = str(l2_chain_id)
        self._max_shards = max_shards
        self._session = None  # aiohttp.ClientSession

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._url:
            log.info("registry_disabled no peerbit_url configured")
            return
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            log.info("registry_connected url=%s", self._url)
        except Exception as exc:
            log.warning("registry_unavailable url=%s err=%s — registry disabled", self._url, exc)
            self._session = None
            return
        await self.register_miner()

    async def stop(self) -> None:
        await self._deregister()
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as exc:
                log.debug("registry_close_err err=%s", exc)
            self._session = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.post(f"{self._url}{path}", json=payload) as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("registry_post_err path=%s err=%s", path, exc)
            return None

    async def _get(self, path: str, params: dict | None = None) -> list | dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.get(f"{self._url}{path}", params=params) as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("registry_get_err path=%s err=%s", path, exc)
            return None

    async def _delete(self, path: str) -> None:
        if self._session is None:
            return
        try:
            async with self._session.delete(f"{self._url}{path}") as r:
                await r.json()
        except Exception as exc:
            log.debug("registry_delete_err path=%s err=%s", path, exc)

    # ── Miner registry ────────────────────────────────────────────────────────

    async def register_miner(self) -> None:
        await self._post("/miners", {
            "address": self._address,
            "models": self._models,
            "backend": self._backend,
            "p2p_addr": self._p2p_addr,
            "chain_id": self._chain_id,
            "max_shards": self._max_shards,
        })
        log.info("registry_registered address=%s models=%s", self._address[:12], self._models)

    async def heartbeat(self) -> None:
        await self._post("/miners", {
            "address": self._address,
            "models": self._models,
            "backend": self._backend,
            "p2p_addr": self._p2p_addr,
            "chain_id": self._chain_id,
            "max_shards": self._max_shards,
        })

    async def get_miners_for_model(self, model_id: str) -> list[dict]:
        data = await self._get("/miners", {"model": model_id})
        return data if isinstance(data, list) else []

    async def list_miners(self) -> list[dict]:
        data = await self._get("/miners")
        return data if isinstance(data, list) else []

    async def _deregister(self) -> None:
        await self._delete(f"/miners/{self._address}")
        log.info("registry_deregistered address=%s", self._address[:12])

    # ── Job board ─────────────────────────────────────────────────────────────

    async def announce_job_accepted(
        self, job_id: str, model_id: str, mode: str, n_shards: int
    ) -> None:
        await self._post("/jobs", {
            "job_id": job_id,
            "model_id": model_id,
            "mode": mode,
            "n_shards": n_shards,
            "status": "partial",
        })

    async def announce_job_complete(
        self, job_id: str, output_hash: str, latency_ms: int
    ) -> None:
        await self._post("/jobs", {
            "job_id": job_id,
            "model_id": "",
            "mode": "",
            "n_shards": 0,
            "status": "complete",
            "output_hash": output_hash,
            "latency_ms": latency_ms,
        })

    # ── Reputation events ─────────────────────────────────────────────────────

    async def log_shard_complete(
        self, job_id: str, shard_idx: int, latency_ms: int
    ) -> None:
        await self._post("/reputation", {
            "event_id": str(uuid.uuid4()),
            "miner": self._address,
            "event_type": "shard_complete",
            "delta": 1,
            "job_id": job_id,
            "shard_idx": shard_idx,
        })

    async def log_shard_failed(self, job_id: str, shard_idx: int) -> None:
        await self._post("/reputation", {
            "event_id": str(uuid.uuid4()),
            "miner": self._address,
            "event_type": "shard_failed",
            "delta": -5,
            "job_id": job_id,
            "shard_idx": shard_idx,
        })

    async def get_reputation_events(self, address: str) -> list[dict]:
        data = await self._get(f"/reputation/{address}")
        return data if isinstance(data, list) else []
