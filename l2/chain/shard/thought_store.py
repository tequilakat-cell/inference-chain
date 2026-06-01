"""
ThoughtStore — aiohttp client for the Peerbit sidecar's thought/rollup API.

All methods degrade gracefully: if the sidecar is unreachable, every operation
is a no-op that returns an empty/False result without raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("thought_store")


@dataclass
class ThoughtResult:
    id: int
    job_id: str
    miner_address: str
    model_id: str
    question_text: str
    thinking_text: str
    answer_text: str
    score: float


class ThoughtStore:
    """
    HTTP client wrapping the Peerbit sidecar's /thoughts and /rollups endpoints.

    url — base URL of the sidecar, e.g. "http://127.0.0.1:7731".
    All public methods degrade gracefully when the sidecar is unavailable.
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
            log.info("thought_store_connected url=%s", self._url)
        except Exception as exc:
            log.warning("thought_store_unavailable url=%s err=%s", self._url, exc)
            self._session = None

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as exc:
                log.debug("thought_store_close_err err=%s", exc)
            self._session = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.get(f"{self._url}{path}", params=params) as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("thought_store_get_err path=%s err=%s", path, exc)
            return None

    async def _post(self, path: str, payload: dict) -> dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.post(f"{self._url}{path}", json=payload) as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("thought_store_post_err path=%s err=%s", path, exc)
            return None

    async def _delete(self, path: str) -> dict | None:
        if self._session is None:
            return None
        try:
            async with self._session.delete(f"{self._url}{path}") as r:
                body = await r.json()
                return body.get("data") if body.get("ok") else None
        except Exception as exc:
            log.debug("thought_store_delete_err path=%s err=%s", path, exc)
            return None

    # ── Thought log ───────────────────────────────────────────────────────────

    async def search(
        self, question: str, model_id: str = "", limit: int = 5
    ) -> list[ThoughtResult]:
        data = await self._get(
            "/thoughts/search",
            {"q": question, "model_id": model_id, "limit": limit},
        )
        if not data:
            return []
        return [_row_to_thought(r) for r in data]

    async def recent(self, model_id: str = "", limit: int = 20) -> list[ThoughtResult]:
        data = await self._get(
            "/thoughts/recent", {"model_id": model_id, "limit": limit}
        )
        if not data:
            return []
        return [_row_to_thought(r) for r in data]

    async def ingest(
        self,
        job_id: str,
        miner_address: str,
        model_id: str,
        question: str,
        thinking: str,
        answer: str,
        proof_sig_hex: str,
        block_number: Optional[int] = None,
        tx_hash: Optional[str] = None,
        peer_origin: Optional[str] = None,
    ) -> bool:
        data = await self._post("/thoughts", {
            "job_id": job_id,
            "miner_address": miner_address,
            "model_id": model_id,
            "question": question,
            "thinking": thinking,
            "answer": answer,
            "proof_sig": proof_sig_hex,
            "block_number": block_number,
            "tx_hash": tx_hash,
            "peer_origin": peer_origin,
        })
        return bool(data and data.get("ingested"))

    # ── Semantic memory ───────────────────────────────────────────────────────

    async def set_embedding(self, job_id: str, embedding) -> bool:
        if not embedding:
            return False
        data = await self._post("/thoughts/embedding", {
            "job_id": job_id,
            "embedding": list(float(x) for x in embedding),
        })
        return bool(data and data.get("updated"))

    async def search_semantic(
        self, embedding, model_id: str = "", limit: int = 20
    ) -> list[ThoughtResult]:
        if not embedding:
            return []
        data = await self._post("/thoughts/search/semantic", {
            "embedding": list(float(x) for x in embedding),
            "model_id": model_id,
            "limit": limit,
        })
        if not data:
            return []
        return [_row_to_thought(r) for r in data]

    # ── Rollups ───────────────────────────────────────────────────────────────

    async def upsert_rollup(
        self,
        rollup_id: str,
        topic: str,
        model_id: str,
        summary: str,
        source_count: int,
        source_job_ids,
        embedding,
        content_hash: bytes = b"",
    ) -> bool:
        data = await self._post("/rollups", {
            "rollup_id": rollup_id,
            "topic": topic,
            "model_id": model_id,
            "summary": summary,
            "source_count": int(source_count),
            "source_job_ids": list(source_job_ids),
            "embedding": list(float(x) for x in embedding) if embedding else None,
        })
        return bool(data and data.get("upserted"))

    async def search_rollups(
        self, embedding, model_id: str = "", limit: int = 5
    ) -> list[dict]:
        if not embedding:
            return []
        data = await self._post("/rollups/search", {
            "embedding": list(float(x) for x in embedding),
            "model_id": model_id,
            "limit": limit,
        })
        return data or []

    async def list_rollups(self, model_id: str = "", limit: int = 20) -> list[dict]:
        data = await self._get("/rollups", {"model_id": model_id, "limit": limit})
        return data or []

    # ── Live TPS (backed by benchmark store) ─────────────────────────────────

    async def update_live_tps(
        self, miner_address: str, model_id: str, actual_tps: float
    ) -> None:
        await self._post("/benchmarks/live-tps", {
            "miner_address": miner_address,
            "model_id": model_id,
            "actual_tps": float(actual_tps),
        })

    async def get_live_tps(
        self, miner_address: str, model_id: str
    ) -> Optional[dict]:
        from urllib.parse import quote
        data = await self._get(
            f"/benchmarks/{miner_address}/{quote(model_id, safe='')}"
        )
        return data  # already dict or None

    # ── Job context prefetch ──────────────────────────────────────────────────

    async def set_job_context(
        self,
        job_id: str,
        query_text: str,
        context_text: str,
        context_hash: str,
        model_id: str,
        n_entries: int,
    ) -> bool:
        data = await self._post("/contexts", {
            "job_id": job_id,
            "query_text": query_text,
            "context_text": context_text,
            "context_hash": context_hash,
            "model_id": model_id,
            "n_entries": n_entries,
        })
        return bool(data and data.get("set"))

    async def get_job_context(self, job_id: str) -> Optional[dict]:
        return await self._get(f"/contexts/{job_id}")

    async def expire_job_contexts(self) -> int:
        data = await self._delete("/contexts/expired")
        return int(data.get("deleted", 0)) if data else 0

    # ── Peer sync ─────────────────────────────────────────────────────────────

    async def record_peer(
        self, peer_address: str, job_id: str, rejected: bool
    ) -> None:
        await self._post("/peers", {
            "peer_address": peer_address,
            "job_id": job_id,
            "rejected": rejected,
        })


def _row_to_thought(r: dict) -> ThoughtResult:
    return ThoughtResult(
        id=int(r.get("id", 0)),
        job_id=r.get("job_id", ""),
        miner_address=r.get("miner_address", ""),
        model_id=r.get("model_id", ""),
        question_text=r.get("question_text", ""),
        thinking_text=r.get("thinking_text", ""),
        answer_text=r.get("answer_text", ""),
        score=float(r.get("score", 0.0)),
    )
