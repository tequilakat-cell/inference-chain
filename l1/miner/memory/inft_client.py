"""
Async client for the Peerbit sidecar's thought store.
Used by the L1 miner to read prior context and write new thoughts.
All operations degrade gracefully — never block or crash inference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import quote

log = logging.getLogger("inft_client")


class InftClient:
    """
    Thin async HTTP wrapper around the Peerbit sidecar's /thoughts endpoints.

    If url is None (or the sidecar cannot be reached), every method is a no-op.

    Design constraints:
      - search_context has a 2-second timeout.
      - ingest has a 5-second timeout.
      - All exceptions are caught and logged at DEBUG level.
    """

    def __init__(self, url: Optional[str]) -> None:
        self._url: Optional[str] = url.rstrip("/") if url else None
        self._session = None  # aiohttp.ClientSession

    async def start(self) -> None:
        if self._url is None:
            return
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            log.info("inft_client_connected url=%s", self._url)
        except Exception as exc:
            log.warning("inft_client_unavailable url=%s err=%s", self._url, exc)
            self._session = None

    async def stop(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as exc:
                log.debug("inft_client_stop_err err=%s", exc)
            self._session = None

    async def search_context(
        self,
        question: str,
        model_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Query the Peerbit sidecar for prior thoughts relevant to `question`.
        Returns [] on timeout, sidecar unavailable, or any error.
        """
        if self._session is None:
            return []
        try:
            async with asyncio.timeout(2.0):
                async with self._session.get(
                    f"{self._url}/thoughts/search",
                    params={"q": question, "model_id": model_id, "limit": limit},
                ) as r:
                    body = await r.json()
                    rows = body.get("data") if body.get("ok") else []
                    return [
                        {
                            "id": row.get("id", 0),
                            "job_id": row.get("job_id", ""),
                            "miner_address": row.get("miner_address", ""),
                            "model_id": row.get("model_id", ""),
                            "question_text": row.get("question_text", ""),
                            "thinking_text": row.get("thinking_text", ""),
                            "answer_text": row.get("answer_text", ""),
                            "score": float(row.get("score", 0.0)),
                        }
                        for row in (rows or [])
                    ]
        except TimeoutError:
            log.debug("inft_client_search_timeout question=%r", question[:40])
            return []
        except Exception as exc:
            log.debug("inft_client_search_err question=%r err=%s", question[:40], exc)
            return []

    async def ingest(
        self,
        job_id: str,
        miner_address: str,
        model_id: str,
        question: str,
        thinking: str,
        answer: str,
        proof_sig: str,
        block_number: Optional[int] = None,
        tx_hash: Optional[str] = None,
    ) -> bool:
        """
        Persist a completed inference record into the Peerbit distributed store.
        Returns True on success, False on error.
        """
        if self._session is None:
            return False
        try:
            async with asyncio.timeout(5.0):
                async with self._session.post(
                    f"{self._url}/thoughts",
                    json={
                        "job_id": job_id,
                        "miner_address": miner_address,
                        "model_id": model_id,
                        "question": question,
                        "thinking": thinking,
                        "answer": answer,
                        "proof_sig": proof_sig,
                        "block_number": block_number,
                        "tx_hash": tx_hash,
                    },
                ) as r:
                    body = await r.json()
                    success = bool(body.get("ok") and body.get("data", {}).get("ingested"))
                    log.debug("inft_client_ingested job=%s ok=%s", job_id, success)
                    return success
        except TimeoutError:
            log.debug("inft_client_ingest_timeout job=%s", job_id)
            return False
        except Exception as exc:
            log.debug("inft_client_ingest_err job=%s err=%s", job_id, exc)
            return False
