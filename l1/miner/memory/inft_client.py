"""
Async client for the pg_inft PostgreSQL extension.
Used by the miner to read prior context and write new thoughts.
All operations degrade gracefully — never block or crash inference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

log = logging.getLogger("inft_client")


class InftClient:
    """
    Thin async wrapper around the pg_inft extension via asyncpg.

    If dsn is None (or the connection pool cannot be established), every
    method is a no-op that returns an empty result without raising.

    Design constraints:
      - search_context has a 2-second timeout.
      - ingest has a 5-second timeout.
      - All exceptions are caught and logged at DEBUG level so inference
        is never interrupted by a database issue.
    """

    def __init__(self, dsn: Optional[str]) -> None:
        self._dsn: Optional[str] = dsn
        self._pool = None  # asyncpg.Pool

    async def start(self) -> None:
        """Open the asyncpg connection pool.  No-op if dsn is None."""
        if self._dsn is None:
            return
        try:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=4,
                command_timeout=10,
            )
            log.info("inft_client_connected dsn=%s", self._dsn[:40])
        except Exception as exc:
            log.warning(
                "inft_client_unavailable dsn=%s err=%s — memory rollup disabled",
                self._dsn[:40], exc,
            )
            self._pool = None

    async def stop(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:
                log.debug("inft_client_stop_err err=%s", exc)
            self._pool = None

    async def search_context(
        self,
        question: str,
        model_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Query inft.inft_search for prior thoughts relevant to `question`.

        Returns a list of dicts with keys:
          id, job_id, miner_address, model_id,
          question_text, thinking_text, answer_text, score

        Returns [] on timeout, pool unavailable, or any error.
        """
        if self._pool is None:
            return []
        try:
            rows = await asyncio.wait_for(
                self._pool.fetch(
                    "SELECT id, job_id, miner_address, model_id, "
                    "       question_text, thinking_text, answer_text, score "
                    "FROM inft.inft_search($1, $2, $3)",
                    question,
                    model_id,
                    limit,
                ),
                timeout=2.0,
            )
            return [
                {
                    "id":            row["id"],
                    "job_id":        row["job_id"] or "",
                    "miner_address": row["miner_address"] or "",
                    "model_id":      row["model_id"] or "",
                    "question_text": row["question_text"] or "",
                    "thinking_text": row["thinking_text"] or "",
                    "answer_text":   row["answer_text"] or "",
                    "score":         float(row["score"] or 0.0),
                }
                for row in rows
            ]
        except asyncio.TimeoutError:
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
        Persist a completed inference record into pg_inft via inft_ingest().

        proof_sig is a 0x-prefixed hex string (65 bytes Ethereum ECDSA sig).
        Returns True on success, False on error.
        """
        if self._pool is None:
            return False
        try:
            # Convert 0x-prefixed hex to raw bytes
            hex_str = proof_sig.lstrip("0x") if proof_sig else ""
            if not hex_str:
                log.debug("inft_client_ingest_no_sig job=%s", job_id)
                return False
            sig_bytes = bytes.fromhex(hex_str)

            row = await asyncio.wait_for(
                self._pool.fetchrow(
                    "SELECT inft.inft_ingest("
                    "  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10"
                    ") AS thought_id",
                    job_id,
                    miner_address,
                    model_id,
                    question,
                    thinking,
                    answer,
                    sig_bytes,
                    block_number,
                    tx_hash,
                    None,          # peer_origin — not applicable for own output
                ),
                timeout=5.0,
            )
            thought_id = row["thought_id"] if row else None
            log.debug(
                "inft_client_ingested job=%s thought_id=%s", job_id, thought_id
            )
            return thought_id is not None
        except asyncio.TimeoutError:
            log.debug("inft_client_ingest_timeout job=%s", job_id)
            return False
        except Exception as exc:
            log.debug("inft_client_ingest_err job=%s err=%s", job_id, exc)
            return False
