"""
ThoughtStore — asyncpg interface to the pg_inft PostgreSQL extension.

Wraps all read/write operations with graceful degradation:
if PostgreSQL is unavailable, all methods are no-ops and inference continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("thought_store")


@dataclass
class ThoughtResult:
    """A single row returned by inft.inft_search."""
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
    Async interface to the pg_inft extension.

    All public methods degrade gracefully: if the PostgreSQL connection pool
    is unavailable (self._pool is None), every operation is a no-op that
    returns an empty/False result without raising.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn: str = dsn
        self._pool = None  # asyncpg.Pool, set by connect()

    async def connect(self) -> None:
        """Create the asyncpg connection pool.  Logs a warning on failure."""
        try:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=4,
                command_timeout=10,
            )
            log.info("thought_store_connected dsn=%s", self._dsn[:40])
        except Exception as exc:
            log.warning(
                "thought_store_unavailable dsn=%s err=%s — memory rollup disabled",
                self._dsn[:40], exc,
            )
            self._pool = None

    async def close(self) -> None:
        """Close the connection pool if it was opened."""
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:
                log.debug("thought_store_close_err err=%s", exc)
            self._pool = None

    async def search(
        self,
        question: str,
        model_id: str = "",
        limit: int = 5,
    ) -> list[ThoughtResult]:
        """
        Search stored thoughts using the pg_inft staged BM25+trigram pipeline.

        Returns an empty list if the pool is unavailable or any error occurs.
        """
        if self._pool is None:
            return []
        try:
            rows = await self._pool.fetch(
                "SELECT id, job_id, miner_address, model_id, "
                "       question_text, thinking_text, answer_text, score "
                "FROM inft.inft_search($1, $2, $3)",
                question,
                model_id,
                limit,
            )
            results = []
            for row in rows:
                results.append(ThoughtResult(
                    id=row["id"],
                    job_id=row["job_id"] or "",
                    miner_address=row["miner_address"] or "",
                    model_id=row["model_id"] or "",
                    question_text=row["question_text"] or "",
                    thinking_text=row["thinking_text"] or "",
                    answer_text=row["answer_text"] or "",
                    score=float(row["score"] or 0.0),
                ))
            return results
        except Exception as exc:
            log.debug("thought_store_search_err question=%r err=%s", question[:40], exc)
            return []

    async def recent(
        self,
        model_id: str = "",
        limit: int = 20,
    ) -> list[ThoughtResult]:
        """
        Return the most recently ingested thoughts ordered by recency.

        Unlike search(), this does NOT run a full-text query — an empty BM25
        query matches no lexemes and returns nothing, so the "recent" view must
        read inft_thought_log directly. Optional model_id filter. Score is 0.0
        (recency view is not ranked). Returns [] if the pool is unavailable.
        """
        if self._pool is None:
            return []
        try:
            cols = ("id, job_id, miner_address, model_id, "
                    "question_text, thinking_text, answer_text")
            if model_id:
                rows = await self._pool.fetch(
                    f"SELECT {cols} FROM inft.inft_thought_log "
                    "WHERE model_id = $1 ORDER BY id DESC LIMIT $2",
                    model_id, limit,
                )
            else:
                rows = await self._pool.fetch(
                    f"SELECT {cols} FROM inft.inft_thought_log "
                    "ORDER BY id DESC LIMIT $1",
                    limit,
                )
            return [
                ThoughtResult(
                    id=row["id"],
                    job_id=row["job_id"] or "",
                    miner_address=row["miner_address"] or "",
                    model_id=row["model_id"] or "",
                    question_text=row["question_text"] or "",
                    thinking_text=row["thinking_text"] or "",
                    answer_text=row["answer_text"] or "",
                    score=0.0,
                )
                for row in rows
            ]
        except Exception as exc:
            log.debug("thought_store_recent_err err=%s", exc)
            return []

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
        """
        Ingest a completed inference record into pg_inft.

        proof_sig_hex is a 0x-prefixed hex string (65 bytes = 130 hex chars).
        Returns False on any error.
        """
        if self._pool is None:
            return False
        try:
            # Convert 0x-prefixed hex signature to raw bytes
            hex_str = proof_sig_hex.lstrip("0x") if proof_sig_hex else ""
            if not hex_str:
                log.debug("thought_store_ingest_no_sig job=%s", job_id)
                return False
            sig_bytes = bytes.fromhex(hex_str)

            row = await self._pool.fetchrow(
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
                peer_origin,
            )
            thought_id = row["thought_id"] if row else None
            log.debug(
                "thought_store_ingested job=%s thought_id=%s", job_id, thought_id
            )
            return thought_id is not None
        except Exception as exc:
            log.debug("thought_store_ingest_err job=%s err=%s", job_id, exc)
            return False

    async def update_live_tps(
        self,
        miner_address: str,
        model_id:      str,
        actual_tps:    float,
    ) -> None:
        """
        Update the EWMA production throughput for this miner/model pair.

        Called after each completed shard job so the sequencer can use live
        performance data alongside the formal benchmark score.
        alpha=0.3 is baked into the SQL function (inft.inft_update_live_tps).
        """
        if self._pool is None:
            return
        try:
            await self._pool.execute(
                "SELECT inft.inft_update_live_tps($1, $2, $3::float8)",
                miner_address,
                model_id,
                float(actual_tps),
            )
        except Exception as exc:
            log.debug(
                "thought_store_live_tps_err miner=%s model=%s err=%s",
                miner_address[:10], model_id, exc,
            )

    async def get_live_tps(
        self,
        miner_address: str,
        model_id:      str,
    ) -> Optional[dict]:
        """
        Return the latest benchmark + live TPS info for this miner/model pair.
        Returns None if pg_inft is unavailable or no row exists.
        Dict keys: tokens_per_sec, live_tps, live_sample_count.
        """
        if self._pool is None:
            return None
        try:
            row = await self._pool.fetchrow(
                "SELECT tokens_per_sec, live_tps, live_sample_count "
                "FROM inft.inft_get_benchmark($1, $2)",
                miner_address,
                model_id,
            )
            return dict(row) if row else None
        except Exception as exc:
            log.debug(
                "thought_store_get_tps_err miner=%s model=%s err=%s",
                miner_address[:10], model_id, exc,
            )
            return None

    async def set_job_context(
        self,
        job_id:       str,
        query_text:   str,
        context_text: str,
        context_hash: str,
        model_id:     str,
        n_entries:    int,
    ) -> bool:
        """
        Write sequencer-assembled context for a job into the local pg_inft store.

        Called at job-dispatch time (sequencer side) and on ContextLoadOffer receipt
        (miner side) so every node has the pre-fetched context ready before inference
        begins.  Returns False on any error without raising.
        """
        if self._pool is None:
            return False
        try:
            await self._pool.execute(
                "SELECT inft.inft_set_job_context($1,$2,$3,$4,$5,$6)",
                job_id, query_text, context_text, context_hash, model_id, n_entries,
            )
            return True
        except Exception as exc:
            log.debug("thought_store_set_ctx_err job=%s err=%s", job_id, exc)
            return False

    async def get_job_context(self, job_id: str) -> Optional[dict]:
        """
        Retrieve pre-fetched context for a job from the local pg_inft store.

        Returns a dict with keys {query_text, context_text, context_hash, model_id,
        n_entries}, or None if not found or expired.  Used by miners on the inference
        hot path to skip the sequential ThoughtStore.search() call.
        """
        if self._pool is None:
            return None
        try:
            row = await self._pool.fetchrow(
                "SELECT query_text, context_text, context_hash, model_id, n_entries "
                "FROM inft.inft_get_job_context($1)",
                job_id,
            )
            return dict(row) if row else None
        except Exception as exc:
            log.debug("thought_store_get_ctx_err job=%s err=%s", job_id, exc)
            return None

    async def expire_job_contexts(self) -> int:
        """Delete expired job context rows.  Returns the number of rows deleted."""
        if self._pool is None:
            return 0
        try:
            row = await self._pool.fetchrow(
                "SELECT inft.inft_expire_job_contexts() AS n"
            )
            return int(row["n"]) if row else 0
        except Exception as exc:
            log.debug("thought_store_expire_ctx_err err=%s", exc)
            return 0

    async def record_peer(
        self,
        peer_address: str,
        job_id: str,
        rejected: bool,
    ) -> None:
        """
        Upsert a row in inft.inft_peer_sync tracking peer statistics.
        No-op if pool is unavailable.
        """
        if self._pool is None:
            return
        try:
            await self._pool.execute(
                """
                INSERT INTO inft.inft_peer_sync
                    (peer_address, last_seen, thoughts_received, proofs_rejected, last_job_id)
                VALUES ($1, now(), 1, $2::int, $3)
                ON CONFLICT (peer_address) DO UPDATE SET
                    last_seen         = now(),
                    thoughts_received = inft.inft_peer_sync.thoughts_received + 1,
                    proofs_rejected   = inft.inft_peer_sync.proofs_rejected + EXCLUDED.proofs_rejected,
                    last_job_id       = EXCLUDED.last_job_id
                """,
                peer_address,
                1 if rejected else 0,
                job_id,
            )
        except Exception as exc:
            log.debug(
                "thought_store_record_peer_err peer=%s err=%s", peer_address, exc
            )
