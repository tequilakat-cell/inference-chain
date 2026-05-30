-- pg_inft--1.1--1.2.sql
-- Proactive Parallel Pre-fetch: per-job context staging table.
-- The sequencer searches pg_inft at job-dispatch time and writes the assembled
-- context here so miners can pull it from their LOCAL postgres replica instead
-- of performing a sequential search on the inference hot path.

SET search_path = inft, public;

-- ── inft_job_context ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS inft.inft_job_context (
    job_id        TEXT        PRIMARY KEY,
    query_text    TEXT        NOT NULL DEFAULT '',
    context_text  TEXT        NOT NULL DEFAULT '',
    context_hash  TEXT        NOT NULL DEFAULT '',
    model_id      TEXT        NOT NULL DEFAULT '',
    n_entries     INT         NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '10 minutes'
);

CREATE INDEX IF NOT EXISTS inft_job_context_expires_idx
    ON inft.inft_job_context (expires_at);

-- ── inft_set_job_context ─────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_set_job_context(
    p_job_id       TEXT,
    p_query_text   TEXT,
    p_context_text TEXT,
    p_context_hash TEXT,
    p_model_id     TEXT,
    p_n_entries    INT
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO inft.inft_job_context
        (job_id, query_text, context_text, context_hash, model_id, n_entries,
         created_at, expires_at)
    VALUES
        (p_job_id, p_query_text, p_context_text, p_context_hash, p_model_id,
         p_n_entries, now(), now() + INTERVAL '10 minutes')
    ON CONFLICT (job_id) DO UPDATE SET
        query_text   = EXCLUDED.query_text,
        context_text = EXCLUDED.context_text,
        context_hash = EXCLUDED.context_hash,
        model_id     = EXCLUDED.model_id,
        n_entries    = EXCLUDED.n_entries,
        expires_at   = now() + INTERVAL '10 minutes';
END;
$$;

-- ── inft_get_job_context ─────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_get_job_context(p_job_id TEXT)
RETURNS TABLE(
    query_text   TEXT,
    context_text TEXT,
    context_hash TEXT,
    model_id     TEXT,
    n_entries    INT
) LANGUAGE sql STABLE AS $$
    SELECT query_text, context_text, context_hash, model_id, n_entries
    FROM   inft.inft_job_context
    WHERE  job_id    = p_job_id
      AND  expires_at > now();
$$;

-- ── inft_expire_job_contexts ─────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_expire_job_contexts()
RETURNS INT LANGUAGE sql AS $$
    WITH deleted AS (
        DELETE FROM inft.inft_job_context
        WHERE expires_at <= now()
        RETURNING job_id
    )
    SELECT count(*)::INT FROM deleted;
$$;
