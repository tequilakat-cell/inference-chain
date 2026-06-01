-- pg_inft--1.3--1.4.sql
-- Semantic memory layer: pgvector embeddings on thoughts + consolidated rollup
-- memories. Enables semantic clustering ("messages similar to X") and stores the
-- distilled, gossiped rollup that future inferences inject.
--
-- REQUIRES the `vector` extension (pgvector) to be created first:
--     CREATE EXTENSION IF NOT EXISTS vector;
-- Embedding model: nomic-embed-text-v1.5 → 768 dimensions.
-- \echo Use "ALTER EXTENSION pg_inft UPDATE TO '1.4'" to load this file. \quit

SET search_path = inft, public;

-- ── Embeddings on thoughts ──────────────────────────────────────────────────
ALTER TABLE inft.inft_thought_log
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Cosine-distance ANN index (HNSW: usable immediately, no training step).
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS inft_thought_log_embedding_idx
        ON inft.inft_thought_log USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN undefined_object OR feature_not_supported THEN
    RAISE NOTICE 'hnsw index unavailable; sequential scan will be used for embeddings';
END$$;

-- ── Consolidated rollup memories ────────────────────────────────────────────
-- One row per rollup: a distilled summary of a cluster of similar thoughts.
CREATE TABLE IF NOT EXISTS inft.inft_rollups (
    id             bigint      PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    rollup_id      text        UNIQUE NOT NULL,        -- uuid (derived from the reduce job)
    topic          text        NOT NULL DEFAULT '',    -- query/topic this rollup summarizes
    model_id       text        NOT NULL DEFAULT '',
    summary_text   text        NOT NULL,               -- the distilled memory
    source_count   int         NOT NULL DEFAULT 0,     -- # thoughts consolidated
    source_job_ids text[]      NOT NULL DEFAULT '{}',
    embedding      vector(768),                        -- embedding of the topic/summary
    content_hash   bytea,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS inft_rollups_model_idx
    ON inft.inft_rollups (model_id, created_at DESC);

DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS inft_rollups_embedding_idx
        ON inft.inft_rollups USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN undefined_object OR feature_not_supported THEN
    RAISE NOTICE 'hnsw index unavailable for rollups; sequential scan will be used';
END$$;

COMMENT ON TABLE inft.inft_rollups IS
    'Consolidated (summarized) memories distilled from clusters of similar thoughts.';

-- ── Write a thought embedding (ingest / backfill) ──────────────────────────
CREATE OR REPLACE FUNCTION inft.inft_set_embedding(
    p_job_id    text,
    p_embedding vector(768)
) RETURNS boolean LANGUAGE plpgsql AS $$
BEGIN
    UPDATE inft.inft_thought_log SET embedding = p_embedding WHERE job_id = p_job_id;
    RETURN FOUND;
END$$;

-- ── Semantic search over thoughts (caller supplies the query embedding) ────
CREATE OR REPLACE FUNCTION inft.inft_search_semantic(
    p_query    vector(768),
    p_model_id text DEFAULT '',
    p_limit    int  DEFAULT 20
) RETURNS TABLE(
    id            bigint,
    job_id        text,
    miner_address text,
    model_id      text,
    question_text text,
    thinking_text text,
    answer_text   text,
    score         float8
) LANGUAGE sql STABLE AS $$
    SELECT id, job_id, miner_address, model_id,
           question_text, thinking_text, answer_text,
           (1 - (embedding <=> p_query))::float8 AS score
    FROM   inft.inft_thought_log
    WHERE  embedding IS NOT NULL
      AND  (p_model_id = '' OR model_id = p_model_id)
    ORDER  BY embedding <=> p_query
    LIMIT  p_limit;
$$;

-- ── Upsert a rollup memory ──────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION inft.inft_upsert_rollup(
    p_rollup_id      text,
    p_topic          text,
    p_model_id       text,
    p_summary        text,
    p_source_count   int,
    p_source_job_ids text[],
    p_embedding      vector(768),
    p_content_hash   bytea
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO inft.inft_rollups
        (rollup_id, topic, model_id, summary_text, source_count,
         source_job_ids, embedding, content_hash)
    VALUES
        (p_rollup_id, p_topic, p_model_id, p_summary, p_source_count,
         p_source_job_ids, p_embedding, p_content_hash)
    ON CONFLICT (rollup_id) DO UPDATE SET
        topic          = EXCLUDED.topic,
        summary_text   = EXCLUDED.summary_text,
        source_count   = EXCLUDED.source_count,
        source_job_ids = EXCLUDED.source_job_ids,
        embedding      = EXCLUDED.embedding,
        content_hash   = EXCLUDED.content_hash;
END$$;

-- ── Semantic search over rollups ────────────────────────────────────────────
CREATE OR REPLACE FUNCTION inft.inft_search_rollups(
    p_query    vector(768),
    p_model_id text DEFAULT '',
    p_limit    int  DEFAULT 5
) RETURNS TABLE(
    rollup_id    text,
    topic        text,
    model_id     text,
    summary_text text,
    source_count int,
    score        float8
) LANGUAGE sql STABLE AS $$
    SELECT rollup_id, topic, model_id, summary_text, source_count,
           (1 - (embedding <=> p_query))::float8 AS score
    FROM   inft.inft_rollups
    WHERE  embedding IS NOT NULL
      AND  (p_model_id = '' OR model_id = p_model_id)
    ORDER  BY embedding <=> p_query
    LIMIT  p_limit;
$$;
