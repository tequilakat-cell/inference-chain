-- pg_inft--1.0--1.1.sql
-- Migration: add miner benchmark score tracking.
--
-- New table: inft.inft_miner_benchmarks
--   Stores sequencer-measured benchmark scores (tokens_per_sec) and a
--   miner-reported exponential moving average of production throughput (live_tps).
--
-- New functions:
--   inft.inft_upsert_benchmark()  — called by chain on BENCHMARK_COMMIT tx
--   inft.inft_update_live_tps()   — called by miner after each completed shard
--   inft.inft_get_benchmark()     — point lookup by (miner_address, model_id)
-- \echo Use "ALTER EXTENSION pg_inft UPDATE TO '1.1'" to load this file. \quit

-- ── Miner benchmark table ──────────────────────────────────────────────────────

CREATE TABLE inft.inft_miner_benchmarks (
    miner_address     text      NOT NULL,
    model_id          text      NOT NULL,

    -- Sequencer-measured score (written on BENCHMARK_COMMIT tx).
    -- tokens_per_sec = n_tokens / (elapsed_ms / 1000).  Sequencer measures
    -- wall-clock time; miner never self-reports elapsed.
    tokens_per_sec    float8    NOT NULL DEFAULT 0,
    n_tokens          int       NOT NULL DEFAULT 0,
    elapsed_ms        int       NOT NULL DEFAULT 0,
    nonce             text      NOT NULL DEFAULT '',
    block_number      bigint    NOT NULL DEFAULT 0,
    expires_at_block  bigint    NOT NULL DEFAULT 0,

    -- Miner-reported production throughput (EWMA, alpha=0.3).
    -- Updated after each completed shard job via inft_update_live_tps().
    -- Calculated by miner from actual output length / inference elapsed.
    live_tps          float8,
    live_sample_count int       NOT NULL DEFAULT 0,
    last_live_update  timestamptz,

    first_seen        timestamptz NOT NULL DEFAULT now(),
    last_updated      timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (miner_address, model_id)
);

COMMENT ON TABLE inft.inft_miner_benchmarks IS
    'Per-(miner, model) benchmark scores and live production throughput tracking. '
    'tokens_per_sec is sequencer-measured (tamper-proof). '
    'live_tps is miner-reported EWMA from actual job completions.';

CREATE INDEX inft_miner_benchmarks_model_idx
    ON inft.inft_miner_benchmarks (model_id);

-- ── inft_upsert_benchmark ─────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_upsert_benchmark(
    p_miner_address   text,
    p_model_id        text,
    p_tokens_per_sec  float8,
    p_n_tokens        int,
    p_elapsed_ms      int,
    p_nonce           text,
    p_block_number    bigint,
    p_expires_at_block bigint
)
    RETURNS void
    LANGUAGE plpgsql
    VOLATILE
AS $$
BEGIN
    INSERT INTO inft.inft_miner_benchmarks (
        miner_address, model_id, tokens_per_sec, n_tokens, elapsed_ms,
        nonce, block_number, expires_at_block, last_updated
    ) VALUES (
        lower(p_miner_address), p_model_id, p_tokens_per_sec, p_n_tokens,
        p_elapsed_ms, p_nonce, p_block_number, p_expires_at_block, now()
    )
    ON CONFLICT (miner_address, model_id) DO UPDATE SET
        tokens_per_sec   = EXCLUDED.tokens_per_sec,
        n_tokens         = EXCLUDED.n_tokens,
        elapsed_ms       = EXCLUDED.elapsed_ms,
        nonce            = EXCLUDED.nonce,
        block_number     = EXCLUDED.block_number,
        expires_at_block = EXCLUDED.expires_at_block,
        last_updated     = now();
END;
$$;

COMMENT ON FUNCTION inft.inft_upsert_benchmark(text,text,float8,int,int,text,bigint,bigint) IS
    'Upsert a sequencer-measured benchmark score. Called on BENCHMARK_COMMIT tx application.';

-- ── inft_update_live_tps ──────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_update_live_tps(
    p_miner_address text,
    p_model_id      text,
    p_actual_tps    float8
)
    RETURNS void
    LANGUAGE plpgsql
    VOLATILE
AS $$
DECLARE
    v_alpha float8 := 0.3;   -- EWMA weight for new sample
BEGIN
    INSERT INTO inft.inft_miner_benchmarks (
        miner_address, model_id, tokens_per_sec, n_tokens, elapsed_ms,
        live_tps, live_sample_count, last_live_update, last_updated
    ) VALUES (
        lower(p_miner_address), p_model_id, 0, 0, 0,
        p_actual_tps, 1, now(), now()
    )
    ON CONFLICT (miner_address, model_id) DO UPDATE SET
        live_tps = CASE
            WHEN inft.inft_miner_benchmarks.live_tps IS NULL
                THEN p_actual_tps
            ELSE v_alpha * p_actual_tps
               + (1.0 - v_alpha) * inft.inft_miner_benchmarks.live_tps
        END,
        live_sample_count = inft.inft_miner_benchmarks.live_sample_count + 1,
        last_live_update  = now(),
        last_updated      = now();
END;
$$;

COMMENT ON FUNCTION inft.inft_update_live_tps(text, text, float8) IS
    'Update production throughput EWMA (alpha=0.3) for a miner/model pair. '
    'Called by the miner after each completed shard job. '
    'Creates a provisional row if no benchmark score exists yet.';

-- ── inft_get_benchmark ────────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_get_benchmark(
    p_miner_address text,
    p_model_id      text
)
    RETURNS TABLE (
        tokens_per_sec    float8,
        live_tps          float8,
        live_sample_count int,
        block_number      bigint,
        expires_at_block  bigint,
        nonce             text,
        last_updated      timestamptz
    )
    LANGUAGE sql
    STABLE
AS $$
    SELECT
        tokens_per_sec,
        live_tps,
        live_sample_count,
        block_number,
        expires_at_block,
        nonce,
        last_updated
    FROM inft.inft_miner_benchmarks
    WHERE miner_address = lower(p_miner_address)
      AND model_id      = p_model_id;
$$;

COMMENT ON FUNCTION inft.inft_get_benchmark(text, text) IS
    'Return benchmark score and live production throughput for a miner/model pair.';

-- ── Network benchmark summary view ────────────────────────────────────────────

CREATE VIEW inft.inft_benchmark_summary AS
SELECT
    miner_address,
    model_id,
    tokens_per_sec                                                AS bench_tps,
    live_tps,
    live_sample_count,
    CASE WHEN live_tps IS NOT NULL AND live_sample_count >= 5
         THEN live_tps
         ELSE tokens_per_sec
    END                                                           AS effective_tps,
    block_number,
    expires_at_block,
    last_updated
FROM inft.inft_miner_benchmarks
ORDER BY effective_tps DESC NULLS LAST;

COMMENT ON VIEW inft.inft_benchmark_summary IS
    'Per-miner benchmark and live TPS summary. '
    'effective_tps = live_tps (when >=5 samples) else bench_tps.';
