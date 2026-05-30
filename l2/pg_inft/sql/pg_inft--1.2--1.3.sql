-- pg_inft--1.2--1.3.sql
-- Distributed miner registry, job board, and reputation ledger.
-- Replaces the OrbitDB/Peerbit Node.js sidecar that previously handled this
-- data via gossipsub-replicated document stores.
-- \echo Use "ALTER EXTENSION pg_inft UPDATE TO '1.3'" to load this file. \quit

SET search_path = inft, public;

-- ── inft_miners ───────────────────────────────────────────────────────────────
-- One row per miner address. Upserted on startup and every heartbeat (60s).
-- active=false means the miner deregistered cleanly or went stale.

CREATE TABLE IF NOT EXISTS inft.inft_miners (
    address        text        PRIMARY KEY,
    models         text[]      NOT NULL DEFAULT '{}',
    backend        text        NOT NULL DEFAULT 'cpu',
    p2p_addr       text        NOT NULL DEFAULT '',
    l2_chain_id    text        NOT NULL DEFAULT '2026',
    max_shards     int         NOT NULL DEFAULT 4,
    reputation     int         NOT NULL DEFAULT 500,
    active         boolean     NOT NULL DEFAULT true,
    registered_at  timestamptz NOT NULL DEFAULT now(),
    last_seen      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS inft_miners_active_idx
    ON inft.inft_miners (active, last_seen DESC);

CREATE INDEX IF NOT EXISTS inft_miners_models_idx
    ON inft.inft_miners USING GIN (models);

COMMENT ON TABLE inft.inft_miners IS
    'Active miner registry. Replaces OrbitDB inferencechain.miners.v1 document store.';

-- ── inft_jobs ─────────────────────────────────────────────────────────────────
-- Lightweight job board: one row per job_id, upserted as shards progress.

CREATE TABLE IF NOT EXISTS inft.inft_jobs (
    job_id       text        PRIMARY KEY,
    model_id     text        NOT NULL DEFAULT '',
    mode         text        NOT NULL DEFAULT 'parallel_sample',
    n_shards     int         NOT NULL DEFAULT 1,
    status       text        NOT NULL DEFAULT 'pending',
    output_hash  text        NOT NULL DEFAULT '',
    requester    text        NOT NULL DEFAULT '',
    latency_ms   int         NOT NULL DEFAULT 0,
    posted_at    timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE INDEX IF NOT EXISTS inft_jobs_status_idx
    ON inft.inft_jobs (status, posted_at DESC);

COMMENT ON TABLE inft.inft_jobs IS
    'Job activity log. Replaces OrbitDB inferencechain.jobs.v1 document store.';

-- ── inft_reputation_events ────────────────────────────────────────────────────
-- Append-only reputation ledger. delta > 0 for success, < 0 for failure/slash.

CREATE TABLE IF NOT EXISTS inft.inft_reputation_events (
    event_id   text        PRIMARY KEY,
    miner      text        NOT NULL,
    event_type text        NOT NULL,
    delta      int         NOT NULL DEFAULT 0,
    job_id     text        NOT NULL DEFAULT '',
    shard_idx  int         NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS inft_reputation_events_miner_idx
    ON inft.inft_reputation_events (miner, created_at DESC);

COMMENT ON TABLE inft.inft_reputation_events IS
    'Reputation event ledger. Replaces OrbitDB inferencechain.events.v1 event store.';

-- ── inft_upsert_miner ────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_upsert_miner(
    p_address     text,
    p_models      text[],
    p_backend     text,
    p_p2p_addr    text,
    p_chain_id    text,
    p_max_shards  int,
    p_active      boolean DEFAULT true
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO inft.inft_miners
        (address, models, backend, p2p_addr, l2_chain_id, max_shards,
         active, registered_at, last_seen)
    VALUES
        (p_address, p_models, p_backend, p_p2p_addr, p_chain_id, p_max_shards,
         p_active, now(), now())
    ON CONFLICT (address) DO UPDATE SET
        models      = EXCLUDED.models,
        backend     = EXCLUDED.backend,
        p2p_addr    = EXCLUDED.p2p_addr,
        l2_chain_id = EXCLUDED.l2_chain_id,
        max_shards  = EXCLUDED.max_shards,
        active      = EXCLUDED.active,
        last_seen   = now();
END;
$$;

-- ── inft_deregister_miner ────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_deregister_miner(p_address text)
RETURNS VOID LANGUAGE sql AS $$
    UPDATE inft.inft_miners SET active = false, last_seen = now()
    WHERE  address = p_address;
$$;

-- ── inft_upsert_job ──────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_upsert_job(
    p_job_id      text,
    p_model_id    text,
    p_mode        text,
    p_n_shards    int,
    p_status      text,
    p_output_hash text,
    p_requester   text,
    p_latency_ms  int
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO inft.inft_jobs
        (job_id, model_id, mode, n_shards, status, output_hash,
         requester, latency_ms, posted_at, completed_at)
    VALUES
        (p_job_id, p_model_id, p_mode, p_n_shards, p_status, p_output_hash,
         p_requester, p_latency_ms, now(),
         CASE WHEN p_status = 'complete' THEN now() ELSE NULL END)
    ON CONFLICT (job_id) DO UPDATE SET
        status       = EXCLUDED.status,
        output_hash  = EXCLUDED.output_hash,
        latency_ms   = EXCLUDED.latency_ms,
        completed_at = CASE
            WHEN EXCLUDED.status = 'complete' THEN now()
            ELSE inft.inft_jobs.completed_at
        END;
END;
$$;

-- ── inft_log_reputation_event ────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_log_reputation_event(
    p_event_id   text,
    p_miner      text,
    p_event_type text,
    p_delta      int,
    p_job_id     text,
    p_shard_idx  int
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO inft.inft_reputation_events
        (event_id, miner, event_type, delta, job_id, shard_idx, created_at)
    VALUES
        (p_event_id, p_miner, p_event_type, p_delta, p_job_id, p_shard_idx, now())
    ON CONFLICT (event_id) DO NOTHING;
    -- Also update running reputation total on the miner row (clamped 0-1000)
    UPDATE inft.inft_miners
    SET reputation = GREATEST(0, LEAST(1000, reputation + p_delta))
    WHERE address = p_miner;
END;
$$;

-- ── inft_get_miners_for_model ────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION inft.inft_get_miners_for_model(p_model_id text)
RETURNS TABLE(
    address    text,
    models     text[],
    backend    text,
    p2p_addr   text,
    max_shards int,
    reputation int,
    last_seen  timestamptz
) LANGUAGE sql STABLE AS $$
    SELECT address, models, backend, p2p_addr, max_shards, reputation, last_seen
    FROM   inft.inft_miners
    WHERE  active    = true
      AND  p_model_id = ANY(models)
      AND  last_seen > now() - INTERVAL '5 minutes'
    ORDER  BY reputation DESC, last_seen DESC;
$$;

-- ── inft_cleanup_stale_miners ────────────────────────────────────────────────
-- Marks miners inactive if their heartbeat is older than the given interval.
-- Call periodically (e.g. every 5 minutes) to mirror OrbitDB's gossipsub TTL.

CREATE OR REPLACE FUNCTION inft.inft_cleanup_stale_miners(
    p_stale_after interval DEFAULT '5 minutes'
) RETURNS int LANGUAGE sql AS $$
    WITH updated AS (
        UPDATE inft.inft_miners
        SET    active = false
        WHERE  active    = true
          AND  last_seen < now() - p_stale_after
        RETURNING address
    )
    SELECT count(*)::int FROM updated;
$$;
