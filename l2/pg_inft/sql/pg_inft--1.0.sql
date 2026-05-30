-- pg_inft--1.0.sql
-- Full schema for the pg_inft distributed LLM memory rollup extension.
--
-- Install with:
--   CREATE EXTENSION pg_inft;
--
-- Requires plpython3u for inft_eth_verify().
-- Requires pg_trgm for trigram search (gracefully degraded if absent).
-- \echo Use "CREATE EXTENSION pg_inft" to load this file.  \quit

-- ── C-backed utility functions ─────────────────────────────────────────────

CREATE FUNCTION inft.pg_inft_version()
    RETURNS text
    LANGUAGE c STRICT IMMUTABLE
    AS '$libdir/pg_inft', 'pg_inft_version';

COMMENT ON FUNCTION inft.pg_inft_version() IS
    'Returns the pg_inft extension version string.';

-- ── Keccak-256 hash function ───────────────────────────────────────────────

CREATE FUNCTION inft.inft_keccak256(input bytea)
    RETURNS bytea
    LANGUAGE c STRICT IMMUTABLE PARALLEL SAFE
    AS '$libdir/pg_inft', 'inft_keccak256';

COMMENT ON FUNCTION inft.inft_keccak256(bytea) IS
    'Keccak-256 hash (domain separation 0x01, NOT SHA3). '
    'keccak256(''\\x''::bytea) = \\xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470';

-- ── Content hash (for proof generation) ───────────────────────────────────

CREATE FUNCTION inft.inft_content_hash(
    job_id      text,
    question    text,
    thinking    text,
    answer      text
)
    RETURNS bytea
    LANGUAGE c STRICT IMMUTABLE PARALLEL SAFE
    AS '$libdir/pg_inft', 'inft_content_hash';

COMMENT ON FUNCTION inft.inft_content_hash(text, text, text, text) IS
    'Computes keccak256(len4(job_id)||job_id||len4(question)||question||'
    'len4(thinking)||thinking||len4(answer)||answer). '
    'len4() is a 4-byte big-endian uint32.';

-- ── Ethereum personal-sign hash ────────────────────────────────────────────

CREATE FUNCTION inft.inft_eth_personal_hash(content_hash bytea)
    RETURNS bytea
    LANGUAGE c STRICT IMMUTABLE PARALLEL SAFE
    AS '$libdir/pg_inft', 'inft_eth_personal_hash';

COMMENT ON FUNCTION inft.inft_eth_personal_hash(bytea) IS
    'Returns keccak256("\x19Ethereum Signed Message:\n32" || content_hash). '
    'Prefix is exactly 28 bytes: 0x19 + "Ethereum Signed Message:\n" (26) + "32" (2).';

-- ── ECDSA signature verifier ─────────────────────────────────────────────
-- Uses plpython3u + eth_account when available; falls back to returning true
-- when plpython3u is not installed (controlled by pg_inft.require_proof_verification GUC).

DO $$
DECLARE
    has_plpython bool;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_language WHERE lanname = 'plpython3u'
    ) INTO has_plpython;

    IF has_plpython THEN
        EXECUTE $body$
            CREATE FUNCTION inft.inft_eth_verify(
                content_hash  bytea,
                proof_sig     bytea,
                miner_addr    text
            )
                RETURNS boolean
                LANGUAGE plpython3u
                VOLATILE CALLED ON NULL INPUT
                SECURITY DEFINER
            AS $py$
            try:
                from eth_account import Account
                from eth_account.messages import encode_defunct

                if content_hash is None or proof_sig is None or miner_addr is None:
                    return False

                ch_bytes  = bytes(content_hash)
                sig_bytes = bytes(proof_sig)

                msg       = encode_defunct(primitive=ch_bytes)
                recovered = Account.recover_message(msg, signature=sig_bytes)
                return recovered.lower() == str(miner_addr).lower()

            except ImportError:
                plpy.warning("eth_account not available; ECDSA verification skipped")
                return True
            except Exception as e:
                plpy.error("ECDSA error: " + str(e))
                return False
            $py$
        $body$;
    ELSE
        EXECUTE $body$
            CREATE FUNCTION inft.inft_eth_verify(
                content_hash  bytea,
                proof_sig     bytea,
                miner_addr    text
            )
                RETURNS boolean
                LANGUAGE plpgsql
                VOLATILE CALLED ON NULL INPUT
                SECURITY DEFINER
            AS $pl$
            BEGIN
                -- plpython3u not installed; proof verification delegated to application layer
                RAISE NOTICE 'inft_eth_verify: plpython3u not available, returning true';
                RETURN true;
            END;
            $pl$
        $body$;
    END IF;
END;
$$;

COMMENT ON FUNCTION inft.inft_eth_verify(bytea, bytea, text) IS
    'Verifies an Ethereum personal-sign ECDSA signature. '
    'Returns true if the recovered address matches miner_addr (case-insensitive). '
    'Degrades to returning true when plpython3u or eth_account is not available.';

-- ── Search result composite type ──────────────────────────────────────────

CREATE TYPE inft.inft_search_result AS (
    id            bigint,
    job_id        text,
    miner_address text,
    model_id      text,
    question_text text,
    thinking_text text,
    answer_text   text,
    score         float8
);

-- ── Search function ────────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_search(
    question  text,
    model_id  text  DEFAULT '',
    lim       int   DEFAULT 5
)
    RETURNS SETOF inft.inft_search_result
    LANGUAGE c STRICT
    AS '$libdir/pg_inft', 'inft_search';

COMMENT ON FUNCTION inft.inft_search(text, text, int) IS
    'Staged BM25 + trigram retrieval pipeline. '
    'Returns up to lim rows scored as 0.5*bm25_question + 0.2*bm25_chunk + 0.3*trgm. '
    'Falls back to BM25-only if pg_trgm is not installed. '
    'Minimum score threshold controlled by GUC inft.min_similarity.';

-- ── Chunker function ───────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_chunk_text(
    input       text,
    target_size int  DEFAULT 300,
    overlap     int  DEFAULT 50
)
    RETURNS SETOF text
    LANGUAGE c
    AS '$libdir/pg_inft', 'inft_chunk_text';

COMMENT ON FUNCTION inft.inft_chunk_text(text, int, int) IS
    'Splits input into overlapping paragraph+sentence chunks. '
    'target_size: target chars per chunk (default 300). '
    'overlap: chars of context carried from previous chunk (default 50). '
    'Skips chunks < 30 chars after trimming.';

-- ── Tables ─────────────────────────────────────────────────────────────────

CREATE TABLE inft.inft_thought_log (
    id              bigint      PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    job_id          text        NOT NULL,
    miner_address   text        NOT NULL,
    model_id        text        NOT NULL,
    question_text   text        NOT NULL,
    thinking_text   text,
    answer_text     text,
    question_tsv    tsvector    GENERATED ALWAYS AS
                        (to_tsvector('english', question_text)) STORED,
    thinking_tsv    tsvector    GENERATED ALWAYS AS
                        (to_tsvector('english', COALESCE(thinking_text, ''))) STORED,
    content_hash    bytea       NOT NULL,
    proof_sig       bytea       NOT NULL,
    block_number    bigint,
    tx_hash         text,
    chain_verified  boolean     NOT NULL DEFAULT false,
    peer_origin     text,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id),
    UNIQUE (content_hash)
);

COMMENT ON TABLE inft.inft_thought_log IS
    'Stores completed LLM inference records with cryptographic proof of authorship.';

CREATE TABLE inft.inft_thought_chunks (
    id          bigint  PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    thought_id  bigint  NOT NULL
                    REFERENCES inft.inft_thought_log(id) ON DELETE CASCADE,
    chunk_index int     NOT NULL,
    chunk_text  text    NOT NULL,
    chunk_tsv   tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', chunk_text)) STORED,
    chunk_hash  bytea   NOT NULL,
    UNIQUE (chunk_hash)
);

COMMENT ON TABLE inft.inft_thought_chunks IS
    'Overlap-chunked segments of thinking_text, indexed for BM25 retrieval.';

CREATE TABLE inft.inft_peer_sync (
    peer_address      text        PRIMARY KEY,
    last_seen         timestamptz DEFAULT now(),
    thoughts_received bigint      DEFAULT 0,
    proofs_rejected   bigint      DEFAULT 0,
    last_job_id       text
);

COMMENT ON TABLE inft.inft_peer_sync IS
    'Tracks P2P peer sync state and per-peer proof rejection counts.';

-- ── Indexes ────────────────────────────────────────────────────────────────

-- BM25 full-text indexes
CREATE INDEX inft_thought_log_question_tsv_gin
    ON inft.inft_thought_log USING GIN (question_tsv);

CREATE INDEX inft_thought_log_thinking_tsv_gin
    ON inft.inft_thought_log USING GIN (thinking_tsv);

CREATE INDEX inft_thought_chunks_chunk_tsv_gin
    ON inft.inft_thought_chunks USING GIN (chunk_tsv);

-- Trigram index on question_text (conditional: only created if pg_trgm is available)
DO $$
BEGIN
    CREATE INDEX inft_thought_log_question_trgm_gin
        ON inft.inft_thought_log
        USING GIN (question_text gin_trgm_ops);
EXCEPTION
    WHEN undefined_object THEN
        RAISE NOTICE 'pg_trgm not installed; skipping trigram index on question_text';
END;
$$;

-- B-tree indexes for filtering
CREATE INDEX inft_thought_log_model_id_idx
    ON inft.inft_thought_log (model_id);

CREATE INDEX inft_thought_log_miner_address_idx
    ON inft.inft_thought_log (miner_address);

CREATE INDEX inft_thought_log_block_number_idx
    ON inft.inft_thought_log (block_number)
    WHERE block_number IS NOT NULL;

-- BRIN index for time-range scans
CREATE INDEX inft_thought_log_ingested_at_brin
    ON inft.inft_thought_log USING BRIN (ingested_at);

-- ── Proof trigger ──────────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_proof_trigger_fn()
    RETURNS trigger
    LANGUAGE c
    AS '$libdir/pg_inft', 'inft_proof_trigger_fn';

COMMENT ON FUNCTION inft.inft_proof_trigger_fn() IS
    'BEFORE INSERT trigger on inft_thought_log. '
    'Validates content_hash and ECDSA proof signature. '
    'Controlled by GUC inft.require_proof_verification.';

CREATE TRIGGER inft_proof_trigger
    BEFORE INSERT ON inft.inft_thought_log
    FOR EACH ROW
    EXECUTE FUNCTION inft.inft_proof_trigger_fn();

-- ── Ingest function ────────────────────────────────────────────────────────

CREATE FUNCTION inft.inft_ingest(
    p_job_id         text,
    p_miner_address  text,
    p_model_id       text,
    p_question_text  text,
    p_thinking_text  text,
    p_answer_text    text,
    p_proof_sig      bytea,
    p_block_number   bigint  DEFAULT NULL,
    p_tx_hash        text    DEFAULT NULL,
    p_peer_origin    text    DEFAULT NULL
)
    RETURNS bigint
    LANGUAGE plpgsql
    VOLATILE
AS $$
DECLARE
    v_content_hash bytea;
    v_thought_id   bigint;
    v_chunk_text   text;
    v_chunk_idx    int := 0;
    v_chunk_hash   bytea;
BEGIN
    -- Compute content_hash from the four fields
    v_content_hash := inft.inft_content_hash(
        p_job_id,
        p_question_text,
        COALESCE(p_thinking_text, ''),
        COALESCE(p_answer_text, '')
    );

    -- Check for existing job_id (idempotent ingest)
    SELECT id INTO v_thought_id
    FROM inft.inft_thought_log
    WHERE job_id = p_job_id;

    IF FOUND THEN
        RETURN v_thought_id;
    END IF;

    -- Insert into thought log (proof trigger fires here)
    INSERT INTO inft.inft_thought_log (
        job_id,
        miner_address,
        model_id,
        question_text,
        thinking_text,
        answer_text,
        content_hash,
        proof_sig,
        block_number,
        tx_hash,
        chain_verified,
        peer_origin
    ) VALUES (
        p_job_id,
        p_miner_address,
        p_model_id,
        p_question_text,
        p_thinking_text,
        p_answer_text,
        v_content_hash,
        p_proof_sig,
        p_block_number,
        p_tx_hash,
        false,
        p_peer_origin
    )
    RETURNING id INTO v_thought_id;

    -- Chunk the thinking_text (skip if null or empty)
    IF p_thinking_text IS NOT NULL AND length(p_thinking_text) > 0 THEN
        FOR v_chunk_text IN
            SELECT chunk
            FROM inft.inft_chunk_text(p_thinking_text, 300, 50) AS chunk
        LOOP
            v_chunk_hash := inft.inft_keccak256(convert_to(v_chunk_text, 'UTF8'));

            INSERT INTO inft.inft_thought_chunks (
                thought_id,
                chunk_index,
                chunk_text,
                chunk_hash
            ) VALUES (
                v_thought_id,
                v_chunk_idx,
                v_chunk_text,
                v_chunk_hash
            )
            ON CONFLICT (chunk_hash) DO NOTHING;

            v_chunk_idx := v_chunk_idx + 1;
        END LOOP;
    END IF;

    RETURN v_thought_id;
END;
$$;

COMMENT ON FUNCTION inft.inft_ingest(text,text,text,text,text,text,bytea,bigint,text,text) IS
    'Ingest a completed inference record with proof. '
    'Idempotent on job_id. Returns the thought log id.';

-- ── Views ──────────────────────────────────────────────────────────────────

CREATE VIEW inft.inft_network_summary AS
SELECT
    count(*)                                     AS total_thoughts,
    count(DISTINCT miner_address)                AS unique_miners,
    count(DISTINCT model_id)                     AS unique_models,
    count(*) FILTER (WHERE chain_verified)       AS chain_verified_count,
    min(ingested_at)                             AS oldest_thought,
    max(ingested_at)                             AS newest_thought,
    (SELECT count(*) FROM inft.inft_thought_chunks)  AS total_chunks
FROM inft.inft_thought_log;

COMMENT ON VIEW inft.inft_network_summary IS
    'Aggregate statistics across all ingested thoughts.';

CREATE VIEW inft.inft_peer_stats AS
SELECT
    ps.peer_address,
    ps.last_seen,
    ps.thoughts_received,
    ps.proofs_rejected,
    ps.last_job_id,
    CASE
        WHEN ps.thoughts_received = 0 THEN 0.0
        ELSE round(
            100.0 * ps.proofs_rejected::numeric / ps.thoughts_received::numeric,
            2
        )
    END AS rejection_rate_pct
FROM inft.inft_peer_sync ps
ORDER BY ps.thoughts_received DESC;

COMMENT ON VIEW inft.inft_peer_stats IS
    'Per-peer thought receive/reject statistics with rejection rate.';
