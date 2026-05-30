-- pg_nest--1.0.sql
-- DDL for the pg_nest extension.
-- Load via: CREATE EXTENSION pg_nest;

\echo Use "CREATE EXTENSION pg_nest" to load this file. \quit

-- ============================================================
-- Schema
-- ============================================================

CREATE SCHEMA IF NOT EXISTS pg_nest;

-- ============================================================
-- Registry
-- Tracks every table registered with nest_create().
-- Used by nest_reindex(), the nest_tables view, and future
-- planner-hook rewriting.
-- ============================================================

CREATE TABLE pg_nest.nest_registry (
    relid       oid         NOT NULL PRIMARY KEY,
    nspname     name        NOT NULL,
    relname     name        NOT NULL,
    jsonb_col   name        NOT NULL,
    id_col      name        NOT NULL DEFAULT 'id',
    time_col    name,
    registered  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE pg_nest.nest_registry IS
    'Catalog of tables managed by pg_nest.';

-- ============================================================
-- Composite return types
-- ============================================================

CREATE TYPE pg_nest.decompose_row AS (
    path         text,
    val_text     text,
    val_num      float8,
    val_bool     boolean,
    val_null     boolean,
    depth        integer
);

CREATE TYPE pg_nest.query_row AS (
    doc_id       bigint,
    val_text     text
);

CREATE TYPE pg_nest.query_num_row AS (
    doc_id       bigint,
    val_num      float8
);

CREATE TYPE pg_nest.query_time_row AS (
    doc_id       bigint,
    val_text     text,
    ts           timestamptz
);

CREATE TYPE pg_nest.bucket_row AS (
    bucket       timestamptz,
    count        bigint,
    vals         text[]
);

CREATE TYPE pg_nest.stats_row AS (
    doc_count        bigint,
    path_count       bigint,
    distinct_paths   bigint
);

CREATE TYPE pg_nest.path_list_row AS (
    path             text,
    freq             bigint,
    distinct_vals    bigint,
    min_num          float8,
    max_num          float8,
    max_depth        integer
);

-- ============================================================
-- Version
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.pg_nest_version()
    RETURNS text
    LANGUAGE C STRICT PARALLEL SAFE IMMUTABLE
    AS '$libdir/pg_nest', 'pg_nest_version';

-- ============================================================
-- nest_decompose(jsonb) → SETOF decompose_row
--
-- Decomposes a JSONB document into flat (path, value, depth) rows.
-- Object keys containing '.', '[', ']', '"', '\' or whitespace are
-- wrapped in double-quotes in the path string.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_decompose(jb jsonb)
    RETURNS SETOF pg_nest.decompose_row
    LANGUAGE C STRICT PARALLEL SAFE IMMUTABLE
    COST 100 ROWS 64
    AS '$libdir/pg_nest', 'pg_nest_decompose';

COMMENT ON FUNCTION pg_nest.nest_decompose(jsonb) IS
    'Decompose a JSONB document into (path, value, depth) rows. '
    'One row per leaf node. Paths use dot-notation with double-quote '
    'escaping for keys that contain special characters.';

-- ============================================================
-- nest_create(source_table, jsonb_column,
--             id_column DEFAULT 'id',
--             time_column DEFAULT NULL) → void
--
-- Sets up pg_nest management for a table:
--   • Creates _nest_<table>_paths companion table
--   • Installs covering B-tree, BRIN, and (when pg_trgm is present)
--     GIN trigram indexes
--   • Installs AFTER INSERT OR UPDATE row trigger
--   • Records the table in pg_nest.nest_registry
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_create(
    source_table  regclass,
    jsonb_column  name,
    id_column     name         DEFAULT 'id',
    time_column   name         DEFAULT NULL
)
    RETURNS void
    LANGUAGE C
    AS '$libdir/pg_nest', 'pg_nest_create';

COMMENT ON FUNCTION pg_nest.nest_create(regclass, name, name, name) IS
    'Register a table with pg_nest: create path-store, indexes, trigger.';

-- ============================================================
-- nest_drop(source_table regclass) → void
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_drop(source_table regclass)
    RETURNS void
    LANGUAGE C STRICT
    AS '$libdir/pg_nest', 'pg_nest_drop';

COMMENT ON FUNCTION pg_nest.nest_drop(regclass) IS
    'Remove pg_nest management from a table (drops trigger and path-store).';

-- ============================================================
-- nest_reindex(source_table regclass) → bigint
--
-- Truncates the path-store and rebuilds it by scanning every row in
-- the source table.  Produces the same result as re-inserting all rows
-- but avoids per-row trigger overhead — use this for large backfills.
-- Returns the number of path rows inserted.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_reindex(source_table regclass)
    RETURNS bigint
    LANGUAGE C STRICT
    AS '$libdir/pg_nest', 'pg_nest_reindex';

COMMENT ON FUNCTION pg_nest.nest_reindex(regclass) IS
    'Rebuild the path-store from scratch (fast bulk backfill).';

-- ============================================================
-- Internal trigger function (not called directly)
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest._nest_trigger_fn()
    RETURNS trigger
    LANGUAGE C
    AS '$libdir/pg_nest', 'pg_nest_trigger';

-- ============================================================
-- nest_query(source_table, path, value) → SETOF query_row
--
-- Returns (doc_id, val_text) for documents where the given path equals
-- the given text value.  Uses the covering (path, val_text) INCLUDE
-- (doc_id, ts) index — index-only scan, no heap fetches.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_query(
    source_table  regclass,
    path          text,
    value         text
)
    RETURNS SETOF pg_nest.query_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 100 ROWS 100
    AS '$libdir/pg_nest', 'pg_nest_query';

COMMENT ON FUNCTION pg_nest.nest_query(regclass, text, text) IS
    'O(log N) exact match on a nested path. Any nesting depth.';

-- ============================================================
-- nest_query_num(source_table, path, lo, hi) → SETOF query_num_row
--
-- Returns (doc_id, val_num) for numeric values in [lo, hi].
-- Uses the covering (path, val_num) INCLUDE (doc_id, ts) index.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_query_num(
    source_table  regclass,
    path          text,
    lo            float8,
    hi            float8
)
    RETURNS SETOF pg_nest.query_num_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 100 ROWS 100
    AS '$libdir/pg_nest', 'pg_nest_query_num';

-- ============================================================
-- nest_query_time(source_table, path, value,
--                time_start, time_end) → SETOF query_time_row
--
-- Combines path equality with a time-window filter.
-- Uses the (ts, path) INCLUDE (doc_id, val_text) index.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_query_time(
    source_table  regclass,
    path          text,
    value         text,
    time_start    timestamptz,
    time_end      timestamptz
)
    RETURNS SETOF pg_nest.query_time_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 100 ROWS 100
    AS '$libdir/pg_nest', 'pg_nest_query_time';

-- ============================================================
-- nest_where(source_table, conditions jsonb) → SETOF bigint
--
-- Multi-condition AND intersection in a single GROUP BY / HAVING query.
-- All conditions must be satisfied by the same document.
--
-- Example:
--   SELECT * FROM events
--   WHERE id IN (
--     SELECT * FROM pg_nest.nest_where(
--       'events',
--       '{"user.country": "US", "type": "purchase", "meta.ab_group": "B"}'
--     )
--   );
--
-- Cost: O(log N + M) where M is the matching set size.
-- No N nested subqueries — single scan, single group-by.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_where(
    source_table  regclass,
    conditions    jsonb
)
    RETURNS SETOF bigint
    LANGUAGE C STRICT PARALLEL SAFE
    COST 200 ROWS 100
    AS '$libdir/pg_nest', 'pg_nest_where';

COMMENT ON FUNCTION pg_nest.nest_where(regclass, jsonb) IS
    'Multi-condition AND intersection. Accepts a flat JSONB object of '
    'path→value conditions and returns doc_ids matching all of them. '
    'Uses a single GROUP BY / HAVING query against the path-store index.';

-- ============================================================
-- nest_path_list(source_table) → SETOF path_list_row
--
-- Aggregate stats per distinct path.  Useful for schema discovery,
-- deciding which paths to materialize, and monitoring data distribution.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_path_list(source_table regclass)
    RETURNS SETOF pg_nest.path_list_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 200 ROWS 500
    AS '$libdir/pg_nest', 'pg_nest_path_list';

COMMENT ON FUNCTION pg_nest.nest_path_list(regclass) IS
    'Schema discovery: returns per-path frequency, cardinality, and numeric range.';

-- ============================================================
-- nest_path_exists(source_table, path) → boolean
--
-- Fast boolean check: does any document in the collection contain
-- this path?  Returns after the first index hit.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_path_exists(
    source_table  regclass,
    path          text
)
    RETURNS boolean
    LANGUAGE C STRICT PARALLEL SAFE
    COST 10
    AS '$libdir/pg_nest', 'pg_nest_path_exists';

-- ============================================================
-- nest_time_bucket_agg(source_table, path, bucket,
--                      time_start, time_end)
-- → SETOF bucket_row
--
-- Aggregates path values into time buckets.  Auto-selects time_bucket()
-- (TimescaleDB, chunk-aware) or date_bin() (PostgreSQL 14+).
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_time_bucket_agg(
    source_table  regclass,
    path          text,
    bucket        interval,
    time_start    timestamptz,
    time_end      timestamptz
)
    RETURNS SETOF pg_nest.bucket_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 200 ROWS 100
    AS '$libdir/pg_nest', 'pg_nest_time_bucket_agg';

-- ============================================================
-- nest_distribute(source_table, shard_column) → void
--
-- Co-locates the source table and its path-store on Citus workers
-- using shard_column.  All JOINs between source and path-store remain
-- node-local.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_distribute(
    source_table  regclass,
    shard_column  name
)
    RETURNS void
    LANGUAGE C STRICT
    AS '$libdir/pg_nest', 'pg_nest_distribute';

-- ============================================================
-- nest_hypertable(source_table, chunk_interval) → void
--
-- Converts the path-store companion table into a TimescaleDB hypertable
-- partitioned on the ts column, enabling chunk-level time pruning.
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_hypertable(
    source_table    regclass,
    chunk_interval  interval
)
    RETURNS void
    LANGUAGE C STRICT
    AS '$libdir/pg_nest', 'pg_nest_hypertable';

-- ============================================================
-- nest_stats(source_table) → stats_row
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_stats(source_table regclass)
    RETURNS pg_nest.stats_row
    LANGUAGE C STRICT PARALLEL SAFE
    COST 200
    AS '$libdir/pg_nest', 'pg_nest_stats';

-- ============================================================
-- nest_materialize_path(source_table, path, column_name) → void
--
-- Pure-SQL helper: adds a GENERATED ALWAYS AS (payload->>path) STORED
-- column to the source table and creates a B-tree index on it.
-- Use for paths that are queried in > 90% of workload — avoids the
-- path-store entirely for those paths, giving bare-metal PG planner
-- integration (seqscan estimates, bloom filters, etc.).
-- ============================================================

CREATE OR REPLACE FUNCTION pg_nest.nest_materialize_path(
    source_table   regclass,
    path           text,
    column_name    name
)
    RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
    tbl text := source_table::text;
    col text := quote_identifier(column_name);
    idx text := quote_identifier('_nest_mat_' || column_name || '_idx');
    -- Build the jsonb extraction expression for a dot-separated path.
    expr text;
    parts text[];
    i     int;
BEGIN
    parts := string_to_array(path, '.');
    expr  := quote_identifier(
                 (SELECT jsonb_col FROM pg_nest.nest_registry
                  WHERE relid = source_table::oid));

    FOR i IN 1 .. array_length(parts, 1) LOOP
        IF i < array_length(parts, 1) THEN
            expr := expr || ' -> ' || quote_literal(parts[i]);
        ELSE
            expr := expr || ' ->> ' || quote_literal(parts[i]);
        END IF;
    END LOOP;

    EXECUTE format(
        'ALTER TABLE %s ADD COLUMN IF NOT EXISTS %s text'
        ' GENERATED ALWAYS AS (%s) STORED',
        tbl, col, expr);

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %s ON %s (%s)',
        idx, tbl, col);
END;
$$;

COMMENT ON FUNCTION pg_nest.nest_materialize_path(regclass, text, name) IS
    'Materialize a frequently-queried nested path as a GENERATED ALWAYS AS '
    'column with its own B-tree index. Bypasses the path-store for that path, '
    'giving full planner visibility (histograms, selectivity estimates).';

-- ============================================================
-- Views
-- ============================================================

-- nest_tables: list all registered tables with health indicators
CREATE OR REPLACE VIEW pg_nest.nest_tables AS
SELECT
    r.nspname                                   AS schema,
    r.relname                                   AS source_table,
    r.jsonb_col,
    r.id_col,
    r.time_col,
    r.registered,
    c.relname                                   AS paths_table,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS paths_size,
    (SELECT COUNT(DISTINCT doc_id)
     FROM pg_catalog.pg_class pc
     WHERE pc.oid = c.oid)                      AS approx_docs
FROM
    pg_nest.nest_registry r
    LEFT JOIN pg_catalog.pg_class c
        ON c.relname = '_nest_' || r.relname || '_paths'
        AND c.relnamespace = (
            SELECT oid FROM pg_catalog.pg_namespace
            WHERE nspname = r.nspname)
ORDER BY r.nspname, r.relname;

COMMENT ON VIEW pg_nest.nest_tables IS
    'Registered pg_nest tables with path-store size information.';

-- nest_index_health: show index bloat indicators for path stores
CREATE OR REPLACE VIEW pg_nest.nest_index_health AS
SELECT
    n.nspname                                          AS schema,
    c.relname                                          AS paths_table,
    i.relname                                          AS index_name,
    am.amname                                          AS index_type,
    pg_size_pretty(pg_relation_size(i.oid))            AS index_size,
    ix.indisvalid                                      AS is_valid,
    ix.indisready                                      AS is_ready
FROM
    pg_catalog.pg_index ix
    JOIN pg_catalog.pg_class c  ON c.oid  = ix.indrelid
    JOIN pg_catalog.pg_class i  ON i.oid  = ix.indexrelid
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_catalog.pg_am am    ON am.oid = i.relam
WHERE
    c.relname LIKE '_nest\_%\_paths' ESCAPE '\'
ORDER BY
    n.nspname, c.relname, i.relname;

COMMENT ON VIEW pg_nest.nest_index_health IS
    'Index validity and size for all pg_nest path-store tables.';

-- ============================================================
-- Extended statistics
-- Improves the planner's cardinality estimates for multi-predicate
-- queries on the path-store. Registered lazily — run ANALYZE on
-- each path-store after nest_create() to populate.
-- These are template comments; actual CREATE STATISTICS must be
-- issued per-table after nest_create() because they reference
-- specific table names.  nest_create() does not yet auto-create
-- them, but you can do so manually:
--
--   CREATE STATISTICS _nest_events_path_text_stats (dependencies, ndistinct)
--     ON path, val_text FROM public._nest_events_paths;
--   ANALYZE public._nest_events_paths;
-- ============================================================
