/*
 * pg_nest.h
 *
 * Distributed nested JSONB + time-series extension.
 *
 * Architecture
 * ------------
 * Each "nest table" is an ordinary PostgreSQL table with a JSONB column.
 * When nest_create() is called we:
 *
 *   1. Record the table in pg_nest.nest_registry.
 *
 *   2. Create a companion path-store table:
 *        _nest_<table>_paths(
 *            doc_id bigint, path text, path_depth int2,
 *            val_text text, val_num float8,
 *            val_bool boolean, val_null boolean, ts timestamptz)
 *
 *   3. Install a row-level trigger that decomposes every inserted/updated
 *      JSONB document into (path, value) rows using a single batch INSERT.
 *
 *   4. Create a set of indexes that make every access O(log N):
 *        - (path, val_text) INCLUDE (doc_id, ts)   ← covering, index-only scans
 *        - (path, val_num)  INCLUDE (doc_id, ts)   ← covering
 *        - (doc_id)                                 ← reverse lookup
 *        - BRIN (ts)                                ← append-friendly, tiny
 *        - GIN  (path gin_trgm_ops)                 ← path prefix/pattern search
 *          (only if pg_trgm is installed)
 *
 * Key-escaping
 * ------------
 * Object keys that contain '.', '[', ']', '"', or '\' are wrapped in
 * double-quotes in the path string: a."key.with.dot".b
 * This is the same convention used by PostgreSQL's JSONPath language.
 *
 * Query complexity
 * ----------------
 * Because the path-store is a flat table with B-tree indexes, ANY depth of
 * JSONB nesting is resolved in O(log N) — a single index scan regardless of
 * how many levels deep the queried key lives.
 *
 * Citus integration
 * -----------------
 * Call nest_distribute(table, shard_key_col) to co-locate the path-store
 * table with the source table on the same Citus worker.
 *
 * TimescaleDB integration
 * -----------------------
 * Call nest_hypertable(table, chunk_interval) to convert the companion
 * path-store into a TimescaleDB hypertable, enabling chunk pruning on
 * time-bounded queries.
 */

#pragma once

#include "postgres.h"
#include "fmgr.h"
#include "utils/jsonb.h"
#include "utils/palloc.h"
#include "lib/stringinfo.h"

/* -------------------------------------------------------------------------
 * Internal path-decomposition API (nest_decompose.c)
 * ------------------------------------------------------------------------- */

typedef struct NestPathEntry
{
    char   *path;       /* dot-notation path, e.g. "a.b.c" or "items[2].price"   */
    char   *val_text;   /* text representation (always set for leaf nodes)         */
    double  val_num;    /* numeric value; valid when has_num == true               */
    bool    val_bool;   /* boolean value; valid when jbvtype == jbvBool            */
    bool    val_null;   /* true when the JSON value is null                        */
    bool    has_num;    /* true when val_num is meaningful                         */
    int     jbvtype;    /* JsonbValue.type for the leaf                            */
    int     depth;      /* nesting depth (0 = top-level key)                      */
} NestPathEntry;

/*
 * Decompose a Jsonb value into a palloc'd array of NestPathEntry.
 * *nentries is set to the number of entries returned.
 * Only leaf nodes are emitted (strings, numbers, booleans, nulls).
 */
extern NestPathEntry *nest_decompose_jsonb(Jsonb *jb, int *nentries);

/* -------------------------------------------------------------------------
 * Path-store table name helpers
 * ------------------------------------------------------------------------- */

extern char *nest_paths_table_name(const char *schema, const char *relname);

/* -------------------------------------------------------------------------
 * SQL-callable function prototypes
 * ------------------------------------------------------------------------- */

/* nest_decompose(jsonb) → SETOF decompose_row */
extern Datum pg_nest_decompose(PG_FUNCTION_ARGS);

/* nest_create(source_table regclass, jsonb_column name,
 *             id_column name DEFAULT 'id',
 *             time_column name DEFAULT NULL) → void */
extern Datum pg_nest_create(PG_FUNCTION_ARGS);

/* nest_drop(source_table regclass) → void */
extern Datum pg_nest_drop(PG_FUNCTION_ARGS);

/* nest_reindex(source_table regclass) → bigint (rows indexed) */
extern Datum pg_nest_reindex(PG_FUNCTION_ARGS);

/* Internal trigger: _nest_trigger_fn() → trigger */
extern Datum pg_nest_trigger(PG_FUNCTION_ARGS);

/* nest_query(source_table regclass, path text, value text)
 * → SETOF query_row */
extern Datum pg_nest_query(PG_FUNCTION_ARGS);

/* nest_query_num(source_table regclass, path text, lo float8, hi float8)
 * → SETOF query_num_row */
extern Datum pg_nest_query_num(PG_FUNCTION_ARGS);

/* nest_query_time(source_table regclass, path text, value text,
 *                time_start timestamptz, time_end timestamptz)
 * → SETOF query_time_row */
extern Datum pg_nest_query_time(PG_FUNCTION_ARGS);

/* nest_query_jsonpath(source_table regclass, jspath jsonpath)
 * → SETOF bigint (matching doc_ids, using the path-store as pre-filter) */
extern Datum pg_nest_query_jsonpath(PG_FUNCTION_ARGS);

/* nest_where(source_table regclass, conditions jsonb)
 * → SETOF bigint
 * Returns doc_ids matching ALL key→value conditions in the jsonb object.
 * Uses a single GROUP-BY / HAVING query against the path store for O(log N)
 * multi-predicate intersection. */
extern Datum pg_nest_where(PG_FUNCTION_ARGS);

/* nest_path_list(source_table regclass)
 * → SETOF (path text, freq bigint, distinct_vals bigint,
 *           min_num float8, max_num float8) */
extern Datum pg_nest_path_list(PG_FUNCTION_ARGS);

/* nest_path_exists(source_table regclass, path text) → boolean */
extern Datum pg_nest_path_exists(PG_FUNCTION_ARGS);

/* nest_time_bucket_agg(source_table regclass, path text,
 *                      bucket interval,
 *                      time_start timestamptz, time_end timestamptz)
 * → SETOF (bucket timestamptz, count bigint, vals text[]) */
extern Datum pg_nest_time_bucket_agg(PG_FUNCTION_ARGS);

/* nest_distribute(source_table regclass, shard_column name) → void */
extern Datum pg_nest_distribute(PG_FUNCTION_ARGS);

/* nest_hypertable(source_table regclass, chunk_interval interval) → void */
extern Datum pg_nest_hypertable(PG_FUNCTION_ARGS);

/* nest_stats(source_table regclass) → stats_row */
extern Datum pg_nest_stats(PG_FUNCTION_ARGS);

/* pg_nest_version() → text */
extern Datum pg_nest_version(PG_FUNCTION_ARGS);

/* -------------------------------------------------------------------------
 * GUCs
 * ------------------------------------------------------------------------- */

extern bool   pg_nest_enable_citus;
extern bool   pg_nest_enable_timescaledb;
extern bool   pg_nest_enable_trgm;       /* try to create GIN trigram index */
extern int    pg_nest_max_depth;          /* max nesting depth (default 32)   */
extern int    pg_nest_array_max_elems;    /* max array elems to expand (64)   */
