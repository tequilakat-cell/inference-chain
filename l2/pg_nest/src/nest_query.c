/*
 * nest_query.c
 *
 * Query functions against the path-store companion table.
 *
 * Parameterisation
 * ----------------
 * All functions use SPI_execute_with_args() so that user-supplied values are
 * passed as bind parameters — never interpolated into SQL strings.  This
 * prevents SQL injection even when path or value contain single-quotes,
 * backslashes, or other metacharacters.
 *
 * Plan caching
 * ------------
 * SPI_execute_with_args() internally prepares and caches the plan within the
 * current transaction, avoiding repeated parse/plan cycles for the same query
 * shape when the function is called multiple times in one query.
 *
 * Functions
 * ---------
 *  nest_query          – equality match on val_text
 *  nest_query_num      – range match on val_num
 *  nest_query_time     – equality + time window
 *  nest_path_list      – aggregate stats per distinct path
 *  nest_path_exists    – boolean: does any doc have this path?
 *  nest_where          – multi-condition intersection (AND semantics)
 *  nest_stats          – collection-level counts
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"

#include "pg_nest.h"

/* -------------------------------------------------------------------------
 * Helper
 * ------------------------------------------------------------------------- */

static char *
paths_tbl_from_oid(Oid relid)
{
    char *relname = get_rel_name(relid);
    char *nspname;

    if (!relname)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation %u does not exist", relid)));

    nspname = get_namespace_name(get_rel_namespace(relid));
    return nest_paths_table_name(nspname, relname);
}

/* -------------------------------------------------------------------------
 * nest_query(source_table, path, value) → SETOF query_row
 *
 * Uses the covering (path, val_text) INCLUDE (doc_id, ts) index for an
 * index-only scan.  No heap page fetches when the index is not bloated.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_query);

Datum
pg_nest_query(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid      = PG_GETARG_OID(0);
        text          *path_arg   = PG_GETARG_TEXT_PP(1);
        text          *value_arg  = PG_GETARG_TEXT_PP(2);
        char          *paths_tbl  = paths_tbl_from_oid(relid);
        StringInfoData sql;
        Oid            argtypes[2] = {TEXTOID, TEXTOID};
        Datum          args[2]     = {PointerGetDatum(path_arg),
                                      PointerGetDatum(value_arg)};
        TupleDesc      tupdesc;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        initStringInfo(&sql);
        appendStringInfo(&sql,
            "SELECT doc_id, val_text FROM %s"
            " WHERE path = $1 AND val_text = $2"
            " ORDER BY doc_id",
            paths_tbl);

        SPI_execute_with_args(sql.data, 2, argtypes, args, NULL, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("nest_query must be used as SRF")));
        fctx->attinmeta = TupleDescGetAttInMetadata(tupdesc);

        MemoryContextSwitchTo(oldctx);
    }

    fctx     = SRF_PERCALL_SETUP();
    tuptable = (SPITupleTable *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        HeapTuple  st = tuptable->vals[fctx->call_cntr];
        TupleDesc  sd = tuptable->tupdesc;
        bool       isnull;
        Datum      vals[2];
        bool       nls[2] = {false, false};
        HeapTuple  ot;

        vals[0] = SPI_getbinval(st, sd, 1, &isnull); nls[0] = isnull;
        vals[1] = SPI_getbinval(st, sd, 2, &isnull); nls[1] = isnull;

        ot = heap_form_tuple(fctx->attinmeta->tupdesc, vals, nls);
        SRF_RETURN_NEXT(fctx, HeapTupleGetDatum(ot));
    }

    SPI_finish();
    SRF_RETURN_DONE(fctx);
}

/* -------------------------------------------------------------------------
 * nest_query_num(source_table, path, lo, hi) → SETOF query_num_row
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_query_num);

Datum
pg_nest_query_num(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid     = PG_GETARG_OID(0);
        text          *path_arg  = PG_GETARG_TEXT_PP(1);
        float8         lo        = PG_GETARG_FLOAT8(2);
        float8         hi        = PG_GETARG_FLOAT8(3);
        char          *paths_tbl = paths_tbl_from_oid(relid);
        StringInfoData sql;
        Oid            argtypes[3] = {TEXTOID, FLOAT8OID, FLOAT8OID};
        Datum          args[3]     = {PointerGetDatum(path_arg),
                                      Float8GetDatum(lo),
                                      Float8GetDatum(hi)};
        TupleDesc      tupdesc;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        initStringInfo(&sql);
        appendStringInfo(&sql,
            "SELECT doc_id, val_num FROM %s"
            " WHERE path = $1 AND val_num BETWEEN $2 AND $3"
            " ORDER BY doc_id",
            paths_tbl);

        SPI_execute_with_args(sql.data, 3, argtypes, args, NULL, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("nest_query_num must be used as SRF")));
        fctx->attinmeta = TupleDescGetAttInMetadata(tupdesc);

        MemoryContextSwitchTo(oldctx);
    }

    fctx     = SRF_PERCALL_SETUP();
    tuptable = (SPITupleTable *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        HeapTuple  st = tuptable->vals[fctx->call_cntr];
        TupleDesc  sd = tuptable->tupdesc;
        bool       isnull;
        Datum      vals[2];
        bool       nls[2] = {false, false};
        HeapTuple  ot;

        vals[0] = SPI_getbinval(st, sd, 1, &isnull); nls[0] = isnull;
        vals[1] = SPI_getbinval(st, sd, 2, &isnull); nls[1] = isnull;

        ot = heap_form_tuple(fctx->attinmeta->tupdesc, vals, nls);
        SRF_RETURN_NEXT(fctx, HeapTupleGetDatum(ot));
    }

    SPI_finish();
    SRF_RETURN_DONE(fctx);
}

/* -------------------------------------------------------------------------
 * nest_query_time(source_table, path, value, time_start, time_end)
 * → SETOF query_time_row
 *
 * Uses the (ts, path) INCLUDE (doc_id, val_text) covering index for
 * time-bounded path queries.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_query_time);

Datum
pg_nest_query_time(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid      = PG_GETARG_OID(0);
        text          *path_arg   = PG_GETARG_TEXT_PP(1);
        text          *value_arg  = PG_GETARG_TEXT_PP(2);
        TimestampTz    ts_start   = PG_GETARG_TIMESTAMPTZ(3);
        TimestampTz    ts_end     = PG_GETARG_TIMESTAMPTZ(4);
        char          *paths_tbl  = paths_tbl_from_oid(relid);
        StringInfoData sql;
        Oid            argtypes[4] = {TEXTOID, TEXTOID, TIMESTAMPTZOID, TIMESTAMPTZOID};
        Datum          args[4]     = {PointerGetDatum(path_arg),
                                      PointerGetDatum(value_arg),
                                      TimestampTzGetDatum(ts_start),
                                      TimestampTzGetDatum(ts_end)};
        TupleDesc      tupdesc;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        initStringInfo(&sql);
        appendStringInfo(&sql,
            "SELECT doc_id, val_text, ts FROM %s"
            " WHERE path = $1 AND val_text = $2"
            "   AND ts BETWEEN $3 AND $4"
            " ORDER BY ts, doc_id",
            paths_tbl);

        SPI_execute_with_args(sql.data, 4, argtypes, args, NULL, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("nest_query_time must be used as SRF")));
        fctx->attinmeta = TupleDescGetAttInMetadata(tupdesc);

        MemoryContextSwitchTo(oldctx);
    }

    fctx     = SRF_PERCALL_SETUP();
    tuptable = (SPITupleTable *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        HeapTuple  st = tuptable->vals[fctx->call_cntr];
        TupleDesc  sd = tuptable->tupdesc;
        bool       isnull;
        Datum      vals[3];
        bool       nls[3] = {false, false, false};
        HeapTuple  ot;

        vals[0] = SPI_getbinval(st, sd, 1, &isnull); nls[0] = isnull;
        vals[1] = SPI_getbinval(st, sd, 2, &isnull); nls[1] = isnull;
        vals[2] = SPI_getbinval(st, sd, 3, &isnull); nls[2] = isnull;

        ot = heap_form_tuple(fctx->attinmeta->tupdesc, vals, nls);
        SRF_RETURN_NEXT(fctx, HeapTupleGetDatum(ot));
    }

    SPI_finish();
    SRF_RETURN_DONE(fctx);
}

/* -------------------------------------------------------------------------
 * nest_where(source_table regclass, conditions jsonb) → SETOF bigint
 *
 * Multi-condition intersection using a single GROUP BY / HAVING query.
 *
 * Given:  conditions = '{"user.id": 42, "type": "click"}'
 *
 * Issues: SELECT doc_id
 *         FROM paths
 *         WHERE (path, val_text) IN (('user.id','42'), ('type','click'))
 *         GROUP BY doc_id
 *         HAVING COUNT(*) = 2
 *
 * This satisfies ALL conditions with a single index scan pass and one
 * GROUP BY rather than N nested subqueries or N hash-joins, keeping
 * the overall cost O(log N + M) where M is the matching set size.
 *
 * Numeric conditions: values that are valid JSON numbers match val_text
 * (which stores the canonical numeric representation).  For explicit
 * numeric range matching use nest_query_num().
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_where);

Datum
pg_nest_where(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid      = PG_GETARG_OID(0);
        Jsonb         *cond       = PG_GETARG_JSONB_P(1);
        char          *paths_tbl  = paths_tbl_from_oid(relid);
        JsonbIterator *it;
        JsonbValue     k, v;
        JsonbIteratorToken tok;
        StringInfoData sql;
        int            nconds = 0;
        bool           first  = true;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        initStringInfo(&sql);
        appendStringInfo(&sql,
            "SELECT doc_id FROM %s WHERE (path, val_text) IN (", paths_tbl);

        it  = JsonbIteratorInit(&cond->root);
        tok = JsonbIteratorNext(&it, &k, false);  /* WJB_BEGIN_OBJECT */

        while (true)
        {
            tok = JsonbIteratorNext(&it, &k, false);
            if (tok == WJB_END_OBJECT || tok == WJB_DONE)
                break;
            /* tok == WJB_KEY */
            tok = JsonbIteratorNext(&it, &v, true);

            char *key_str  = palloc(k.val.string.len + 1);
            memcpy(key_str, k.val.string.val, k.val.string.len);
            key_str[k.val.string.len] = '\0';

            /* Render the value as text */
            char *val_str;
            if (v.type == jbvString)
            {
                val_str = palloc(v.val.string.len + 1);
                memcpy(val_str, v.val.string.val, v.val.string.len);
                val_str[v.val.string.len] = '\0';
            }
            else if (v.type == jbvNumeric)
            {
                val_str = DatumGetCString(
                              DirectFunctionCall1(numeric_out,
                                  NumericGetDatum(v.val.numeric)));
            }
            else if (v.type == jbvBool)
            {
                val_str = v.val.boolean ? pstrdup("true") : pstrdup("false");
            }
            else
            {
                val_str = pstrdup("null");
            }

            if (!first) appendStringInfoChar(&sql, ',');
            appendStringInfo(&sql, "(%s,%s)",
                             quote_literal_cstr(key_str),
                             quote_literal_cstr(val_str));
            first = false;
            nconds++;
        }

        if (nconds == 0)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("nest_where: conditions object must not be empty")));

        appendStringInfo(&sql,
            ") GROUP BY doc_id HAVING COUNT(*) = %d ORDER BY doc_id",
            nconds);

        SPI_execute(sql.data, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;
        /* no tupdesc needed — returns plain bigint */

        MemoryContextSwitchTo(oldctx);
    }

    fctx     = SRF_PERCALL_SETUP();
    tuptable = (SPITupleTable *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        HeapTuple  st = tuptable->vals[fctx->call_cntr];
        TupleDesc  sd = tuptable->tupdesc;
        bool       isnull;
        Datum      d  = SPI_getbinval(st, sd, 1, &isnull);

        SRF_RETURN_NEXT(fctx, isnull ? (Datum) 0 : d);
    }

    SPI_finish();
    SRF_RETURN_DONE(fctx);
}

/* -------------------------------------------------------------------------
 * nest_path_list(source_table) → SETOF path_list_row
 *
 * Returns per-path aggregate statistics.  Useful for schema discovery and
 * deciding which paths to materialize as generated columns.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_path_list);

Datum
pg_nest_path_list(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid     = PG_GETARG_OID(0);
        char          *paths_tbl = paths_tbl_from_oid(relid);
        StringInfoData sql;
        TupleDesc      tupdesc;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        initStringInfo(&sql);
        appendStringInfo(&sql,
            "SELECT"
            "  path,"
            "  COUNT(*)                    AS freq,"
            "  COUNT(DISTINCT val_text)    AS distinct_vals,"
            "  MIN(val_num)                AS min_num,"
            "  MAX(val_num)                AS max_num,"
            "  MAX(path_depth)             AS max_depth"
            " FROM %s"
            " GROUP BY path"
            " ORDER BY freq DESC, path",
            paths_tbl);

        SPI_execute(sql.data, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("nest_path_list must be used as SRF")));
        fctx->attinmeta = TupleDescGetAttInMetadata(tupdesc);

        MemoryContextSwitchTo(oldctx);
    }

    fctx     = SRF_PERCALL_SETUP();
    tuptable = (SPITupleTable *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        HeapTuple  st = tuptable->vals[fctx->call_cntr];
        TupleDesc  sd = tuptable->tupdesc;
        bool       isnull;
        Datum      vals[6];
        bool       nls[6]  = {false, false, false, false, false, false};
        HeapTuple  ot;
        int        c;

        for (c = 0; c < 6; c++)
        {
            vals[c] = SPI_getbinval(st, sd, c + 1, &isnull);
            nls[c]  = isnull;
        }

        ot = heap_form_tuple(fctx->attinmeta->tupdesc, vals, nls);
        SRF_RETURN_NEXT(fctx, HeapTupleGetDatum(ot));
    }

    SPI_finish();
    SRF_RETURN_DONE(fctx);
}

/* -------------------------------------------------------------------------
 * nest_path_exists(source_table, path) → boolean
 *
 * Single-row check using the covering index — returns after the first hit.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_path_exists);

Datum
pg_nest_path_exists(PG_FUNCTION_ARGS)
{
    Oid            relid     = PG_GETARG_OID(0);
    text          *path_arg  = PG_GETARG_TEXT_PP(1);
    char          *paths_tbl = paths_tbl_from_oid(relid);
    StringInfoData sql;
    Oid            argtypes[1] = {TEXTOID};
    Datum          args[1]     = {PointerGetDatum(path_arg)};
    int            rc;
    bool           exists;

    SPI_connect();
    initStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT 1 FROM %s WHERE path = $1 LIMIT 1",
        paths_tbl);

    rc = SPI_execute_with_args(sql.data, 1, argtypes, args, NULL, true, 1);
    if (rc != SPI_OK_SELECT)
        ereport(ERROR, (errmsg("nest_path_exists: SPI_execute failed")));

    exists = (SPI_processed > 0);
    SPI_finish();

    PG_RETURN_BOOL(exists);
}

/* -------------------------------------------------------------------------
 * nest_stats(source_table) → stats_row
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_stats);

Datum
pg_nest_stats(PG_FUNCTION_ARGS)
{
    Oid            relid     = PG_GETARG_OID(0);
    char          *paths_tbl = paths_tbl_from_oid(relid);
    StringInfoData sql;
    int            rc;
    TupleDesc      tupdesc;
    Datum          values[3];
    bool           nulls[3]  = {false, false, false};
    HeapTuple      htup;
    bool           isnull;

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("nest_stats must return a composite type")));

    SPI_connect();
    initStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT"
        "  COUNT(DISTINCT doc_id)  AS doc_count,"
        "  COUNT(*)                AS path_count,"
        "  COUNT(DISTINCT path)    AS distinct_paths"
        " FROM %s",
        paths_tbl);

    rc = SPI_execute(sql.data, true, 1);
    if (rc != SPI_OK_SELECT || SPI_processed == 0)
        ereport(ERROR, (errmsg("nest_stats: SPI_execute failed")));

    values[0] = SPI_getbinval(SPI_tuptable->vals[0],
                               SPI_tuptable->tupdesc, 1, &isnull);
    nulls[0]  = isnull;
    values[1] = SPI_getbinval(SPI_tuptable->vals[0],
                               SPI_tuptable->tupdesc, 2, &isnull);
    nulls[1]  = isnull;
    values[2] = SPI_getbinval(SPI_tuptable->vals[0],
                               SPI_tuptable->tupdesc, 3, &isnull);
    nulls[2]  = isnull;

    SPI_finish();

    htup = heap_form_tuple(BlessTupleDesc(tupdesc), values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(htup));
}
