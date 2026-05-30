/*
 * nest_timeseries.c
 *
 * Time-series and distribution helpers.
 *
 * nest_time_bucket_agg
 * --------------------
 * Aggregates path-store rows into time buckets.  Automatically detects
 * whether TimescaleDB's time_bucket() function is available (more efficient
 * for hypertables) and falls back to PostgreSQL 14+ date_bin() otherwise.
 * The BRIN index on ts makes range scans cheap even without TimescaleDB.
 *
 * nest_hypertable
 * ---------------
 * Converts the companion path-store table into a TimescaleDB hypertable,
 * enabling chunk-level pruning on time queries.  The trigger's batch insert
 * design (one SPI call per document) keeps write overhead low even with
 * chunk routing.  No-op when pg_nest.enable_timescaledb = off or when
 * TimescaleDB is not loaded.
 *
 * nest_distribute
 * ---------------
 * Distributes both the source table and its companion path-store using
 * Citus create_distributed_table, co-locating them on the shard column so
 * that joins between the source and path-store stay local.  No-op when
 * pg_nest.enable_citus = off or Citus is not loaded.
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"

#include "pg_nest.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static char *
paths_tbl_for(Oid relid)
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

/*
 * Return true when TimescaleDB's time_bucket() is available.
 * We probe pg_proc rather than looking for the extension, because some
 * deployments expose time_bucket() via a different schema.
 */
static bool
has_time_bucket(void)
{
    int rc = SPI_execute(
        "SELECT 1 FROM pg_proc p"
        " JOIN pg_namespace n ON n.oid = p.pronamespace"
        " WHERE p.proname = 'time_bucket'"
        "   AND n.nspname IN ('public','timescaledb_experimental','_timescaledb_internal')"
        " LIMIT 1",
        true, 1);
    return (rc == SPI_OK_SELECT && SPI_processed > 0);
}

/* -------------------------------------------------------------------------
 * nest_time_bucket_agg
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_time_bucket_agg);

Datum
pg_nest_time_bucket_agg(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    SPITupleTable   *tuptable;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Oid            relid      = PG_GETARG_OID(0);
        text          *path_arg   = PG_GETARG_TEXT_PP(1);
        Interval      *bucket_iv  = PG_GETARG_INTERVAL_P(2);
        TimestampTz    ts_start   = PG_GETARG_TIMESTAMPTZ(3);
        TimestampTz    ts_end     = PG_GETARG_TIMESTAMPTZ(4);
        char          *paths_tbl  = paths_tbl_for(relid);
        StringInfoData sql;
        Oid            argtypes[3] = {TEXTOID, TIMESTAMPTZOID, TIMESTAMPTZOID};
        Datum          args[3]     = {PointerGetDatum(path_arg),
                                      TimestampTzGetDatum(ts_start),
                                      TimestampTzGetDatum(ts_end)};
        char          *iv_literal;
        TupleDesc      tupdesc;
        bool           use_tsdb;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        SPI_connect();
        use_tsdb    = pg_nest_enable_timescaledb && has_time_bucket();
        iv_literal  = DatumGetCString(
                          DirectFunctionCall1(interval_out,
                              IntervalPGetDatum(bucket_iv)));

        initStringInfo(&sql);

        if (use_tsdb)
        {
            /*
             * time_bucket() is chunk-aware: TimescaleDB can prune chunks that
             * don't overlap the bucket boundary, making this faster on
             * hypertables than date_bin().
             */
            appendStringInfo(&sql,
                "SELECT"
                "  time_bucket(%s::interval, ts)  AS bucket,"
                "  COUNT(*)                        AS count,"
                "  array_agg(val_text ORDER BY ts) AS vals"
                " FROM %s"
                " WHERE path = $1 AND ts BETWEEN $2 AND $3"
                " GROUP BY 1"
                " ORDER BY 1",
                quote_literal_cstr(iv_literal),
                paths_tbl);
        }
        else
        {
            /*
             * date_bin() (PostgreSQL 14+) is the standard alternative.
             * Anchor at 2000-01-01 so buckets align with epoch.
             */
            appendStringInfo(&sql,
                "SELECT"
                "  date_bin(%s::interval, ts, TIMESTAMPTZ '2000-01-01') AS bucket,"
                "  COUNT(*)                        AS count,"
                "  array_agg(val_text ORDER BY ts) AS vals"
                " FROM %s"
                " WHERE path = $1 AND ts BETWEEN $2 AND $3"
                " GROUP BY 1"
                " ORDER BY 1",
                quote_literal_cstr(iv_literal),
                paths_tbl);
        }

        SPI_execute_with_args(sql.data, 3, argtypes, args, NULL, true, 0);

        fctx->user_fctx = SPI_tuptable;
        fctx->max_calls  = (uint64) SPI_processed;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR,
                    (errmsg("nest_time_bucket_agg must be used as SRF")));
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
        bool       nls[3]  = {false, false, false};
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
 * nest_hypertable(source_table regclass, chunk_interval interval) → void
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_hypertable);

Datum
pg_nest_hypertable(PG_FUNCTION_ARGS)
{
    Oid         relid      = PG_GETARG_OID(0);
    Interval   *chunk_iv   = PG_GETARG_INTERVAL_P(1);
    char       *paths_tbl  = paths_tbl_for(relid);
    char       *iv_s;
    StringInfoData sql;

    if (!pg_nest_enable_timescaledb)
        PG_RETURN_VOID();

    iv_s = DatumGetCString(
               DirectFunctionCall1(interval_out, IntervalPGetDatum(chunk_iv)));

    SPI_connect();
    initStringInfo(&sql);

    appendStringInfo(&sql,
        "SELECT create_hypertable(%s, 'ts',"
        "  chunk_time_interval => %s::interval,"
        "  if_not_exists       => true,"
        "  migrate_data        => true)",
        quote_literal_cstr(paths_tbl),
        quote_literal_cstr(iv_s));

    PG_TRY();
    {
        SPI_execute(sql.data, false, 0);
    }
    PG_CATCH();
    {
        FlushErrorState();
        ereport(WARNING,
                (errmsg("pg_nest: nest_hypertable skipped — "
                        "TimescaleDB may not be loaded or 'ts' column absent")));
    }
    PG_END_TRY();

    SPI_finish();
    PG_RETURN_VOID();
}

/* -------------------------------------------------------------------------
 * nest_distribute(source_table regclass, shard_column name) → void
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_distribute);

Datum
pg_nest_distribute(PG_FUNCTION_ARGS)
{
    Oid         relid       = PG_GETARG_OID(0);
    char       *shard_col   = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char       *relname     = get_rel_name(relid);
    char       *nspname;
    char       *paths_tbl;
    StringInfoData sql;
    char       *qfqn;   /* quoted fully-qualified name of source table */

    if (!pg_nest_enable_citus)
        PG_RETURN_VOID();

    if (!relname)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation %u does not exist", relid)));

    nspname   = get_namespace_name(get_rel_namespace(relid));
    paths_tbl = nest_paths_table_name(nspname, relname);
    qfqn      = psprintf("%s.%s",
                         quote_identifier(nspname),
                         quote_identifier(relname));

    SPI_connect();
    initStringInfo(&sql);

    /* Distribute the source table */
    appendStringInfo(&sql,
        "SELECT create_distributed_table(%s, %s)",
        quote_literal_cstr(qfqn),
        quote_literal_cstr(shard_col));

    PG_TRY();
    {
        SPI_execute(sql.data, false, 0);
    }
    PG_CATCH();
    {
        FlushErrorState();
        ereport(WARNING,
                (errmsg("pg_nest: nest_distribute source table skipped — "
                        "Citus may not be loaded")));
        SPI_finish();
        PG_RETURN_VOID();
    }
    PG_END_TRY();

    /*
     * Co-locate the path-store with the source table.
     * Because doc_id is derived from the source table's shard key, all path
     * rows for a given document land on the same worker as the document itself,
     * making JOIN operations between source and path-store fully local.
     */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT create_distributed_table(%s, %s,"
        "  colocate_with => %s)",
        quote_literal_cstr(paths_tbl),
        quote_literal_cstr(shard_col),
        quote_literal_cstr(qfqn));

    PG_TRY();
    {
        SPI_execute(sql.data, false, 0);
    }
    PG_CATCH();
    {
        FlushErrorState();
        ereport(WARNING,
                (errmsg("pg_nest: nest_distribute paths table skipped")));
    }
    PG_END_TRY();

    SPI_finish();
    PG_RETURN_VOID();
}
