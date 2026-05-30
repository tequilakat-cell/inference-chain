/*
 * nest_table.c
 *
 * nest_create()    – create path-store companion table, trigger, and indexes
 * nest_drop()      – remove trigger and drop path-store table
 * nest_reindex()   – truncate + bulk-rebuild path store from existing rows
 * _nest_trigger()  – row-level AFTER INSERT/UPDATE trigger handler
 *
 * Trigger design
 * --------------
 * Rather than issuing one INSERT per path entry (O(N) SPI round-trips per
 * document), the trigger collects all NestPathEntry values and issues a
 * single bulk INSERT with a VALUES list.  For a document with 50 paths this
 * reduces SPI overhead by ~50×.
 *
 * Index strategy
 * --------------
 *  B-tree (path, val_text) INCLUDE (doc_id, ts)  – covering, index-only scan
 *  B-tree (path, val_num)  INCLUDE (doc_id, ts)  – covering, index-only scan
 *  B-tree (doc_id)                               – reverse lookup
 *  BRIN   (ts)  pages_per_range=128              – tiny; great for time-series
 *  GIN    (path gin_trgm_ops)                    – path prefix/pattern queries
 *         (only when pg_trgm extension is present and enable_trgm = on)
 *  B-tree (ts, path) INCLUDE (doc_id, val_text)  – time-bounded path queries
 *         (only when a time column was specified)
 *
 * Registry
 * --------
 * nest_create inserts a row into pg_nest.nest_registry so that tooling
 * (views, future planner hooks, nest_reindex) can discover registered tables
 * without scanning pg_class.
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "commands/trigger.h"
#include "utils/rel.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"

#include "pg_nest.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static void
spi_run(const char *sql)
{
    int rc = SPI_execute(sql, false, 0);
    if (rc < 0)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("pg_nest SPI error %d: %s", rc, sql)));
}

/*
 * Escape a float8 to text, preserving full IEEE 754 precision.
 * Uses %.17g which guarantees roundtrip fidelity.
 */
static const char *
float8_to_str(double v)
{
    return psprintf("%.17g", v);
}

/*
 * Return true when pg_trgm extension is installed in this database.
 */
static bool
has_pg_trgm(void)
{
    int rc = SPI_execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'", true, 1);
    return (rc == SPI_OK_SELECT && SPI_processed > 0);
}

/* -------------------------------------------------------------------------
 * nest_create
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_create);

Datum
pg_nest_create(PG_FUNCTION_ARGS)
{
    Oid         relid      = PG_GETARG_OID(0);
    char       *jsonb_col  = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char       *id_col     = PG_ARGISNULL(2) ? "id"
                             : text_to_cstring(PG_GETARG_TEXT_PP(2));
    char       *time_col   = PG_ARGISNULL(3) ? NULL
                             : text_to_cstring(PG_GETARG_TEXT_PP(3));

    char       *relname    = get_rel_name(relid);
    char       *nspname;
    char       *paths_tbl;
    StringInfoData sql;
    bool        trgm;

    if (!relname)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation %u does not exist", relid)));

    nspname   = get_namespace_name(get_rel_namespace(relid));
    paths_tbl = nest_paths_table_name(nspname, relname);

    SPI_connect();
    initStringInfo(&sql);

    /* 1. Create the path-store table */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE TABLE IF NOT EXISTS %s ("
        "  doc_id      bigint       NOT NULL,"
        "  path        text         NOT NULL,"
        "  path_depth  int2         NOT NULL DEFAULT 0,"
        "  val_text    text,"
        "  val_num     float8,"
        "  val_bool    boolean,"
        "  val_null    boolean      NOT NULL DEFAULT false,"
        "  ts          timestamptz"
        ")",
        paths_tbl);
    spi_run(sql.data);

    /* 2. Covering B-tree indexes for text and numeric queries
     *    INCLUDE (doc_id, ts) enables index-only scans that avoid heap fetches */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE INDEX IF NOT EXISTS %s ON %s (path, val_text) INCLUDE (doc_id, ts)",
        quote_identifier(psprintf("_nest_%s_text_idx", relname)),
        paths_tbl);
    spi_run(sql.data);

    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE INDEX IF NOT EXISTS %s ON %s (path, val_num) INCLUDE (doc_id, ts)",
        quote_identifier(psprintf("_nest_%s_num_idx", relname)),
        paths_tbl);
    spi_run(sql.data);

    /* 3. Reverse-lookup index: find all paths belonging to a document */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE INDEX IF NOT EXISTS %s ON %s (doc_id)",
        quote_identifier(psprintf("_nest_%s_docid_idx", relname)),
        paths_tbl);
    spi_run(sql.data);

    /* 4. BRIN index on ts – append-friendly, orders-of-magnitude smaller than
     *    B-tree, no worse than B-tree for time-ordered inserts */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE INDEX IF NOT EXISTS %s ON %s USING brin (ts)"
        " WITH (pages_per_range = 128)",
        quote_identifier(psprintf("_nest_%s_ts_brin_idx", relname)),
        paths_tbl);
    spi_run(sql.data);

    /* 5. Optional GIN trigram index on path column for prefix/LIKE/regex queries */
    trgm = pg_nest_enable_trgm && has_pg_trgm();
    if (trgm)
    {
        resetStringInfo(&sql);
        appendStringInfo(&sql,
            "CREATE INDEX IF NOT EXISTS %s ON %s USING gin (path gin_trgm_ops)",
            quote_identifier(psprintf("_nest_%s_path_trgm_idx", relname)),
            paths_tbl);
        spi_run(sql.data);
    }

    /* 6. Optional time+path covering index for time-bounded path queries */
    if (time_col)
    {
        resetStringInfo(&sql);
        appendStringInfo(&sql,
            "CREATE INDEX IF NOT EXISTS %s ON %s (ts, path)"
            " INCLUDE (doc_id, val_text)"
            " WHERE ts IS NOT NULL",
            quote_identifier(psprintf("_nest_%s_ts_path_idx", relname)),
            paths_tbl);
        spi_run(sql.data);
    }

    /* 7. Install AFTER INSERT OR UPDATE trigger on the source table */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "CREATE OR REPLACE TRIGGER %s "
        "AFTER INSERT OR UPDATE ON %s.%s "
        "FOR EACH ROW EXECUTE FUNCTION pg_nest._nest_trigger_fn(%s, %s%s)",
        quote_identifier(psprintf("_nest_%s_trg", relname)),
        quote_identifier(nspname),
        quote_identifier(relname),
        quote_literal_cstr(jsonb_col),
        quote_literal_cstr(id_col),
        time_col ? psprintf(", %s", quote_literal_cstr(time_col)) : "");
    spi_run(sql.data);

    /* 8. Record in registry */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "INSERT INTO pg_nest.nest_registry"
        "  (relid, nspname, relname, jsonb_col, id_col, time_col)"
        " VALUES (%u, %s, %s, %s, %s, %s)"
        " ON CONFLICT (relid) DO UPDATE"
        "   SET jsonb_col = EXCLUDED.jsonb_col,"
        "       id_col    = EXCLUDED.id_col,"
        "       time_col  = EXCLUDED.time_col",
        relid,
        quote_literal_cstr(nspname),
        quote_literal_cstr(relname),
        quote_literal_cstr(jsonb_col),
        quote_literal_cstr(id_col),
        time_col ? quote_literal_cstr(time_col) : "NULL");
    spi_run(sql.data);

    SPI_finish();
    PG_RETURN_VOID();
}

/* -------------------------------------------------------------------------
 * nest_drop
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_drop);

Datum
pg_nest_drop(PG_FUNCTION_ARGS)
{
    Oid         relid    = PG_GETARG_OID(0);
    char       *relname  = get_rel_name(relid);
    char       *nspname;
    char       *paths_tbl;
    StringInfoData sql;

    if (!relname)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation %u does not exist", relid)));

    nspname   = get_namespace_name(get_rel_namespace(relid));
    paths_tbl = nest_paths_table_name(nspname, relname);

    SPI_connect();
    initStringInfo(&sql);

    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "DROP TRIGGER IF EXISTS %s ON %s.%s",
        quote_identifier(psprintf("_nest_%s_trg", relname)),
        quote_identifier(nspname),
        quote_identifier(relname));
    spi_run(sql.data);

    resetStringInfo(&sql);
    appendStringInfo(&sql, "DROP TABLE IF EXISTS %s CASCADE", paths_tbl);
    spi_run(sql.data);

    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "DELETE FROM pg_nest.nest_registry WHERE relid = %u", relid);
    spi_run(sql.data);

    SPI_finish();
    PG_RETURN_VOID();
}

/* -------------------------------------------------------------------------
 * nest_reindex(source_table regclass) → bigint
 *
 * Truncates and rebuilds the path store from scratch by scanning every row
 * in the source table.  More efficient than DELETE + trigger for large tables
 * because it does one bulk INSERT per source row via SPI.
 * Returns the number of path rows inserted.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_reindex);

Datum
pg_nest_reindex(PG_FUNCTION_ARGS)
{
    Oid         relid     = PG_GETARG_OID(0);
    char       *relname   = get_rel_name(relid);
    char       *nspname;
    char       *paths_tbl;
    StringInfoData sql;
    int         rc;
    int64       total_rows = 0;
    /* Fields from registry */
    char       *jsonb_col = NULL;
    char       *id_col    = NULL;
    char       *time_col  = NULL;
    uint64      nrows, i;

    if (!relname)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation %u does not exist", relid)));

    nspname   = get_namespace_name(get_rel_namespace(relid));
    paths_tbl = nest_paths_table_name(nspname, relname);

    SPI_connect();
    initStringInfo(&sql);

    /* Look up registry to get column names */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT jsonb_col, id_col, time_col"
        " FROM pg_nest.nest_registry WHERE relid = %u",
        relid);
    rc = SPI_execute(sql.data, true, 1);
    if (rc != SPI_OK_SELECT || SPI_processed == 0)
        ereport(ERROR,
                (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                 errmsg("table \"%s.%s\" is not registered with pg_nest; "
                        "call nest_create() first", nspname, relname)));

    {
        bool isnull;
        jsonb_col = DatumGetCString(
                        SPI_getbinval(SPI_tuptable->vals[0],
                                      SPI_tuptable->tupdesc, 1, &isnull));
        id_col    = DatumGetCString(
                        SPI_getbinval(SPI_tuptable->vals[0],
                                      SPI_tuptable->tupdesc, 2, &isnull));
        {
            Datum d = SPI_getbinval(SPI_tuptable->vals[0],
                                    SPI_tuptable->tupdesc, 3, &isnull);
            time_col = isnull ? NULL : DatumGetCString(d);
        }
    }

    /* Truncate existing path store */
    resetStringInfo(&sql);
    appendStringInfo(&sql, "TRUNCATE %s", paths_tbl);
    spi_run(sql.data);

    /* Scan source table and bulk-index each row */
    resetStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT %s, %s%s FROM %s.%s",
        quote_identifier(id_col),
        quote_identifier(jsonb_col),
        time_col ? psprintf(", %s", quote_identifier(time_col)) : "",
        quote_identifier(nspname),
        quote_identifier(relname));

    rc = SPI_execute(sql.data, true, 0);
    if (rc != SPI_OK_SELECT)
        ereport(ERROR, (errmsg("nest_reindex: failed to scan source table")));

    nrows = SPI_processed;

    for (i = 0; i < nrows; i++)
    {
        HeapTuple  row       = SPI_tuptable->vals[i];
        TupleDesc  tupdesc   = SPI_tuptable->tupdesc;
        bool       isnull;
        Datum      id_datum, jb_datum;
        int64      doc_id;
        Jsonb     *jb;
        NestPathEntry *entries;
        int        nentries, j;
        char      *ts_str  = NULL;
        StringInfoData ins;

        id_datum = SPI_getbinval(row, tupdesc, 1, &isnull);
        if (isnull) continue;
        doc_id = DatumGetInt64(id_datum);

        jb_datum = SPI_getbinval(row, tupdesc, 2, &isnull);
        if (isnull) continue;
        jb = DatumGetJsonbP(jb_datum);

        if (time_col)
        {
            Datum ts_datum = SPI_getbinval(row, tupdesc, 3, &isnull);
            if (!isnull)
                ts_str = DatumGetCString(
                             DirectFunctionCall1(timestamptz_out, ts_datum));
        }

        entries = nest_decompose_jsonb(jb, &nentries);
        if (nentries == 0)
            continue;

        initStringInfo(&ins);
        appendStringInfo(&ins,
            "INSERT INTO %s"
            " (doc_id, path, path_depth, val_text, val_num, val_bool, val_null, ts)"
            " VALUES",
            paths_tbl);

        for (j = 0; j < nentries; j++)
        {
            NestPathEntry *e = &entries[j];
            if (j > 0) appendStringInfoChar(&ins, ',');

            appendStringInfo(&ins, "(%lld,%s,%d,%s,%s,%s,%s,%s)",
                (long long) doc_id,
                quote_literal_cstr(e->path),
                e->depth,
                e->val_null ? "NULL" : quote_literal_cstr(e->val_text),
                e->has_num  ? float8_to_str(e->val_num) : "NULL",
                e->jbvtype == jbvBool
                    ? (e->val_bool ? "TRUE" : "FALSE") : "NULL",
                e->val_null ? "TRUE" : "FALSE",
                ts_str ? quote_literal_cstr(ts_str) : "NULL");
        }

        spi_run(ins.data);
        total_rows += nentries;
        pfree(ins.data);
    }

    SPI_finish();
    PG_RETURN_INT64(total_rows);
}

/* -------------------------------------------------------------------------
 * _nest_trigger_fn() – AFTER INSERT OR UPDATE trigger handler
 *
 * TG_ARGV[0] = JSONB column name
 * TG_ARGV[1] = ID column name
 * TG_ARGV[2] = time column name (optional)
 *
 * Collects all path entries then issues ONE batch INSERT with a VALUES list.
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_trigger);

Datum
pg_nest_trigger(PG_FUNCTION_ARGS)
{
    TriggerData    *tdata;
    Trigger        *trigger;
    HeapTuple       new_tuple;
    Relation        rel;
    TupleDesc       tupdesc;
    char           *jsonb_col, *id_col, *time_col;
    char           *nspname, *relname, *paths_tbl;
    AttrNumber      jcol_att, idcol_att, tcol_att;
    bool            isnull;
    Datum           jval, id_datum;
    Jsonb          *jb;
    int64           doc_id;
    NestPathEntry  *entries;
    int             nentries, i;
    char           *ts_str    = NULL;
    StringInfoData  ins;

    if (!CALLED_AS_TRIGGER(fcinfo))
        ereport(ERROR, (errmsg("_nest_trigger_fn must be called as a trigger")));

    tdata     = (TriggerData *) fcinfo->context;
    trigger   = tdata->tg_trigger;
    new_tuple = tdata->tg_newtuple;
    rel       = tdata->tg_relation;
    tupdesc   = RelationGetDescr(rel);

    if (trigger->tgnargs < 2)
        ereport(ERROR,
                (errmsg("_nest_trigger_fn: expected at least 2 arguments")));

    jsonb_col = trigger->tgargs[0];
    id_col    = trigger->tgargs[1];
    time_col  = trigger->tgnargs >= 3 ? trigger->tgargs[2] : NULL;

    nspname   = get_namespace_name(RelationGetNamespace(rel));
    relname   = RelationGetRelationName(rel);
    paths_tbl = nest_paths_table_name(nspname, relname);

    /* Locate column attribute numbers */
    jcol_att  = get_attnum(RelationGetRelid(rel), jsonb_col);
    idcol_att = get_attnum(RelationGetRelid(rel), id_col);
    tcol_att  = time_col ? get_attnum(RelationGetRelid(rel), time_col)
                         : InvalidAttrNumber;

    if (jcol_att == InvalidAttrNumber)
        ereport(ERROR,
                (errmsg("pg_nest: JSONB column \"%s\" not found in \"%s\"",
                        jsonb_col, relname)));
    if (idcol_att == InvalidAttrNumber)
        ereport(ERROR,
                (errmsg("pg_nest: ID column \"%s\" not found in \"%s\"",
                        id_col, relname)));

    /* Get the document ID */
    id_datum = heap_getattr(new_tuple, idcol_att, tupdesc, &isnull);
    if (isnull)
        return PointerGetDatum(new_tuple);

    doc_id = DatumGetInt64(id_datum);

    /* Get the JSONB value */
    jval = heap_getattr(new_tuple, jcol_att, tupdesc, &isnull);
    if (isnull)
        return PointerGetDatum(new_tuple);

    jb = DatumGetJsonbP(jval);

    /* Get the timestamp if configured */
    if (tcol_att != InvalidAttrNumber)
    {
        Datum ts_datum = heap_getattr(new_tuple, tcol_att, tupdesc, &isnull);
        if (!isnull)
            ts_str = DatumGetCString(
                         DirectFunctionCall1(timestamptz_out, ts_datum));
    }

    /* Decompose the JSONB document */
    entries  = nest_decompose_jsonb(jb, &nentries);
    if (nentries == 0)
        return PointerGetDatum(new_tuple);

    SPI_connect();
    initStringInfo(&ins);

    /* On UPDATE, remove old paths for this document */
    if (TRIGGER_FIRED_BY_UPDATE(tdata->tg_event))
    {
        char *del = psprintf("DELETE FROM %s WHERE doc_id = %lld",
                             paths_tbl, (long long) doc_id);
        spi_run(del);
    }

    /*
     * Batch INSERT: one SPI call for the entire document.
     * This reduces SPI overhead from O(paths) to O(1) per row trigger.
     */
    appendStringInfo(&ins,
        "INSERT INTO %s"
        " (doc_id, path, path_depth, val_text, val_num, val_bool, val_null, ts)"
        " VALUES",
        paths_tbl);

    for (i = 0; i < nentries; i++)
    {
        NestPathEntry *e = &entries[i];
        if (i > 0) appendStringInfoChar(&ins, ',');

        appendStringInfo(&ins, "(%lld,%s,%d,%s,%s,%s,%s,%s)",
            (long long) doc_id,
            quote_literal_cstr(e->path),
            e->depth,
            e->val_null ? "NULL" : quote_literal_cstr(e->val_text),
            e->has_num  ? float8_to_str(e->val_num) : "NULL",
            e->jbvtype == jbvBool ? (e->val_bool ? "TRUE" : "FALSE") : "NULL",
            e->val_null ? "TRUE" : "FALSE",
            ts_str ? quote_literal_cstr(ts_str) : "NULL");
    }

    spi_run(ins.data);
    SPI_finish();

    return PointerGetDatum(new_tuple);
}
