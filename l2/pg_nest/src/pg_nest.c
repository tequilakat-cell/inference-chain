/*
 * pg_nest.c
 *
 * Extension entry point: _PG_init, _PG_fini, GUC registration,
 * PG_MODULE_MAGIC, and the nest_paths_table_name() helper.
 */

#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "utils/guc.h"
#include "utils/elog.h"
#include "lib/stringinfo.h"

#include "pg_nest.h"

PG_MODULE_MAGIC;

/* -------------------------------------------------------------------------
 * GUCs
 * ------------------------------------------------------------------------- */

bool  pg_nest_enable_citus        = true;
bool  pg_nest_enable_timescaledb  = true;
bool  pg_nest_enable_trgm         = true;
int   pg_nest_max_depth           = 32;
int   pg_nest_array_max_elems     = 64;

/* -------------------------------------------------------------------------
 * _PG_init / _PG_fini
 * ------------------------------------------------------------------------- */

void _PG_init(void);
void _PG_fini(void);

void
_PG_init(void)
{
    DefineCustomBoolVariable(
        "pg_nest.enable_citus",
        "Enable Citus distribution helpers in pg_nest",
        NULL,
        &pg_nest_enable_citus,
        true,
        PGC_SUSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomBoolVariable(
        "pg_nest.enable_timescaledb",
        "Enable TimescaleDB hypertable helpers in pg_nest",
        NULL,
        &pg_nest_enable_timescaledb,
        true,
        PGC_SUSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomBoolVariable(
        "pg_nest.enable_trgm",
        "Create GIN trigram index on path column when pg_trgm is available",
        NULL,
        &pg_nest_enable_trgm,
        true,
        PGC_SUSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomIntVariable(
        "pg_nest.max_depth",
        "Maximum JSONB nesting depth to index (deeper paths are silently skipped)",
        NULL,
        &pg_nest_max_depth,
        32, 1, 256,
        PGC_SUSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomIntVariable(
        "pg_nest.array_max_elems",
        "Maximum number of array elements to expand per JSONB array (0 = unlimited)",
        NULL,
        &pg_nest_array_max_elems,
        64, 0, 100000,
        PGC_SUSET,
        0,
        NULL, NULL, NULL
    );

    MarkGUCPrefixReserved("pg_nest");
}

void
_PG_fini(void)
{
    /* nothing to clean up */
}

/* -------------------------------------------------------------------------
 * pg_nest_version()
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_version);

Datum
pg_nest_version(PG_FUNCTION_ARGS)
{
    PG_RETURN_TEXT_P(cstring_to_text("1.0"));
}

/* -------------------------------------------------------------------------
 * Helper: construct path-store table name
 *
 * Returns palloc'd string: schema._nest_<relname>_paths
 * If schema is NULL, uses "public".
 * ------------------------------------------------------------------------- */

char *
nest_paths_table_name(const char *schema, const char *relname)
{
    StringInfoData buf;

    initStringInfo(&buf);
    appendStringInfo(&buf, "%s._nest_%s_paths",
                     schema ? schema : "public",
                     relname);
    return buf.data;
}
