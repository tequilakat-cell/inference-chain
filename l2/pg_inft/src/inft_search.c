/*
 * inft_search.c — staged BM25 + trigram retrieval pipeline for pg_inft.
 *
 * Exports:
 *   inft_search(question text, model_id text, lim int)
 *       → SETOF inft_search_result
 *
 * The pipeline runs a single CTE query that:
 *   1. Scores question-level tsvector matches  (BM25 via ts_rank_cd)
 *   2. Scores chunk-level tsvector matches     (BM25 via ts_rank_cd)
 *   3. Merges and scores trigram similarity    (word_similarity from pg_trgm)
 *   4. Combines: score = 0.5*qr + 0.2*tr + 0.3*trgm
 *   5. Filters by the pg_inft.min_similarity GUC
 *
 * If pg_trgm is not installed, word_similarity() is unavailable; the function
 * falls back to a trgm-free query using only the BM25 components.
 */

#include "pg_inft.h"
#include "inft_keccak.h"   /* only needed for the VARHDRSZ etc. pull-in */

#include <string.h>

#include "executor/spi.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"

/* ── Return-type composite descriptor ──────────────────────────────────────
 * inft_search_result:
 *   (id bigint, job_id text, miner_address text, model_id text,
 *    question_text text, thinking_text text, answer_text text, score float8)
 */
#define RESULT_NCOLS 8

/* Column indices (0-based) in SPI result */
#define COL_ID            1
#define COL_JOB_ID        2
#define COL_MINER_ADDRESS 3
#define COL_MODEL_ID      4
#define COL_QUESTION_TEXT 5
#define COL_THINKING_TEXT 6
#define COL_ANSWER_TEXT   7
#define COL_SCORE         8

/* ── Full CTE query (with trigram) ─────────────────────────────────────── */
static const char *QUERY_WITH_TRGM =
"WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),\n"
"q_hits AS (\n"
"    SELECT tl.id, ts_rank_cd(tl.question_tsv, q.tsq, 4) AS qr, 0.0::float8 AS tr\n"
"    FROM inft.inft_thought_log tl, q\n"
"    WHERE tl.question_tsv @@ q.tsq AND ($2 = '' OR tl.model_id = $2)\n"
"    ORDER BY qr DESC LIMIT 80\n"
"),\n"
"c_hits AS (\n"
"    SELECT tc.thought_id AS id, 0.0::float8 AS qr,\n"
"           max(ts_rank_cd(tc.chunk_tsv, q.tsq, 4)) AS tr\n"
"    FROM inft.inft_thought_chunks tc\n"
"    JOIN inft.inft_thought_log tl ON tl.id = tc.thought_id, q\n"
"    WHERE tc.chunk_tsv @@ q.tsq AND ($2 = '' OR tl.model_id = $2)\n"
"    GROUP BY tc.thought_id ORDER BY tr DESC LIMIT 40\n"
"),\n"
"merged AS (\n"
"    SELECT id, max(qr) AS qr, max(tr) AS tr\n"
"    FROM (SELECT * FROM q_hits UNION ALL SELECT * FROM c_hits) x\n"
"    GROUP BY id\n"
"),\n"
"rescored AS (\n"
"    SELECT m.id, m.qr, m.tr,\n"
"           COALESCE(word_similarity($1, tl.question_text), 0.0) AS trgm\n"
"    FROM merged m JOIN inft.inft_thought_log tl ON tl.id = m.id\n"
"),\n"
"final AS (\n"
"    SELECT id, 0.5*qr + 0.2*tr + 0.3*trgm AS score\n"
"    FROM rescored\n"
"    WHERE 0.5*qr + 0.2*tr + 0.3*trgm >= $3\n"
"    ORDER BY score DESC LIMIT $4\n"
")\n"
"SELECT tl.id, tl.job_id, tl.miner_address, tl.model_id,\n"
"       tl.question_text, tl.thinking_text, tl.answer_text, f.score\n"
"FROM final f JOIN inft.inft_thought_log tl ON tl.id = f.id\n"
"ORDER BY f.score DESC";

/* ── Fallback CTE query (without word_similarity) ──────────────────────── */
static const char *QUERY_NO_TRGM =
"WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),\n"
"q_hits AS (\n"
"    SELECT tl.id, ts_rank_cd(tl.question_tsv, q.tsq, 4) AS qr, 0.0::float8 AS tr\n"
"    FROM inft.inft_thought_log tl, q\n"
"    WHERE tl.question_tsv @@ q.tsq AND ($2 = '' OR tl.model_id = $2)\n"
"    ORDER BY qr DESC LIMIT 80\n"
"),\n"
"c_hits AS (\n"
"    SELECT tc.thought_id AS id, 0.0::float8 AS qr,\n"
"           max(ts_rank_cd(tc.chunk_tsv, q.tsq, 4)) AS tr\n"
"    FROM inft.inft_thought_chunks tc\n"
"    JOIN inft.inft_thought_log tl ON tl.id = tc.thought_id, q\n"
"    WHERE tc.chunk_tsv @@ q.tsq AND ($2 = '' OR tl.model_id = $2)\n"
"    GROUP BY tc.thought_id ORDER BY tr DESC LIMIT 40\n"
"),\n"
"merged AS (\n"
"    SELECT id, max(qr) AS qr, max(tr) AS tr\n"
"    FROM (SELECT * FROM q_hits UNION ALL SELECT * FROM c_hits) x\n"
"    GROUP BY id\n"
"),\n"
"final AS (\n"
"    SELECT id, 0.5*qr + 0.2*tr AS score\n"
"    FROM merged\n"
"    WHERE 0.5*qr + 0.2*tr >= $3\n"
"    ORDER BY score DESC LIMIT $4\n"
")\n"
"SELECT tl.id, tl.job_id, tl.miner_address, tl.model_id,\n"
"       tl.question_text, tl.thinking_text, tl.answer_text, f.score\n"
"FROM final f JOIN inft.inft_thought_log tl ON tl.id = f.id\n"
"ORDER BY f.score DESC";

/* ── Per-call state stored in SRF context ───────────────────────────────── */
typedef struct SearchState
{
    int      nrows;
    int      cur;
    SPITupleTable *tuptable;
    TupleDesc     tupdesc;      /* result composite type descriptor */
} SearchState;

/* ── inft_search SRF implementation ─────────────────────────────────────── */

PG_FUNCTION_INFO_V1(inft_search);
Datum
inft_search(PG_FUNCTION_ARGS)
{
    FuncCallContext  *funcctx;
    SearchState      *state;

    /* ── First call: run the SPI query and stash results ───────────────── */
    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        TupleDesc      outtupdesc;
        TypeFuncClass  tfc;
        text          *question_t;
        text          *model_id_t;
        int            lim;
        const char    *question;
        const char    *model_id;
        double         min_sim;
        Oid            argtypes[4];
        Datum          args[4];
        char           nulls[4];
        bool           trgm_ok;
        int            spi_ret;

        funcctx = SRF_FIRSTCALL_INIT();
        oldctx  = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        /* Build the result type descriptor from the SQL-declared composite type */
        tfc = get_call_result_type(fcinfo, NULL, &outtupdesc);
        if (tfc != TYPEFUNC_COMPOSITE)
            ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("inft_search: cannot determine result type")));
        BlessTupleDesc(outtupdesc);

        state = (SearchState *) palloc0(sizeof(SearchState));
        state->tupdesc = outtupdesc;
        funcctx->user_fctx = state;

        /* ── Collect function arguments ─────────────────────────────────── */
        question_t = PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_PP(0);
        model_id_t = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
        lim        = PG_ARGISNULL(2) ? 5    : PG_GETARG_INT32(2);

        question = question_t ? text_to_cstring(question_t) : "";
        model_id = model_id_t ? text_to_cstring(model_id_t) : "";

        if (lim <= 0) lim = 5;

        min_sim = pg_inft_min_similarity;

        /* ── Execute via SPI with PG_TRY for trgm fallback ─────────────── */
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR,
                (errcode(ERRCODE_CONNECTION_FAILURE),
                 errmsg("inft_search: SPI_connect failed")));

        argtypes[0] = TEXTOID;
        argtypes[1] = TEXTOID;
        argtypes[2] = FLOAT8OID;
        argtypes[3] = INT4OID;
        nulls[0] = nulls[1] = nulls[2] = nulls[3] = ' ';

        args[0] = CStringGetTextDatum(question);
        args[1] = CStringGetTextDatum(model_id);
        args[2] = Float8GetDatum(min_sim);
        args[3] = Int32GetDatum(lim);

        trgm_ok = true;
        spi_ret = -1;

        PG_TRY();
        {
            spi_ret = SPI_execute_with_args(
                QUERY_WITH_TRGM, 4, argtypes, args, nulls, true, 0
            );
        }
        PG_CATCH();
        {
            /* word_similarity() not available — suppress and retry */
            FlushErrorState();
            trgm_ok = false;
        }
        PG_END_TRY();

        if (!trgm_ok || spi_ret != SPI_OK_SELECT)
        {
            /* Retry without trigram */
            spi_ret = SPI_execute_with_args(
                QUERY_NO_TRGM, 4, argtypes, args, nulls, true, 0
            );
        }

        if (spi_ret != SPI_OK_SELECT)
        {
            SPI_finish();
            ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("inft_search: SPI query failed (ret=%d)", spi_ret)));
        }

        state->nrows    = (int) SPI_processed;
        state->cur      = 0;
        state->tuptable = SPI_tuptable;

        /* Keep SPI results alive in our memory context */
        if (state->nrows > 0)
            SPI_tuptable = NULL;   /* prevent SPI_finish from freeing it */

        SPI_finish();

        funcctx->max_calls = (uint64) state->nrows;

        MemoryContextSwitchTo(oldctx);
    }

    /* ── Subsequent calls: emit one row per iteration ──────────────────── */
    funcctx = SRF_PERCALL_SETUP();
    state   = (SearchState *) funcctx->user_fctx;

    if (state->cur < state->nrows)
    {
        HeapTuple  spi_tuple;
        TupleDesc  spi_desc;
        Datum      values[RESULT_NCOLS];
        bool       nulls2[RESULT_NCOLS];
        bool       isnull;
        Datum      d;
        HeapTuple  out;

        spi_tuple = state->tuptable->vals[state->cur];
        spi_desc  = state->tuptable->tupdesc;
        state->cur++;

        /* id bigint */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_ID, &isnull);
        values[0] = isnull ? 0 : d;
        nulls2[0] = isnull;

        /* job_id text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_JOB_ID, &isnull);
        values[1] = isnull ? CStringGetTextDatum("") : d;
        nulls2[1] = false;

        /* miner_address text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_MINER_ADDRESS, &isnull);
        values[2] = isnull ? CStringGetTextDatum("") : d;
        nulls2[2] = false;

        /* model_id text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_MODEL_ID, &isnull);
        values[3] = isnull ? CStringGetTextDatum("") : d;
        nulls2[3] = false;

        /* question_text text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_QUESTION_TEXT, &isnull);
        values[4] = isnull ? CStringGetTextDatum("") : d;
        nulls2[4] = false;

        /* thinking_text text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_THINKING_TEXT, &isnull);
        values[5] = isnull ? CStringGetTextDatum("") : d;
        nulls2[5] = isnull;

        /* answer_text text */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_ANSWER_TEXT, &isnull);
        values[6] = isnull ? CStringGetTextDatum("") : d;
        nulls2[6] = isnull;

        /* score float8 */
        d = SPI_getbinval(spi_tuple, spi_desc, COL_SCORE, &isnull);
        values[7] = isnull ? Float8GetDatum(0.0) : d;
        nulls2[7] = false;

        out = heap_form_tuple(state->tupdesc, values, nulls2);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(out));
    }

    SRF_RETURN_DONE(funcctx);
}
