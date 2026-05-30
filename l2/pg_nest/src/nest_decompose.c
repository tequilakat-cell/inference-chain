/*
 * nest_decompose.c
 *
 * JSONB recursive walker: decomposes a Jsonb value into a flat array of
 * NestPathEntry (one per leaf node).
 *
 * Path encoding
 * -------------
 * Object keys use dot-separated notation.  Keys that contain '.', '[', ']',
 * '"', '\', or whitespace are wrapped in double-quotes to prevent ambiguity —
 * the same convention as PostgreSQL's JSONPath language:
 *
 *   {"a": {"b.c": 1}}  →  path = 'a."b.c"'
 *
 * Array indices use bracket notation appended directly after the parent path:
 *
 *   {"items": [10, 20]}  →  path = 'items[0]',  path = 'items[1]'
 *
 * Top-level array:
 *
 *   [10, 20]  →  path = '[0]', path = '[1]'
 *
 * Depth
 * -----
 * NestPathEntry.depth reflects the nesting level of the leaf (0 = top-level
 * key inside a root object; array indexing does not increase depth beyond the
 * containing object level).
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/jsonb.h"
#include "utils/builtins.h"
#include "utils/numeric.h"
#include "lib/stringinfo.h"

#include "pg_nest.h"

/* -------------------------------------------------------------------------
 * Key-escaping helpers
 * ------------------------------------------------------------------------- */

/*
 * Returns true when a JSON object key must be double-quoted in the path.
 * We quote when the key contains '.', '[', ']', '"', '\', any whitespace,
 * or when it is empty.
 */
static bool
key_needs_quoting(const char *key, int len)
{
    int i;

    if (len == 0)
        return true;

    for (i = 0; i < len; i++)
    {
        unsigned char c = (unsigned char) key[i];
        if (c == '.' || c == '[' || c == ']' || c == '"' || c == '\\' ||
            c == ' ' || c == '\t' || c == '\n' || c == '\r')
            return true;
    }
    return false;
}

/*
 * Append a key segment to the path StringInfo, quoting if necessary.
 * The caller must have already appended '.' (for non-top-level keys).
 */
static void
append_key_segment(StringInfo path, const char *key, int len)
{
    int i;

    if (!key_needs_quoting(key, len))
    {
        appendBinaryStringInfo(path, key, len);
        return;
    }

    appendStringInfoChar(path, '"');
    for (i = 0; i < len; i++)
    {
        if (key[i] == '"' || key[i] == '\\')
            appendStringInfoChar(path, '\\');
        appendStringInfoChar(path, key[i]);
    }
    appendStringInfoChar(path, '"');
}

/* -------------------------------------------------------------------------
 * Walker accumulator
 * ------------------------------------------------------------------------- */

typedef struct WalkerState
{
    NestPathEntry  *entries;
    int             nentries;
    int             cap;
} WalkerState;

static void
ws_init(WalkerState *ws)
{
    ws->cap      = 64;
    ws->nentries = 0;
    ws->entries  = palloc(ws->cap * sizeof(NestPathEntry));
}

static void
ws_append(WalkerState *ws, NestPathEntry *e)
{
    if (ws->nentries == ws->cap)
    {
        ws->cap *= 2;
        ws->entries = repalloc(ws->entries, ws->cap * sizeof(NestPathEntry));
    }
    ws->entries[ws->nentries++] = *e;
}

/* -------------------------------------------------------------------------
 * Helper: record a leaf node
 * ------------------------------------------------------------------------- */

static void
emit_leaf(WalkerState *ws, const char *path, int depth, JsonbValue *v)
{
    NestPathEntry e;

    e.path    = pstrdup(path);
    e.depth   = depth;
    e.has_num = false;
    e.val_bool = false;
    e.val_null = false;
    e.jbvtype  = v->type;

    switch (v->type)
    {
        case jbvNull:
            e.val_text = pstrdup("null");
            e.val_null = true;
            e.val_num  = 0.0;
            break;

        case jbvBool:
            e.val_bool = v->val.boolean;
            e.val_text = v->val.boolean ? pstrdup("true") : pstrdup("false");
            e.val_num  = v->val.boolean ? 1.0 : 0.0;
            e.has_num  = true;
            break;

        case jbvNumeric:
        {
            char *ns = DatumGetCString(
                           DirectFunctionCall1(numeric_out,
                               NumericGetDatum(v->val.numeric)));
            e.val_text = ns;
            e.val_num  = DatumGetFloat8(
                             DirectFunctionCall1(numeric_float8,
                                 NumericGetDatum(v->val.numeric)));
            e.has_num  = true;
            break;
        }

        case jbvString:
            e.val_text = palloc(v->val.string.len + 1);
            memcpy(e.val_text, v->val.string.val, v->val.string.len);
            e.val_text[v->val.string.len] = '\0';
            e.val_num  = 0.0;
            break;

        default:
            e.val_text = pstrdup("<unknown>");
            e.val_num  = 0.0;
            break;
    }

    ws_append(ws, &e);
}

/* -------------------------------------------------------------------------
 * Recursive descent walker
 * ------------------------------------------------------------------------- */

static void walk_container(JsonbContainer *jbc, StringInfo path, int depth, WalkerState *ws);

static void
walk_container(JsonbContainer *jbc, StringInfo path, int depth, WalkerState *ws)
{
    JsonbIterator       *it;
    JsonbValue           v;
    JsonbIteratorToken   tok;
    bool                 is_array;
    int                  arr_idx  = 0;
    int                  path_base;  /* path length before this container */

    if (depth > pg_nest_max_depth)
        return;

    it  = JsonbIteratorInit(jbc);
    tok = JsonbIteratorNext(&it, &v, false);  /* consume BEGIN_OBJECT/BEGIN_ARRAY */
    is_array = (tok == WJB_BEGIN_ARRAY);
    path_base = path->len;

    for (;;)
    {
        int seg_start;  /* path length before this element's segment */

        /* Restore path to container base for each new element */
        path->len         = path_base;
        path->data[path_base] = '\0';

        if (is_array)
        {
            /* Arrays: get value directly (skip_nested=true → binary for containers) */
            tok = JsonbIteratorNext(&it, &v, true);
            if (tok == WJB_END_ARRAY || tok == WJB_DONE)
                break;

            if (pg_nest_array_max_elems > 0 && arr_idx >= pg_nest_array_max_elems)
            {
                /* Skip remaining elements */
                while (tok != WJB_END_ARRAY && tok != WJB_DONE)
                    tok = JsonbIteratorNext(&it, &v, true);
                break;
            }

            /* Append [N] index */
            if (path->len > 0)
                appendStringInfo(path, "[%d]", arr_idx);
            else
                appendStringInfo(path, "[%d]", arr_idx);   /* top-level array */

            arr_idx++;
        }
        else
        {
            /* Objects: get key, then value */
            tok = JsonbIteratorNext(&it, &v, false);
            if (tok == WJB_END_OBJECT || tok == WJB_DONE)
                break;

            /* tok should be WJB_KEY */
            if (path->len > 0)
                appendStringInfoChar(path, '.');
            append_key_segment(path, v.val.string.val, v.val.string.len);

            /* Now get the value */
            tok = JsonbIteratorNext(&it, &v, true);
        }

        seg_start = path->len;  /* save length after building this segment */
        (void) seg_start;

        if (v.type == jbvBinary)
            walk_container(v.val.binary.data, path, depth + 1, ws);
        else
            emit_leaf(ws, path->data, depth, &v);
    }

    /* Restore to container base (cleanup) */
    path->len         = path_base;
    path->data[path_base] = '\0';
}

/* -------------------------------------------------------------------------
 * Public API: nest_decompose_jsonb
 * ------------------------------------------------------------------------- */

NestPathEntry *
nest_decompose_jsonb(Jsonb *jb, int *nentries)
{
    WalkerState    ws;
    StringInfoData path;

    ws_init(&ws);
    initStringInfo(&path);

    walk_container(&jb->root, &path, 0, &ws);

    pfree(path.data);
    *nentries = ws.nentries;
    return ws.entries;
}

/* -------------------------------------------------------------------------
 * SQL SRF: nest_decompose(jsonb)
 * Returns SETOF (path text, val_text text, val_num float8,
 *                val_bool boolean, val_null boolean, depth int)
 * ------------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_nest_decompose);

Datum
pg_nest_decompose(PG_FUNCTION_ARGS)
{
    FuncCallContext *fctx;
    NestPathEntry   *entries;
    int              nentries;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldctx;
        Jsonb         *jb;
        TupleDesc      tupdesc;

        fctx   = SRF_FIRSTCALL_INIT();
        oldctx = MemoryContextSwitchTo(fctx->multi_call_memory_ctx);

        jb      = PG_GETARG_JSONB_P(0);
        entries = nest_decompose_jsonb(jb, &nentries);

        fctx->user_fctx = entries;
        fctx->max_calls  = (uint64) nentries;

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("nest_decompose must be called as a set-returning function")));
        fctx->attinmeta = TupleDescGetAttInMetadata(tupdesc);

        MemoryContextSwitchTo(oldctx);
    }

    fctx    = SRF_PERCALL_SETUP();
    entries = (NestPathEntry *) fctx->user_fctx;

    if (fctx->call_cntr < fctx->max_calls)
    {
        NestPathEntry *e = &entries[fctx->call_cntr];
        Datum          values[6];
        bool           nulls[6]  = {false, false, false, false, false, false};
        HeapTuple      htup;

        values[0] = CStringGetTextDatum(e->path);
        values[1] = CStringGetTextDatum(e->val_text);

        if (e->has_num)
            values[2] = Float8GetDatum(e->val_num);
        else
            nulls[2] = true;

        values[3] = BoolGetDatum(e->val_bool);
        values[4] = BoolGetDatum(e->val_null);
        values[5] = Int32GetDatum(e->depth);

        htup = heap_form_tuple(fctx->attinmeta->tupdesc, values, nulls);
        SRF_RETURN_NEXT(fctx, HeapTupleGetDatum(htup));
    }

    SRF_RETURN_DONE(fctx);
}
