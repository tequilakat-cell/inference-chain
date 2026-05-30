/*
 * inft_chunk.c — paragraph + sentence chunker for the pg_inft extension.
 *
 * Exports:
 *   inft_chunk_text(input text, target_size int, overlap int)
 *       → SETOF text
 *
 * Algorithm:
 *   1. Split on "\n\n" (double newline) for primary paragraph boundaries.
 *   2. If a paragraph > target_size chars, split further at ". "/" ? "/" ! "
 *      sentence boundaries.
 *   3. Merge segments < 50 chars with the next one (so short headings attach
 *      to the following body text).
 *   4. For each chunk (except the first), prepend `overlap` chars from the
 *      end of the previous chunk as a continuity prefix.
 *   5. Skip chunks that are < 30 chars after trimming.
 *
 * Default values: target_size = 300, overlap = 50.
 */

#include "pg_inft.h"

#include <string.h>
#include <ctype.h>

#include "funcapi.h"
#include "utils/array.h"

/* ── Internal chunk list ─────────────────────────────────────────────────── */

#define MAX_CHUNKS 8192   /* hard upper bound on number of chunks */

typedef struct ChunkList
{
    char  **chunks;
    int     nchunks;
    int     capacity;
} ChunkList;

static ChunkList *
chunk_list_new(void)
{
    ChunkList *cl  = (ChunkList *) palloc(sizeof(ChunkList));
    cl->capacity   = 256;
    cl->nchunks    = 0;
    cl->chunks     = (char **) palloc(cl->capacity * sizeof(char *));
    return cl;
}

static void
chunk_list_append(ChunkList *cl, const char *s, size_t len)
{
    char *copy;
    if (cl->nchunks >= cl->capacity)
    {
        if (cl->capacity >= MAX_CHUNKS) return;
        cl->capacity *= 2;
        if (cl->capacity > MAX_CHUNKS) cl->capacity = MAX_CHUNKS;
        cl->chunks = (char **) repalloc(cl->chunks, cl->capacity * sizeof(char *));
    }
    copy = (char *) palloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';
    cl->chunks[cl->nchunks++] = copy;
}

/* ── String helpers ─────────────────────────────────────────────────────── */

/* inft_ltrim — skip leading whitespace, return pointer into s */
static const char *
inft_ltrim(const char *s)
{
    while (*s && isspace((unsigned char)*s))
        s++;
    return s;
}

/* rtrim_len — return length without trailing whitespace */
static size_t
rtrim_len(const char *s, size_t len)
{
    while (len > 0 && isspace((unsigned char)s[len - 1]))
        len--;
    return len;
}

/* ── Sentence splitter ──────────────────────────────────────────────────── */

/*
 * split_at_sentences — split `text` (len bytes) into sentence segments
 * wherever ". ", "? ", or "! " occurs, appending to `out`.
 */
static void
split_at_sentences(ChunkList *out, const char *text, size_t len, int target_size)
{
    size_t start = 0;
    size_t i;
    char   c;

    for (i = 0; i + 1 < len; i++)
    {
        c = text[i];
        if ((c == '.' || c == '?' || c == '!') && text[i + 1] == ' ')
        {
            size_t seg_len = (i + 1) - start;   /* include the punctuation */
            if ((int)seg_len >= target_size || i + 1 >= len - 1)
            {
                /* Emit this sentence as a segment */
                size_t tlen = rtrim_len(text + start, seg_len);
                if (tlen > 0)
                    chunk_list_append(out, text + start, tlen);
                start = i + 2;   /* skip the space after punctuation */
                i++;             /* outer loop will also increment */
            }
        }
    }
    /* Tail */
    if (start < len)
    {
        size_t tlen = rtrim_len(text + start, len - start);
        if (tlen > 0)
            chunk_list_append(out, text + start, tlen);
    }
}

/* ── Main chunker ───────────────────────────────────────────────────────── */

/*
 * do_chunk — perform the full chunking algorithm, return a ChunkList of
 * finalised (overlap-prefixed, trimmed) chunk strings.
 */
static ChunkList *
do_chunk(const char *input, size_t input_len, int target_size, int overlap)
{
    ChunkList  *paragraphs;
    ChunkList  *segments;
    ChunkList  *merged;
    ChunkList  *result;
    const char *prev     = NULL;
    size_t      prev_len = 0;
    size_t      start;
    size_t      i;
    int         p, s, m_idx;

    /* ── Step 1: split on \n\n ──────────────────────────────────────────── */
    paragraphs = chunk_list_new();
    start = 0;
    for (i = 0; i + 1 < input_len; i++)
    {
        if (input[i] == '\n' && input[i + 1] == '\n')
        {
            if (i > start)
            {
                const char *seg  = inft_ltrim(input + start);
                size_t      slen = rtrim_len(seg, (i - start) - (size_t)(seg - (input + start)));
                if (slen > 0)
                    chunk_list_append(paragraphs, seg, slen);
            }
            start = i + 2;
            i++;
        }
    }
    if (start < input_len)
    {
        const char *seg  = inft_ltrim(input + start);
        size_t      slen = rtrim_len(seg, input_len - start - (size_t)(seg - (input + start)));
        if (slen > 0)
            chunk_list_append(paragraphs, seg, slen);
    }

    /* ── Step 2: sub-split large paragraphs at sentence boundaries ──────── */
    segments = chunk_list_new();
    for (p = 0; p < paragraphs->nchunks; p++)
    {
        const char *para = paragraphs->chunks[p];
        size_t      plen = strlen(para);

        if ((int)plen > target_size)
            split_at_sentences(segments, para, plen, target_size);
        else
            chunk_list_append(segments, para, plen);
    }

    /* ── Step 3: merge short segments (< 50 chars) with next ───────────── */
    merged = chunk_list_new();
    for (s = 0; s < segments->nchunks; )
    {
        const char *cur  = segments->chunks[s];
        size_t      clen = strlen(cur);

        if ((int)clen < 50 && s + 1 < segments->nchunks)
        {
            const char *nxt  = segments->chunks[s + 1];
            size_t      nlen = strlen(nxt);
            size_t      mlen = clen + 1 + nlen;
            char       *mc   = (char *) palloc(mlen + 1);
            memcpy(mc, cur, clen);
            mc[clen] = ' ';
            memcpy(mc + clen + 1, nxt, nlen);
            mc[mlen] = '\0';
            segments->chunks[s + 1] = mc;
            s++;
        }
        else
        {
            chunk_list_append(merged, cur, clen);
            s++;
        }
    }

    /* ── Steps 4+5: apply overlap prefix and filter short chunks ─────────── */
    result = chunk_list_new();
    for (m_idx = 0; m_idx < merged->nchunks; m_idx++)
    {
        const char *seg  = merged->chunks[m_idx];
        size_t      slen = strlen(seg);

        /* Trim */
        const char *tseg = inft_ltrim(seg);
        size_t      tlen = rtrim_len(tseg, slen - (size_t)(tseg - seg));

        if ((int)tlen < 30)
        {
            /* Update prev for overlap even if we skip this chunk */
            prev     = tseg;
            prev_len = tlen;
            continue;
        }

        if (m_idx == 0 || prev == NULL || overlap <= 0)
        {
            /* First chunk: no prefix */
            chunk_list_append(result, tseg, tlen);
        }
        else
        {
            /* Prepend `overlap` chars from end of previous chunk */
            size_t ov;
            size_t off;
            size_t total;
            char  *buf;
            ov    = (size_t)overlap;
            if (ov > prev_len) ov = prev_len;
            off   = prev_len - ov;
            total = ov + tlen;
            buf   = (char *) palloc(total + 1);
            memcpy(buf, prev + off, ov);
            memcpy(buf + ov, tseg, tlen);
            buf[total] = '\0';
            chunk_list_append(result, buf, total);
        }

        prev     = tseg;
        prev_len = tlen;
    }

    return result;
}

/* ── SRF state ──────────────────────────────────────────────────────────── */

typedef struct ChunkState
{
    ChunkList *list;
    int        cur;
} ChunkState;

/* ── inft_chunk_text SRF implementation ─────────────────────────────────── */

PG_FUNCTION_INFO_V1(inft_chunk_text);
Datum
inft_chunk_text(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    ChunkState      *state;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldctx;

        funcctx = SRF_FIRSTCALL_INIT();
        oldctx  = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        state = (ChunkState *) palloc0(sizeof(ChunkState));

        if (!PG_ARGISNULL(0))
        {
            text  *input_t    = PG_GETARG_TEXT_PP(0);
            int    target_sz  = PG_ARGISNULL(1) ? 300 : PG_GETARG_INT32(1);
            int    overlap    = PG_ARGISNULL(2) ? 50  : PG_GETARG_INT32(2);

            const char *input_s  = VARDATA_ANY(input_t);
            size_t      input_l  = VARSIZE_ANY_EXHDR(input_t);

            if (target_sz <= 0) target_sz = 300;
            if (overlap < 0)    overlap   = 0;

            state->list = do_chunk(input_s, input_l, target_sz, overlap);
        }
        else
        {
            state->list = chunk_list_new();  /* empty */
        }

        state->cur = 0;
        funcctx->user_fctx  = state;
        funcctx->max_calls  = (uint64) (state->list ? state->list->nchunks : 0);

        MemoryContextSwitchTo(oldctx);
    }

    funcctx = SRF_PERCALL_SETUP();
    state   = (ChunkState *) funcctx->user_fctx;

    if (state->list && state->cur < state->list->nchunks)
    {
        const char *chunk = state->list->chunks[state->cur++];
        SRF_RETURN_NEXT(funcctx, PointerGetDatum(cstring_to_text(chunk)));
    }

    SRF_RETURN_DONE(funcctx);
}
