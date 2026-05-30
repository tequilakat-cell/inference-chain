/*
 * inft_proof.c — content-hash computation, Ethereum personal-sign hash,
 *                and the BEFORE INSERT trigger that validates proof sigs.
 *
 * Functions exported to SQL:
 *   inft_content_hash(job_id, question, thinking, answer) → bytea
 *   inft_eth_personal_hash(content_hash bytea)            → bytea
 *   inft_proof_trigger_fn()                               → trigger
 */

#include "pg_inft.h"
#include "inft_keccak.h"

#include <string.h>

#include "commands/trigger.h"
#include "executor/spi.h"
#include "utils/rel.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

/* ── helpers ─────────────────────────────────────────────────────────────── */

/*
 * write_len4_field — append a 4-byte big-endian length prefix followed by
 * `len` bytes of `data` to the buffer starting at `*buf`.
 * Advances *buf by (4 + len) bytes.
 */
static void
write_len4_field(uint8_t **buf, const char *data, size_t len)
{
    uint32_t n = (uint32_t) len;
    (*buf)[0] = (uint8_t)(n >> 24);
    (*buf)[1] = (uint8_t)(n >> 16);
    (*buf)[2] = (uint8_t)(n >> 8);
    (*buf)[3] = (uint8_t)(n);
    *buf += 4;
    if (len > 0)
    {
        memcpy(*buf, data, len);
        *buf += len;
    }
}

/* ── inft_content_hash ───────────────────────────────────────────────────── */

/*
 * inft_content_hash(job_id text, question text, thinking text, answer text)
 *    → bytea
 *
 * Computes:
 *   keccak256(
 *     len4(job_id)   || job_id   ||
 *     len4(question) || question ||
 *     len4(thinking) || thinking ||
 *     len4(answer)   || answer
 *   )
 *
 * Each len4() is a 4-byte big-endian uint32 giving the byte length of the
 * following UTF-8 string.  NULL fields are treated as empty strings.
 */
PG_FUNCTION_INFO_V1(inft_content_hash);
Datum
inft_content_hash(PG_FUNCTION_ARGS)
{
    const char *job_id   = PG_ARGISNULL(0) ? "" : text_to_cstring(PG_GETARG_TEXT_PP(0));
    const char *question = PG_ARGISNULL(1) ? "" : text_to_cstring(PG_GETARG_TEXT_PP(1));
    const char *thinking = PG_ARGISNULL(2) ? "" : text_to_cstring(PG_GETARG_TEXT_PP(2));
    const char *answer   = PG_ARGISNULL(3) ? "" : text_to_cstring(PG_GETARG_TEXT_PP(3));

    size_t jlen = strlen(job_id);
    size_t qlen = strlen(question);
    size_t tlen = strlen(thinking);
    size_t alen = strlen(answer);

    /* Total buffer: 4 * 4-byte prefixes + 4 string payloads */
    size_t total = 4*4 + jlen + qlen + tlen + alen;

    uint8_t *buf_start = (uint8_t *) palloc(total);
    uint8_t *ptr       = buf_start;

    uint8_t hash[32];
    bytea  *result;

    write_len4_field(&ptr, job_id,   jlen);
    write_len4_field(&ptr, question, qlen);
    write_len4_field(&ptr, thinking, tlen);
    write_len4_field(&ptr, answer,   alen);

    keccak256(buf_start, total, hash);

    result = (bytea *) palloc(VARHDRSZ + 32);
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash, 32);

    PG_RETURN_BYTEA_P(result);
}

/* ── inft_eth_personal_hash ──────────────────────────────────────────────── */

/*
 * inft_eth_personal_hash(content_hash bytea) → bytea
 *
 * Computes:
 *   keccak256("\x19Ethereum Signed Message:\n32" || content_hash)
 *
 * The prefix is exactly 28 bytes:
 *   0x19  (1 byte)
 *   "Ethereum Signed Message:\n"  (26 bytes, note literal \n = 0x0a)
 *   "32"  (2 ASCII bytes)
 *
 * Total prefix = 28 bytes.  content_hash must be exactly 32 bytes.
 */
PG_FUNCTION_INFO_V1(inft_eth_personal_hash);
Datum
inft_eth_personal_hash(PG_FUNCTION_ARGS)
{
    bytea      *chash  = PG_GETARG_BYTEA_PP(0);
    size_t      clen   = VARSIZE_ANY_EXHDR(chash);
    const uint8_t *cdata = (const uint8_t *) VARDATA_ANY(chash);

    /* The Ethereum personal-sign prefix: 28 bytes */
    static const uint8_t PREFIX[28] = {
        0x19,
        'E','t','h','e','r','e','u','m',' ',
        'S','i','g','n','e','d',' ','M','e','s','s','a','g','e',':','\n',
        '3','2'
    };

    uint8_t payload[28 + 32];
    uint8_t hash[32];
    bytea  *result;

    if (clen != 32)
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("inft_eth_personal_hash: content_hash must be 32 bytes, got %zu", clen)));

    memcpy(payload, PREFIX, 28);
    memcpy(payload + 28, cdata, 32);

    keccak256(payload, sizeof(payload), hash);

    result = (bytea *) palloc(VARHDRSZ + 32);
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash, 32);

    PG_RETURN_BYTEA_P(result);
}

/* ── inft_proof_trigger_fn ───────────────────────────────────────────────── */

/*
 * inft_proof_trigger_fn() → trigger
 *
 * BEFORE INSERT on pg_inft.inft_thought_log.
 *
 * Steps:
 *  1. Compute the expected content_hash from (job_id, question_text,
 *     thinking_text, answer_text) in the NEW row.
 *  2. Compare with NEW.content_hash — ERROR if mismatch.
 *  3. Call pg_inft.inft_eth_verify(content_hash, proof_sig, miner_address)
 *     via SPI.
 *  4. If it returns false and require_proof_verification is on, ERROR.
 *
 * Returns the (potentially unmodified) NEW row so the INSERT proceeds.
 */
PG_FUNCTION_INFO_V1(inft_proof_trigger_fn);
Datum
inft_proof_trigger_fn(PG_FUNCTION_ARGS)
{
    TriggerData    *tg = (TriggerData *) fcinfo->context;
    HeapTuple       new_row;
    TupleDesc       tupdesc;
    bool            isnull;
    AttrNumber      att_job_id;
    AttrNumber      att_miner_address;
    AttrNumber      att_question_text;
    AttrNumber      att_thinking_text;
    AttrNumber      att_answer_text;
    AttrNumber      att_content_hash;
    AttrNumber      att_proof_sig;
    Datum           d_job_id;
    Datum           d_miner;
    Datum           d_question;
    Datum           d_thinking;
    Datum           d_answer;
    Datum           d_content_hash;
    Datum           d_proof_sig;
    const char     *job_id;
    const char     *miner_addr;
    const char     *question;
    const char     *thinking;
    const char     *answer;
    bytea          *row_hash;
    bytea          *proof_sig;
    size_t          jlen, qlen, tlen, alen, total;
    uint8_t        *buf;
    uint8_t        *ptr;
    uint32_t        n;
    uint8_t         expected_hash[32];
    /* SPI block vars */
    const char     *query = "SELECT inft.inft_eth_verify($1, $2, $3)";
    Oid             argtypes[3] = { BYTEAOID, BYTEAOID, TEXTOID };
    Datum           args[3];
    char            nulls[3]   = { ' ', ' ', ' ' };
    int             spi_ret;
    bool            verify_ok;

    /* Sanity: must be called as a row-level BEFORE INSERT trigger */
    if (!CALLED_AS_TRIGGER(fcinfo))
        ereport(ERROR,
            (errcode(ERRCODE_E_R_I_E_TRIGGER_PROTOCOL_VIOLATED),
             errmsg("inft_proof_trigger_fn must be called as a trigger")));

    if (!TRIGGER_FIRED_BEFORE(tg->tg_event) ||
        !TRIGGER_FIRED_FOR_ROW(tg->tg_event) ||
        !TRIGGER_FIRED_BY_INSERT(tg->tg_event))
        ereport(ERROR,
            (errcode(ERRCODE_E_R_I_E_TRIGGER_PROTOCOL_VIOLATED),
             errmsg("inft_proof_trigger_fn: expected BEFORE INSERT FOR EACH ROW")));

    new_row = tg->tg_trigtuple;
    tupdesc = tg->tg_relation->rd_att;

    /* ── Extract field values from the new row ─────────────────────────── */
    att_job_id        = SPI_fnumber(tupdesc, "job_id");
    att_miner_address = SPI_fnumber(tupdesc, "miner_address");
    att_question_text = SPI_fnumber(tupdesc, "question_text");
    att_thinking_text = SPI_fnumber(tupdesc, "thinking_text");
    att_answer_text   = SPI_fnumber(tupdesc, "answer_text");
    att_content_hash  = SPI_fnumber(tupdesc, "content_hash");
    att_proof_sig     = SPI_fnumber(tupdesc, "proof_sig");

    d_job_id   = heap_getattr(new_row, att_job_id,        tupdesc, &isnull);
    job_id     = isnull ? "" : TextDatumGetCString(d_job_id);

    d_miner    = heap_getattr(new_row, att_miner_address, tupdesc, &isnull);
    miner_addr = isnull ? "" : TextDatumGetCString(d_miner);

    d_question = heap_getattr(new_row, att_question_text, tupdesc, &isnull);
    question   = isnull ? "" : TextDatumGetCString(d_question);

    d_thinking = heap_getattr(new_row, att_thinking_text, tupdesc, &isnull);
    thinking   = isnull ? "" : TextDatumGetCString(d_thinking);

    d_answer   = heap_getattr(new_row, att_answer_text,   tupdesc, &isnull);
    answer     = isnull ? "" : TextDatumGetCString(d_answer);

    d_content_hash = heap_getattr(new_row, att_content_hash, tupdesc, &isnull);
    row_hash       = isnull ? NULL : DatumGetByteaP(d_content_hash);

    d_proof_sig = heap_getattr(new_row, att_proof_sig, tupdesc, &isnull);
    proof_sig   = isnull ? NULL : DatumGetByteaP(d_proof_sig);

    /* ── Step 1: compute expected content_hash ─────────────────────────── */
    jlen  = strlen(job_id);
    qlen  = strlen(question);
    tlen  = strlen(thinking);
    alen  = strlen(answer);
    total = 4*4 + jlen + qlen + tlen + alen;

    buf = (uint8_t *) palloc(total);
    ptr = buf;

    n = (uint32_t)jlen; ptr[0]=(uint8_t)(n>>24); ptr[1]=(uint8_t)(n>>16); ptr[2]=(uint8_t)(n>>8); ptr[3]=(uint8_t)n; ptr+=4; if(jlen>0){memcpy(ptr,job_id,jlen); ptr+=jlen;}
    n = (uint32_t)qlen; ptr[0]=(uint8_t)(n>>24); ptr[1]=(uint8_t)(n>>16); ptr[2]=(uint8_t)(n>>8); ptr[3]=(uint8_t)n; ptr+=4; if(qlen>0){memcpy(ptr,question,qlen); ptr+=qlen;}
    n = (uint32_t)tlen; ptr[0]=(uint8_t)(n>>24); ptr[1]=(uint8_t)(n>>16); ptr[2]=(uint8_t)(n>>8); ptr[3]=(uint8_t)n; ptr+=4; if(tlen>0){memcpy(ptr,thinking,tlen); ptr+=tlen;}
    n = (uint32_t)alen; ptr[0]=(uint8_t)(n>>24); ptr[1]=(uint8_t)(n>>16); ptr[2]=(uint8_t)(n>>8); ptr[3]=(uint8_t)n; ptr+=4; if(alen>0){memcpy(ptr,answer,alen); ptr+=alen;}

    keccak256(buf, total, expected_hash);

    /* ── Step 2: compare with NEW.content_hash ─────────────────────────── */
    if (row_hash == NULL || VARSIZE_ANY_EXHDR(row_hash) != 32 ||
        memcmp(expected_hash, VARDATA_ANY(row_hash), 32) != 0)
    {
        ereport(ERROR,
            (errcode(ERRCODE_DATA_EXCEPTION),
             errmsg("inft_proof_trigger: content_hash mismatch for job_id='%s'", job_id)));
    }

    /* ── Step 3: call inft_eth_verify via SPI ──────────────────────────── */
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR,
            (errcode(ERRCODE_CONNECTION_FAILURE),
             errmsg("inft_proof_trigger: SPI_connect failed")));

    args[0] = PointerGetDatum(row_hash);
    args[1] = proof_sig ? PointerGetDatum(proof_sig) : (Datum)0;
    args[2] = CStringGetTextDatum(miner_addr);
    if (!proof_sig)
        nulls[1] = 'n';

    spi_ret   = SPI_execute_with_args(query, 3, argtypes, args, nulls, true, 1);
    verify_ok = true;

    if (spi_ret == SPI_OK_SELECT && SPI_processed == 1)
    {
        bool   is_null;
        Datum  result_d = SPI_getbinval(SPI_tuptable->vals[0],
                                        SPI_tuptable->tupdesc, 1, &is_null);
        if (!is_null)
            verify_ok = DatumGetBool(result_d);
        else
            verify_ok = false;
    }
    else
    {
        elog(WARNING, "inft_proof_trigger: inft_eth_verify query failed (ret=%d)", spi_ret);
        verify_ok = !pg_inft_require_proof_verification;
    }

    SPI_finish();

    /* ── Step 4: reject if verification failed ─────────────────────── */
    if (!verify_ok && pg_inft_require_proof_verification)
    {
        ereport(ERROR,
            (errcode(ERRCODE_DATA_EXCEPTION),
             errmsg("inft_proof_trigger: ECDSA proof verification failed for "
                    "job_id='%s' miner='%s'", job_id, miner_addr)));
    }

    /* Return NEW row unmodified */
    return PointerGetDatum(new_row);
}
