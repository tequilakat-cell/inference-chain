/*
 * pg_inft.c — extension entry point, GUC registration, version function.
 *
 * Implements _PG_init() which registers all four GUC variables that control
 * the extension's runtime behaviour.  Each GUC is session-local (no restart
 * required) so operators can tune them per connection or per transaction.
 */

#include "pg_inft.h"

PG_MODULE_MAGIC;

/* ── GUC storage ─────────────────────────────────────────────────────────── */

double  pg_inft_min_similarity          = 0.25;
int     pg_inft_max_context_tokens      = 512;
bool    pg_inft_require_proof_verification = true;
bool    pg_inft_model_scope_strict      = false;

/* ── _PG_init ─────────────────────────────────────────────────────────────── */

void _PG_init(void);

void
_PG_init(void)
{
    DefineCustomRealVariable(
        "pg_inft.min_similarity",
        "Minimum composite BM25+trigram score for inft_search() results.",
        "Results with a combined score below this threshold are excluded. "
        "Range [0.0, 1.0]. Default 0.25.",
        &pg_inft_min_similarity,
        0.25,           /* boot_val  */
        0.0,            /* min_val   */
        1.0,            /* max_val   */
        PGC_USERSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomIntVariable(
        "pg_inft.max_context_tokens",
        "Maximum context tokens injected into inference prompts.",
        "Limits the total size of prior-thinking context prepended to prompts. "
        "Default 512.",
        &pg_inft_max_context_tokens,
        512,            /* boot_val  */
        0,              /* min_val   */
        65536,          /* max_val   */
        PGC_USERSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomBoolVariable(
        "pg_inft.require_proof_verification",
        "Reject inft_thought_log inserts whose ECDSA proof fails verification.",
        "When on (default), the proof trigger calls inft_eth_verify() and rejects "
        "rows whose signature does not recover to miner_address.  Set to off to "
        "allow inserts without a valid eth_account library.",
        &pg_inft_require_proof_verification,
        true,
        PGC_USERSET,
        0,
        NULL, NULL, NULL
    );

    DefineCustomBoolVariable(
        "pg_inft.model_scope_strict",
        "Restrict inft_search() to exact model_id matches when model_id is non-empty.",
        "When off (default), an empty model_id string searches across all models. "
        "When on, inft_search() always filters by the provided model_id even if it "
        "is non-empty but does not match any stored row.",
        &pg_inft_model_scope_strict,
        false,
        PGC_USERSET,
        0,
        NULL, NULL, NULL
    );
}

/* ── pg_inft_version() → text ─────────────────────────────────────────────── */

PG_FUNCTION_INFO_V1(pg_inft_version);
Datum
pg_inft_version(PG_FUNCTION_ARGS)
{
    PG_RETURN_TEXT_P(cstring_to_text(PG_INFT_VERSION));
}
