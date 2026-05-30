/*
 * pg_inft.h — shared declarations for the pg_inft PostgreSQL extension.
 *
 * This header is included by every C translation unit in the extension.
 * It pulls in the minimal PostgreSQL headers needed and declares the GUC
 * variables that are shared across files.
 */

#ifndef PG_INFT_H
#define PG_INFT_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "miscadmin.h"

/* ── GUC variables (defined in pg_inft.c) ─────────────────────────────────── */

/* Minimum composite BM25+trigram score for inft_search results. */
extern double  pg_inft_min_similarity;

/* Maximum context token budget injected into prompts. */
extern int     pg_inft_max_context_tokens;

/* When true, the proof trigger rejects rows with invalid ECDSA signatures. */
extern bool    pg_inft_require_proof_verification;

/* When true, inft_search only returns rows matching the exact model_id. */
extern bool    pg_inft_model_scope_strict;

/* ── Extension version ─────────────────────────────────────────────────────── */
#define PG_INFT_VERSION "1.0"

/* ── Function prototypes ────────────────────────────────────────────────────── */

/* Each source file declares its own PG_FUNCTION_INFO_V1 locally.
 * Only prototypes are listed here so other files can call them if needed. */
extern Datum inft_keccak256(PG_FUNCTION_ARGS);
extern Datum inft_content_hash(PG_FUNCTION_ARGS);
extern Datum inft_eth_personal_hash(PG_FUNCTION_ARGS);
extern Datum inft_proof_trigger_fn(PG_FUNCTION_ARGS);
extern Datum inft_search(PG_FUNCTION_ARGS);
extern Datum inft_chunk_text(PG_FUNCTION_ARGS);
extern Datum pg_inft_version(PG_FUNCTION_ARGS);

#endif /* PG_INFT_H */
