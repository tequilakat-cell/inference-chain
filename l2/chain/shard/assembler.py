"""
Result assembler — mode-dispatch router.

Accepts a list of ShardResults for a completed job and calls the appropriate
mode-specific assembler to produce the final output string + its keccak hash.
"""

from __future__ import annotations

import logging

from ..types import ShardResult, ShardMode
from ..crypto import keccak256_hex
from .modes import parallel_sample, context_split, speculative
from .modes import tensor_parallel as tensor_parallel_mode
from .modes import pipeline_parallel as pipeline_parallel_mode

log = logging.getLogger("chain.shard.assembler")


def assemble(
    mode:    str,
    results: list[ShardResult],
    strategy: str = "first",   # only used by parallel_sample
) -> tuple[str, str]:
    """
    Assemble shard results into a final output.

    Returns:
        (final_output: str, output_hash: str)  — output_hash is 0x-prefixed keccak256
    """
    if not results:
        return "", keccak256_hex(b"")

    if mode == ShardMode.PARALLEL_SAMPLE:
        output = parallel_sample.assemble(results, strategy=strategy)

    elif mode == ShardMode.CONTEXT_SPLIT:
        output = context_split.assemble(results)

    elif mode == ShardMode.SPECULATIVE:
        output = speculative.assemble_from_results(results)

    elif mode in (ShardMode.TENSOR_PARALLEL, ShardMode.PIPELINE_PARALLEL):
        output = pipeline_parallel_mode.assemble(results)

    else:
        log.warning("assembler: unknown mode %r, using first result", mode)
        output = results[0].output.strip()

    output_hash = keccak256_hex(output.encode("utf-8"))
    log.info(
        "assembled mode=%s n_shards=%d output_len=%d hash=%s",
        mode, len(results), len(output), output_hash[:16],
    )
    return output, output_hash
