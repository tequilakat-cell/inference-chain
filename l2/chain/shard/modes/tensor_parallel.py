"""
Tensor-parallel assembler.

In TENSOR_PARALLEL mode each pipeline stage emits a ShardResult:
  - Stages 0 .. N-2: output = "[tp_stage:{i}:{activation_hash}]"  (placeholder, proves work done)
  - Stage N-1       : output = the actual decoded text

The final assembled output is simply the last stage's text.
Intermediate stage results are kept for auditing / slash proofs.
"""

from __future__ import annotations

from ...types import ShardResult


def assemble(results: list[ShardResult]) -> str:
    """Return the decoded text produced by the final pipeline stage."""
    if not results:
        return ""
    # Results are sorted by shard_index in assembler.py before this call.
    final = results[-1]
    output = final.output.strip()
    # Strip the placeholder prefix in case the last stage accidentally used it.
    if output.startswith("[tp_stage:"):
        return ""
    return output
