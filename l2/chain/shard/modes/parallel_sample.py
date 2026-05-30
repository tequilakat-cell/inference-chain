"""
Parallel sample mode.

All N miners receive the FULL prompt simultaneously and race to respond.
The sequencer selects the winner based on strategy:
  - "first":  whoever submits first wins (minimises p99 latency)
  - "vote":   majority-vote over outputs (maximises consistency)

Typical speedup: reduces worst-case latency from max(t_i) to min(t_i).
For heterogeneous miners (fast CPU + slow CPU) this can be 2-5x.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ...types import ShardResult


def split_prompt(prompt: str, n_shards: int) -> list[str]:
    """In parallel_sample mode, every shard gets the full prompt."""
    return [prompt] * n_shards


def assemble(
    results:  list[ShardResult],
    strategy: str = "first",
) -> str:
    """
    Combine shard results into a final output.

    Args:
        results:  Non-empty list of ShardResult, ordered by submission time.
        strategy: "first" | "vote"
    """
    if not results:
        return ""

    if strategy == "first":
        # Already ordered by submission time; pick the fastest
        return results[0].output.strip()

    if strategy == "vote":
        # Majority vote — strip whitespace before comparing
        votes = [r.output.strip() for r in results]
        most_common = Counter(votes).most_common(1)[0][0]
        return most_common

    return results[0].output.strip()
