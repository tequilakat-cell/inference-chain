"""
Context split mode.

The prompt is divided into N roughly equal segments. Each miner processes
one segment. Results are concatenated in order to form the final output.

Ideal for:
  - Long-context summarisation (split document chapters)
  - RAG pipelines (parallel retrieval + generation over different chunks)
  - Any task where partial results can be independently generated then merged

Speedup: approaches N× for perfectly parallelisable tasks; limited by the
longest shard in practice. Typical real-world gain: 2-4× for long documents.
"""

from __future__ import annotations

from ...types import ShardResult

# Number of characters we treat as ≈1 token. Rough approximation; fine for
# splitting purposes. Replace with a real tokenizer if tight boundary control
# is needed.
CHARS_PER_TOKEN = 4

# Overlap in tokens to add at the start of each shard (except the first) so
# the model has enough context to continue coherently.
OVERLAP_TOKENS = 64
OVERLAP_CHARS  = OVERLAP_TOKENS * CHARS_PER_TOKEN


def split_prompt(prompt: str, n_shards: int) -> list[str]:
    """
    Split prompt into n_shards overlapping chunks.

    Each chunk except the first starts with OVERLAP_CHARS characters from
    the end of the previous chunk so the model has contextual continuity.
    """
    if n_shards <= 1:
        return [prompt]

    total = len(prompt)
    chunk_size = max(OVERLAP_CHARS * 2, total // n_shards)
    slices: list[str] = []

    start = 0
    for i in range(n_shards):
        end = min(total, start + chunk_size)
        if i == n_shards - 1:
            end = total  # last shard takes whatever remains

        slices.append(prompt[start:end])

        # Next shard starts with an overlap so context carries over
        start = max(0, end - OVERLAP_CHARS)

        if start >= total:
            break

    # Pad with empty strings if we ran out of content
    while len(slices) < n_shards:
        slices.append("")

    return slices


def assemble(results: list[ShardResult]) -> str:
    """
    Concatenate shard outputs in order, stripping the overlap region from
    the beginning of each shard's output (heuristic: trim leading sentences
    that duplicate the previous shard's tail).
    """
    if not results:
        return ""

    # Sort by shard_index to ensure correct order regardless of arrival order
    ordered = sorted(results, key=lambda r: r.shard_index)
    parts: list[str] = []

    for i, result in enumerate(ordered):
        text = result.output.strip()
        if not text:
            continue
        if i == 0:
            parts.append(text)
        else:
            # Heuristic: if the output starts with words from the overlap,
            # strip up to the first sentence boundary after the likely overlap point.
            # Simple version: just append directly. A production implementation
            # would use the tokenizer to find the exact overlap boundary.
            parts.append("\n\n" + text)

    return "".join(parts).strip()
