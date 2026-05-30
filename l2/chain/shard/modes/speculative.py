"""
Speculative decoding mode.

Shard 0 — the "draft miner" — uses a small, fast model to generate candidate
tokens quickly (e.g. Qwen2.5-0.5B at 50 tok/s).

Shard 1 — the "verifier miner" — runs the full target model but only needs a
single parallel forward pass to accept/reject each draft token rather than
autoregressive generation. Rejected tokens are corrected from the verifier's
distribution.

Expected throughput gain: 2.5-4× vs single-model generation, matching the
acceptance rate of the draft model (typically 70-90% for similar-family models).

Protocol:
  1. Draft miner generates tokens greedily and P2P-broadcasts them as they arrive
     (streaming, one chunk per DRAFT_CHUNK_TOKENS tokens).
  2. Verifier miner receives draft chunks via P2P "speculative_draft" topic,
     runs a batched verify pass, and returns an acceptance mask + corrections.
  3. Sequencer assembles the final output from accepted draft tokens + corrections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

from ...types import ShardResult

log = logging.getLogger("chain.shard.speculative")

DRAFT_CHUNK_TOKENS = 16   # How many draft tokens to send per P2P chunk


@dataclass
class DraftChunk:
    """Streaming draft tokens from the draft miner to the verifier."""
    job_id:      str
    shard_index: int           # always 0
    chunk_index: int           # 0, 1, 2, … for streaming chunks
    tokens:      list[str]     # token strings (not ids)
    is_final:    bool = False  # True on last chunk
    miner:       str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifyResult:
    """Verifier miner's response to a set of draft tokens."""
    job_id:      str
    accepted:    list[bool]    # accepted[i] = True if draft token i is accepted
    corrections: list[str]     # replacement tokens where accepted[i] = False
    miner:       str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def split_prompt(prompt: str, n_shards: int) -> list[str]:
    """
    Both miners receive the full prompt.
    Shard 0 = draft miner, shard 1 = verifier miner.
    """
    assert n_shards == 2, "speculative mode requires exactly 2 shards"
    return [prompt, prompt]


def assemble(draft_result: ShardResult, verify_result: ShardResult) -> str:
    """
    Merge draft and verifier outputs into the final decoded sequence.

    In a full implementation this would use the token-level acceptance mask
    from VerifyResult. Here we use a text-level heuristic: prefer the
    verifier's output (which has had the full model validate it) but fall
    back to the draft if the verifier's output is empty.
    """
    verifier_text = verify_result.output.strip()
    draft_text    = draft_result.output.strip()

    if verifier_text:
        return verifier_text

    log.warning("speculative_assemble: verifier output empty, using draft")
    return draft_text


def assemble_from_results(results: list[ShardResult]) -> str:
    """Dispatcher called by assembler.py — expects exactly 2 results."""
    if len(results) < 2:
        return results[0].output.strip() if results else ""

    # shard 0 = draft, shard 1 = verify
    ordered = sorted(results, key=lambda r: r.shard_index)
    return assemble(draft_result=ordered[0], verify_result=ordered[1])
