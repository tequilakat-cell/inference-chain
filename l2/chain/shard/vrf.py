"""
VRF-based deterministic miner selection.

Security property: the assigned miner for each shard is unpredictable before
the parent block_hash is committed, but is deterministic and publicly verifiable
afterwards. This prevents the sequencer from grinding to self-assign.

Key invariant: parent_hash (already finalised) is used as the entropy source,
NOT the block being built.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..crypto import keccak256

log = logging.getLogger("chain.shard.vrf")


def select_miners(
    job_id:      str,
    n_shards:    int,
    block_hash:  str,            # must be the PARENT block hash, already committed
    validators:  list[tuple[str, int]],  # (address, stake_inft) — stake > 0 only
    exclude:     Optional[set[str]] = None,
    allow_reuse: bool = True,
) -> list[str]:
    """
    Return a list of n_shards miner addresses, one per shard.
    Assignment is stake-weighted: a miner with 2× the stake is 2× as likely to be selected.

    Distinct miners are preferred: each is assigned at most one shard until the
    eligible pool is exhausted. When allow_reuse is True (default) and more shards
    remain than there are distinct miners, the pool is replenished and miners may
    receive additional shards — so a requested shard count is always honoured even
    on a single-miner network. This is valid for parallel_sample / speculative /
    context_split, where one miner can run several independent shards.

    Set allow_reuse=False to cap the result at the number of distinct eligible
    miners (used by select_fallback, which must route to a *different* miner).

    Args:
        job_id:      The job identifier (UUID string).
        n_shards:    How many shards to assign.
        block_hash:  0x-prefixed parent block hash (entropy source).
        validators:  List of (address, stake) tuples for all active validators.
        exclude:     Addresses to skip (e.g. previously slashed miners for this job).
        allow_reuse: Permit a miner to take >1 shard when distinct miners run out.

    Returns:
        List of miner addresses. Length == n_shards when allow_reuse is True,
        else min(n_shards, eligible_miners).
    """
    eligible = [
        (addr.lower(), stake)
        for addr, stake in validators
        if stake > 0 and (exclude is None or addr.lower() not in {e.lower() for e in exclude})
    ]
    if not eligible:
        log.warning("vrf_no_eligible_miners job=%s", job_id)
        return []

    # Sort by address for determinism (same input → same output on every node)
    eligible.sort(key=lambda x: x[0])

    assigned: list[str] = []
    used: set[str] = set()

    bh_bytes = bytes.fromhex(block_hash.removeprefix("0x"))

    for shard_idx in range(n_shards):
        remaining = [(a, s) for a, s in eligible if a not in used]
        if not remaining:
            if not allow_reuse:
                log.warning("vrf_miners_exhausted job=%s shard=%d", job_id, shard_idx)
                break
            # Replenish the pool: every miner becomes eligible again so the
            # requested shard count is honoured even with fewer distinct miners.
            used.clear()
            remaining = list(eligible)

        remaining_stake = sum(s for _, s in remaining)

        seed = keccak256(
            job_id.encode("utf-8")
            + shard_idx.to_bytes(4, "big")
            + bh_bytes
        )
        target = int.from_bytes(seed, "big") % remaining_stake

        cumulative = 0
        chosen = remaining[-1][0]   # fallback
        for addr, stake in remaining:
            cumulative += stake
            if cumulative > target:
                chosen = addr
                break

        assigned.append(chosen)
        used.add(chosen)

    log.debug(
        "vrf_assigned job=%s n_shards=%d block_hash=%s miners=%s",
        job_id, n_shards, block_hash[:12], assigned,
    )
    return assigned


def select_fallback(
    job_id:          str,
    shard_idx:       int,
    block_hash:      str,
    validators:      list[tuple[str, int]],
    previously_used: set[str],
    attempt:         int = 1,
) -> Optional[str]:
    """
    Select a replacement miner for a timed-out shard.
    Uses attempt number as extra entropy so repeated timeouts yield different fallbacks.
    """
    result = select_miners(
        job_id=job_id,
        n_shards=1,
        block_hash=block_hash,
        validators=[(a, s) for a, s in validators],
        exclude=previously_used,
        allow_reuse=False,   # a fallback must be a *different* miner
    )
    return result[0] if result else None
