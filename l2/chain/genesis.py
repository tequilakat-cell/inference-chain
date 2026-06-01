"""
Genesis block construction.

Reads genesis.json, allocates initial balances and stakes, and produces
block 0 + the initial StateDB. Called once on chain startup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .types import Block, BlockHeader, Transaction
from .state import StateDB, AccountState
from .crypto import keccak256_hex, ZERO_HASH, hash_dict


def load_config(genesis_path: str = "genesis.json") -> dict:
    p = Path(genesis_path)
    if not p.exists():
        raise FileNotFoundError(f"genesis.json not found at {genesis_path}")
    with open(p) as f:
        return json.load(f)


def build_genesis(config: dict) -> tuple[Block, StateDB]:
    """
    Build the genesis block and initial state from the genesis config dict.
    Returns (block_0, state_db).
    """
    state = StateDB()
    state.block_number = 0

    # Allocate initial balances
    for entry in config.get("initial_balances", []):
        addr = entry["address"].lower()
        state.set_account(addr, AccountState(
            balance_inft=int(entry["balance_inft"]),
            stake_inft=0,
            nonce=0,
            reputation=500,
        ))

    # Allocate initial stakes for validators
    for entry in config.get("initial_validators", []):
        addr = entry["address"].lower()
        acc = state.account(addr)
        state.set_account(addr, AccountState(
            balance_inft=acc.balance_inft,
            stake_inft=int(entry["stake_inft"]),
            nonce=acc.nonce,
            reputation=500,
        ))

    state_root = state.state_root()

    header = BlockHeader(
        block_number=0,
        parent_hash=ZERO_HASH,
        timestamp=int(time.time() * 1000),
        sequencer=config.get("sequencer_address", "0x0000000000000000000000000000000000000000"),
        tx_root=ZERO_HASH,
        state_root=state_root,
        shard_root=ZERO_HASH,
        gas_used=0,
        extra_data=config.get("chain_name", "InferenceChain"),
    )

    block_hash = keccak256_hex(
        json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    )

    block = Block(
        header=header,
        transactions=tuple(),
        sequencer_sig="",
        block_hash=block_hash,
    )

    return block, state


CHAIN_DEFAULTS = {
    "chain_id":               2026,
    "chain_name":             "InferenceChain",
    "block_time_ms":          1_000,
    "state_root_interval":    100,
    "fraud_proof_window_s":   604_800,   # 7 days
    "shard_offer_timeout_ms": 120_000,
    "shard_result_timeout_ms":120_000,
    "assembly_timeout_ms":    130_000,
    "min_stake_inft":         100,
    "slash_pct_offer":        10,
    "slash_pct_result":       30,
    "max_shards_per_job":     8,
    "max_concurrent_shards":  4,         # per miner
    # Context load pre-phase (Phase 3 / Option B parallel)
    "context_load_timeout_ms": 8_000,    # how long to wait for all ContextLoadResults
    "context_load_enabled":    True,     # set False to disable the pre-phase entirely
    "kv_cache_dir":            "/tmp/inft_kv",  # where miners store prompt-cache files (Phase 4)
    "kv_cache_ttl_s":          3600,     # evict KV cache files older than this
    # Peerbit sidecar URL (optional; set in genesis.json or PEERBIT_URL env var)
    "peerbit_url": None,    # e.g. "http://127.0.0.1:7731"
    # Miner hardware benchmark (Phase 5)
    # Sequencer measures wall-clock time from challenge send → response received.
    # Score = benchmark_n_tokens / elapsed_s  (tokens/sec).
    # Miners with no valid score get min_layer_frac of pipeline layers.
    "benchmark_prompt":
        "Explain the difference between supervised and unsupervised learning in machine learning, with two examples of each.",
    "benchmark_n_tokens":        64,       # tokens the miner must generate
    "benchmark_validity_blocks": 5760,     # score expires after N blocks (~1 day at 1 s/block)
    "benchmark_timeout_s":       120.0,    # seconds sequencer waits for miner response
    "benchmark_required":        False,    # True = exclude unscored miners from pipeline jobs
    "min_layer_frac":            0.05,     # minimum fraction of layers any single miner gets
    "max_layer_frac":            0.80,     # maximum fraction of layers any single miner gets
}
