"""
Block validator.

Stateless structural validation used by both the sequencer (self-check)
and peers when they receive a gossiped block.
"""

from __future__ import annotations

import json
import logging

from .types import Block
from .crypto import (
    keccak256_hex, verify_sig, merkle_root, encode_block_header, ZERO_HASH
)

log = logging.getLogger("chain.block_validator")


def validate_block(
    block:             Block,
    parent:            Block,
    expected_sequencer: str,
    chain_id:          int,
) -> tuple[bool, str]:
    """
    Returns (valid, reason). Checks structure, sequencer sig, and root consistency.
    Does NOT re-execute transactions (that is the fraud-proof's job for L1).
    """
    h = block.header

    # Block number must be exactly parent + 1
    if h.block_number != parent.header.block_number + 1:
        return False, f"block_number gap: {h.block_number} vs parent {parent.header.block_number}"

    # Parent hash must match
    if h.parent_hash != parent.block_hash:
        return False, f"parent_hash mismatch"

    # Timestamp must be ≥ parent timestamp
    if h.timestamp < parent.header.timestamp:
        return False, f"timestamp regression"

    # Sequencer must match configured sequencer
    if h.sequencer.lower() != expected_sequencer.lower():
        return False, f"wrong sequencer: {h.sequencer}"

    # Block hash must match header hash
    expected_hash = keccak256_hex(encode_block_header(h.to_dict()))
    if block.block_hash != expected_hash:
        return False, f"block_hash mismatch: got {block.block_hash[:16]}, expected {expected_hash[:16]}"

    # Sequencer signature must be valid over block_hash
    block_hash_bytes = bytes.fromhex(block.block_hash.removeprefix("0x"))
    if not verify_sig(block_hash_bytes, block.sequencer_sig, expected_sequencer):
        return False, "invalid sequencer signature"

    # tx_root must match transactions
    from .block_builder import _tx_merkle_root
    computed_tx_root = _tx_merkle_root(list(block.transactions))
    if h.tx_root != computed_tx_root:
        return False, f"tx_root mismatch"

    return True, ""
