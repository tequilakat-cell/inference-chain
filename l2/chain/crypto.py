"""
Cryptographic primitives for InferenceChain.

Wraps eth_account + hashlib so every other module has a single import point.
The merkle tree implementation is a simple binary tree over keccak256 leaves —
compatible with OpenZeppelin's MerkleProof.sol for L1 fraud-proof verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from eth_account import Account
from eth_account.messages import encode_defunct


# ── Hashing ───────────────────────────────────────────────────────────────────

def keccak256(data: bytes) -> bytes:
    """Raw keccak256 — returns 32 bytes."""
    from Crypto.Hash import keccak as _keccak
    k = _keccak.new(digest_bits=256)
    k.update(data)
    return k.digest()


def keccak256_hex(data: bytes) -> str:
    """keccak256 as 0x-prefixed hex string."""
    return "0x" + keccak256(data).hex()


def hash_dict(d: dict) -> str:
    """Deterministic JSON → keccak256 hex. Keys are sorted."""
    serialised = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return keccak256_hex(serialised)


ZERO_HASH = "0x" + "00" * 32


# ── Merkle tree ───────────────────────────────────────────────────────────────

def _pair_hash(a: str, b: str) -> str:
    """Hash two leaf hashes together — sorts to match OZ MerkleProof.sol."""
    a_bytes = bytes.fromhex(a.removeprefix("0x"))
    b_bytes = bytes.fromhex(b.removeprefix("0x"))
    # Sort lexicographically so the tree is order-independent (OZ compatible)
    if a_bytes <= b_bytes:
        return keccak256_hex(a_bytes + b_bytes)
    return keccak256_hex(b_bytes + a_bytes)


def merkle_root(leaves: list[str]) -> str:
    """
    Compute merkle root from a list of 0x-prefixed hex leaf hashes.
    Odd-length layers duplicate the last element (standard approach).
    Returns ZERO_HASH for empty list.
    """
    if not leaves:
        return ZERO_HASH
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [_pair_hash(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def merkle_proof(leaves: list[str], index: int) -> list[str]:
    """Return the sibling hashes needed to prove leaves[index] is in the tree."""
    proof = []
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        sibling = index ^ 1          # XOR flips last bit → sibling index
        proof.append(layer[sibling])
        layer = [_pair_hash(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
        index //= 2
    return proof


def verify_merkle_proof(root: str, leaf: str, proof: list[str], index: int) -> bool:
    """Verify a merkle proof. Returns True if the leaf is in the tree at index."""
    current = leaf
    for sibling in proof:
        if index % 2 == 0:
            current = _pair_hash(current, sibling)
        else:
            current = _pair_hash(sibling, current)
        index //= 2
    return current == root


# ── Signing ───────────────────────────────────────────────────────────────────

def sign(private_key_hex: str, data: bytes) -> str:
    """
    Sign raw bytes with an Ethereum private key.
    Returns 0x-prefixed 65-byte hex (r, s, v format).
    """
    key = private_key_hex if private_key_hex.startswith("0x") else "0x" + private_key_hex
    msg = encode_defunct(data)
    signed = Account.sign_message(msg, private_key=key)
    return "0x" + signed.signature.hex()


def recover(data: bytes, signature_hex: str) -> str:
    """Recover the signing address from data + signature. Returns checksummed address."""
    sig = signature_hex if signature_hex.startswith("0x") else "0x" + signature_hex
    msg = encode_defunct(data)
    return Account.recover_message(msg, signature=sig)


def address_from_key(private_key_hex: str) -> str:
    """Derive the checksummed Ethereum address from a private key."""
    key = private_key_hex if private_key_hex.startswith("0x") else "0x" + private_key_hex
    return Account.from_key(key).address


def verify_sig(data: bytes, signature_hex: str, expected_address: str) -> bool:
    """Return True if signature_hex was produced by expected_address over data."""
    try:
        recovered = recover(data, signature_hex)
        return recovered.lower() == expected_address.lower()
    except Exception:
        return False


# ── RLP-lite encoding (for block hashing) ────────────────────────────────────
# We use a simplified deterministic encoding rather than full RLP to avoid
# the eth-rlp dependency for block hash computation.

def encode_block_header(header_dict: dict) -> bytes:
    """Deterministic encoding of a block header dict for hashing."""
    return json.dumps(header_dict, sort_keys=True, separators=(",", ":")).encode()


def encode_transaction(tx_dict: dict) -> bytes:
    """Deterministic encoding of a transaction dict for hashing."""
    # Exclude signature and tx_hash fields when computing the hash-preimage
    d = {k: v for k, v in tx_dict.items() if k not in ("signature", "tx_hash")}
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
