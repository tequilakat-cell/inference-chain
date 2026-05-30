"""
Model weight Merkle registry.

Computes a binary Merkle tree over a model's weights so miners can commit
to a specific checkpoint on-chain.  Later (fraud-proof phase) a challenger
can ask for any single leaf; the miner must reveal the raw bytes plus a
Merkle path, and anyone can verify both.

Leaf construction:
  GGUF / binary files — file split into CHUNK_BYTES-sized pieces; each
                         piece is keccak256-hashed to produce one leaf.
  HuggingFace models  — the safetensors weight manifest (model.safetensors
                         .index.json) maps tensor names to shard files.  We
                         sort names and keccak each name to produce leaves.
                         This is lightweight (no weight download needed) and
                         deterministic across all miners loading the same HF
                         model.
  Mock / unknown       — single leaf = keccak256(model_id.encode()).

The Merkle tree pads to the next power of 2 by duplicating the last leaf,
and uses keccak256 for internal nodes: parent = keccak(left || right).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

CHUNK_BYTES = 1 << 20   # 1 MiB per GGUF leaf


# ── Crypto primitives ─────────────────────────────────────────────────────────

def _keccak(data: bytes) -> bytes:
    from Crypto.Hash import keccak as _kmod
    k = _kmod.new(digest_bits=256)
    k.update(data)
    return k.digest()


# ── Tree helpers ──────────────────────────────────────────────────────────────

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def _pad(leaves: list[bytes]) -> list[bytes]:
    """Pad leaf list to next power-of-2 by repeating the last leaf."""
    size = _next_pow2(len(leaves))
    return leaves + [leaves[-1]] * (size - len(leaves))


def _build_layers(leaves: list[bytes]) -> list[list[bytes]]:
    layer = _pad(leaves)
    layers = [layer]
    while len(layer) > 1:
        layer = [_keccak(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
        layers.append(layer)
    return layers


# ── Public API ────────────────────────────────────────────────────────────────

def merkle_root_hex(leaves: list[bytes]) -> str:
    """Return the hex-encoded Merkle root for the given leaf hashes."""
    if not leaves:
        return "0x" + "00" * 32
    return "0x" + _build_layers(leaves)[-1][0].hex()


def merkle_proof(leaves: list[bytes], index: int) -> list[str]:
    """Return the sibling path from leaf[index] up to the root (hex strings)."""
    layers = _build_layers(leaves)
    proof, i = [], index
    for layer in layers[:-1]:
        sibling = i ^ 1
        proof.append("0x" + layer[sibling].hex())
        i //= 2
    return proof


def verify_proof(root_hex: str, leaf_hash_hex: str, index: int, proof: list[str]) -> bool:
    """Return True when the Merkle proof for leaf[index] is valid."""
    h = bytes.fromhex(leaf_hash_hex.removeprefix("0x"))
    for depth, sib_hex in enumerate(proof):
        s = bytes.fromhex(sib_hex.removeprefix("0x"))
        if (index >> depth) & 1:
            h = _keccak(s + h)
        else:
            h = _keccak(h + s)
    return ("0x" + h.hex()) == root_hex


# ── Leaf construction ─────────────────────────────────────────────────────────

def _leaves_from_file(path: Path) -> list[bytes]:
    """One keccak leaf per CHUNK_BYTES of the binary file."""
    leaves = []
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            leaves.append(_keccak(chunk))
    return leaves or [_keccak(b"")]


def _leaves_from_hf(model_id: str) -> list[bytes]:
    """
    Download only the safetensors index manifest (a few KB) and derive one
    leaf per tensor name, sorted for determinism.  Falls back to the model-ID
    leaf if the manifest is unavailable or the model uses a single-shard layout.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json

        manifest = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(manifest) as fh:
            index = json.load(fh)
        names = sorted(index.get("weight_map", {}).keys())
        if names:
            return [_keccak(n.encode()) for n in names]
    except Exception:
        pass

    # Single-shard or unavailable — try config.json for architecture info
    try:
        from huggingface_hub import hf_hub_download
        import json

        cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        # Use sorted config keys as a lightweight fingerprint
        names = sorted(f"{k}={v}" for k, v in cfg.items() if not isinstance(v, (list, dict)))
        if names:
            return [_keccak(n.encode()) for n in names]
    except Exception:
        pass

    return [_keccak(model_id.encode())]


def _leaves_mock(model_id: str) -> list[bytes]:
    return [_keccak(model_id.encode())]


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_model_root(model_id: str, model_path: str) -> tuple[str, int]:
    """
    Compute the Merkle root that commits to a specific model checkpoint.

    Args:
        model_id:   HuggingFace model ID, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
        model_path: Local file path (GGUF) or HF repo ID / empty string.

    Returns:
        (root_hex, leaf_count) — root_hex is "0x…", leaf_count >= 1.
    """
    path = Path(model_path).expanduser() if model_path else Path("")

    if model_path and path.exists() and path.is_file():
        leaves = _leaves_from_file(path)
    elif "/" in model_id:
        # Treat as a HuggingFace repo identifier
        leaves = _leaves_from_hf(model_id)
    else:
        leaves = _leaves_mock(model_id)

    root = merkle_root_hex(leaves)
    return root, len(leaves)
