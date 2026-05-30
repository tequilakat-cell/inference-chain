"""
Extracts <think>...</think> blocks from raw model output and builds
Ethereum ECDSA proof signatures for ingest into pg_inft.

Two top-level responsibilities:
  1. Parse model output to separate "thinking" from "answer" text.
  2. Build a cryptographic proof (content_hash + eth personal-sign) that
     ties the inference output to the miner's Ethereum key pair.
"""

from __future__ import annotations

import re
import struct
import logging

log = logging.getLogger("thought_extractor")

# ── Thinking block extraction ─────────────────────────────────────────────────

# Match both <think>...</think> and <thinking>...</thinking> (non-greedy, DOTALL)
_THINK_RE = re.compile(
    r"<think(?:ing)?>(.*?)</think(?:ing)?>",
    re.DOTALL | re.IGNORECASE,
)

# Stray </think> or </thinking> prefix that some models leave in the answer
_STRAY_CLOSE_RE = re.compile(
    r"^\s*</think(?:ing)?>\s*",
    re.IGNORECASE,
)


def extract_thinking(raw_output: str) -> tuple[str, str]:
    """
    Extract thinking and answer text from raw model output.

    Returns (thinking_text, answer_text).

    If <think>...</think> or <thinking>...</thinking> tags are present:
      - thinking_text = concatenated contents of all matched blocks (stripped).
      - answer_text   = remainder of the string after removing the blocks,
                        with any leading </think> / </thinking> prefix stripped.

    If no tags are found:
      - thinking_text = ''
      - answer_text   = raw_output.strip()
    """
    matches = list(_THINK_RE.finditer(raw_output))

    if not matches:
        return "", raw_output.strip()

    # Collect all thinking fragments
    thinking_parts = []
    for m in matches:
        fragment = m.group(1).strip()
        if fragment:
            thinking_parts.append(fragment)
    thinking_text = "\n\n".join(thinking_parts)

    # Remove all matched blocks from the output to get the answer
    answer_raw = _THINK_RE.sub("", raw_output)

    # Strip stray closing tag that some models emit before the answer
    answer_raw = _STRAY_CLOSE_RE.sub("", answer_raw)

    answer_text = answer_raw.strip()

    return thinking_text, answer_text


# ── Content hash (pure Python, mirrors the C implementation) ─────────────────

def _keccak256(data: bytes) -> bytes:
    """
    Compute Keccak-256 (domain separation 0x01, NOT SHA3-256).

    Tries pysha3 first, then eth_hash[pycryptodome], then sha3 from hashlib
    (which on CPython ≥ 3.6 can produce Keccak-256 when the OpenSSL build
    includes it — but that is not guaranteed, so pysha3 is preferred).
    """
    try:
        import sha3  # pysha3 — provides hashlib.sha3_256 as Keccak-256
        import hashlib
        h = hashlib.new("keccak_256")
        h.update(data)
        return h.digest()
    except (ImportError, ValueError):
        pass

    try:
        from eth_hash.auto import keccak  # eth_hash with pycryptodome backend
        return keccak(data)
    except ImportError:
        pass

    try:
        # Fallback: eth_account ships its own keccak implementation
        from eth_account._utils.legacy_transactions import keccak as acc_keccak  # type: ignore
        return acc_keccak(primitive=data)
    except Exception:
        pass

    raise ImportError(
        "No Keccak-256 implementation available. "
        "Install pysha3 (`pip install pysha3`) or eth-hash[pycryptodome]."
    )


def build_content_hash(
    job_id: str,
    question: str,
    thinking: str,
    answer: str,
) -> bytes:
    """
    Compute the content hash for an inference record.

    Encoding:
      keccak256(
          len4(job_id)   || job_id   ||
          len4(question) || question ||
          len4(thinking) || thinking ||
          len4(answer)   || answer
      )

    len4(s) is a 4-byte big-endian uint32 giving the UTF-8 byte length of s.

    This mirrors the C function `inft_content_hash` in inft_proof.c exactly.
    """
    fields = [
        job_id.encode("utf-8"),
        question.encode("utf-8"),
        thinking.encode("utf-8"),
        answer.encode("utf-8"),
    ]

    parts = []
    for f in fields:
        parts.append(struct.pack(">I", len(f)))  # 4-byte big-endian length
        parts.append(f)

    payload = b"".join(parts)
    return _keccak256(payload)


# ── Proof building ────────────────────────────────────────────────────────────

def build_proof(
    job_id: str,
    question: str,
    thinking: str,
    answer: str,
    private_key_hex: str,
) -> str:
    """
    Build an Ethereum personal-sign ECDSA proof for the inference record.

    Steps:
      1. Compute content_hash = build_content_hash(job_id, question, thinking, answer)
      2. eth_account personal-sign the 32-byte content_hash with the miner's key.
      3. Return the 65-byte signature (r32 + s32 + v1) as a 0x-prefixed hex string.

    The resulting signature can be verified on-chain or via inft_eth_verify().

    private_key_hex may be with or without the "0x" prefix.
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError as exc:
        raise ImportError(
            "eth_account is required for proof building. "
            "Install it with: pip install eth-account"
        ) from exc

    content_hash = build_content_hash(job_id, question, thinking, answer)

    # encode_defunct wraps the hash with the Ethereum personal-sign prefix:
    # "\x19Ethereum Signed Message:\n32"
    message = encode_defunct(primitive=content_hash)

    # Normalise private key (eth_account accepts with or without 0x)
    key = private_key_hex.strip()
    if not key.startswith("0x"):
        key = "0x" + key

    signed = Account.sign_message(message, private_key=key)

    # signed.signature is a HexBytes of length 65: r(32) + s(32) + v(1)
    sig_bytes = bytes(signed.signature)
    assert len(sig_bytes) == 65, f"Unexpected signature length: {len(sig_bytes)}"

    return "0x" + sig_bytes.hex()
