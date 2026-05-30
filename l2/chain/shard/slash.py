"""
Slash logic for non-responsive shard miners.

Two tiers:
  SLASH      (TxType.SLASH)      — miner accepted offer but offer window expired:  10% stake
  SLASH_HARD (TxType.SLASH_HARD) — miner accepted but never returned result:       30% stake

Slash transactions are synthesised by the sequencer (no user signature required)
and injected directly into the mempool.
"""

from __future__ import annotations

import json
import logging
import time

from ..types import TxType, Transaction
from ..crypto import keccak256_hex

log = logging.getLogger("chain.shard.slash")


def build_slash_tx(
    miner:      str,
    job_id:     str,
    shard_idx:  int,
    chain_id:   int,
    hard:       bool = False,
) -> Transaction:
    """
    Build a sequencer-synthesized slash transaction.
    """
    tx_type = TxType.SLASH_HARD if hard else TxType.SLASH
    payload = json.dumps({
        "miner":       miner,
        "job_id":      job_id,
        "shard_index": shard_idx,
        "reason":      "result_timeout" if hard else "offer_timeout",
        "ts":          int(time.time()),
    }, separators=(",", ":"))

    tx_hash = keccak256_hex(
        (str(tx_type) + miner + job_id + str(shard_idx) + payload).encode()
    )

    log.info(
        "slash_tx_built miner=%s job=%s shard=%d hard=%s",
        miner[:10], job_id[:8], shard_idx, hard,
    )

    return Transaction(
        tx_type=tx_type,
        sender="",        # sequencer-synthesized: no sender
        nonce=0,
        payload=payload,
        gas_price=0,
        signature="",
        tx_hash=tx_hash,
    )
