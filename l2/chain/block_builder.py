"""
Block builder.

Called by the sequencer every block_time_ms to assemble the next block
from pending mempool transactions. Computes tx_root, state_root, shard_root
and produces a signed BlockHeader + Block ready for broadcast.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from .types import Block, BlockHeader, Transaction
from .state import StateDB
from .mempool import Mempool
from .crypto import keccak256_hex, merkle_root, ZERO_HASH, encode_block_header, sign

log = logging.getLogger("chain.block_builder")


class BlockBuilder:
    def __init__(
        self,
        sequencer_address:  str,
        sequencer_privkey:  str,
        chain_id:           int,
        max_txs_per_block:  int = 500,
    ):
        self.sequencer_address = sequencer_address
        self._privkey          = sequencer_privkey
        self.chain_id          = chain_id
        self.max_txs           = max_txs_per_block

    async def build_next(
        self,
        parent:  Block,
        state:   StateDB,
        mempool: Mempool,
    ) -> tuple[Block, StateDB]:
        """
        Pop transactions from the mempool, apply them, and produce the next block.
        Returns (new_block, new_state).
        """
        txs = await mempool.pop_batch(self.max_txs)

        # Apply transactions to a working copy of state
        new_state = state.copy()
        new_state.block_number = parent.header.block_number + 1
        accepted: list[Transaction] = []

        for tx in txs:
            try:
                new_state = new_state.apply_transaction(tx)
                accepted.append(tx)
            except Exception as exc:
                log.debug("tx_rejected tx=%s reason=%s", tx.tx_hash[:12], exc)
                # Bad tx is silently dropped — not re-queued

        # Compute roots
        tx_root    = _tx_merkle_root(accepted)
        state_root = new_state.state_root()
        shard_root = new_state.shard_root()

        header = BlockHeader(
            block_number=new_state.block_number,
            parent_hash=parent.block_hash,
            timestamp=int(time.time() * 1000),
            sequencer=self.sequencer_address,
            tx_root=tx_root,
            state_root=state_root,
            shard_root=shard_root,
            gas_used=len(accepted),
            extra_data=f"v1 chain_id={self.chain_id}",
        )

        block_hash = keccak256_hex(
            encode_block_header(header.to_dict())
        )
        sequencer_sig = sign(self._privkey, bytes.fromhex(block_hash.removeprefix("0x")))

        block = Block(
            header=header,
            transactions=tuple(accepted),
            sequencer_sig=sequencer_sig,
            block_hash=block_hash,
        )

        log.info(
            "block_built number=%d txs=%d state_root=%s",
            header.block_number, len(accepted), state_root[:16],
        )
        return block, new_state


def _tx_merkle_root(txs: list[Transaction]) -> str:
    if not txs:
        return ZERO_HASH
    leaves = [tx.tx_hash for tx in txs]
    return merkle_root(leaves)
