"""
Rollup poster — commits L2 state roots to L1 every state_root_interval blocks.

Posts a StateRootCommitment to InferenceChainRollup.sol via web3.py.
Tracks the last posted block in a SQLite DB so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

from .types import Block, StateRootCommitment
from .crypto import keccak256_hex, sign, address_from_key
from .genesis import CHAIN_DEFAULTS

log = logging.getLogger("chain.rollup_poster")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS commitments (
    l2_block_number INTEGER PRIMARY KEY,
    state_root      TEXT NOT NULL,
    tx_batch_hash   TEXT NOT NULL,
    l1_tx_hash      TEXT,
    committed_at    INTEGER,
    finalized       INTEGER DEFAULT 0
);
"""


class RollupPoster:
    def __init__(
        self,
        l1_rpc_url:       str,
        rollup_address:   str,
        rollup_abi:       list,
        sequencer_privkey: str,
        cfg:              dict = None,
        db_path:          str  = "rollup.db",
    ):
        self._cfg       = {**CHAIN_DEFAULTS, **(cfg or {})}
        self._interval  = self._cfg["state_root_interval"]
        self._privkey   = sequencer_privkey
        self._address   = address_from_key(sequencer_privkey)
        self._db_path   = db_path

        w3 = Web3(Web3.HTTPProvider(l1_rpc_url, request_kwargs={"timeout": 30}))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._w3       = w3
        self._contract = w3.eth.contract(
            address=Web3.to_checksum_address(rollup_address),
            abi=rollup_abi,
        )
        self._account  = Account.from_key(sequencer_privkey)

        self._init_db()
        log.info(
            "rollup_poster_ready address=%s rollup=%s interval=%d",
            self._address, rollup_address, self._interval,
        )

    def _init_db(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.execute(CREATE_TABLE)
        con.commit()
        con.close()

    async def maybe_post(self, block: Block, state) -> None:
        """Called by sequencer after each block. Posts when interval is reached."""
        if block.header.block_number % self._interval != 0:
            return
        if block.header.block_number == 0:
            return

        already = self._last_posted_block()
        if already >= block.header.block_number:
            return

        await self._post_commitment(block, state)

    def _last_posted_block(self) -> int:
        con = sqlite3.connect(self._db_path)
        row = con.execute(
            "SELECT MAX(l2_block_number) FROM commitments WHERE l1_tx_hash IS NOT NULL"
        ).fetchone()
        con.close()
        return row[0] or 0

    async def _post_commitment(self, block: Block, state) -> None:
        l2_num       = block.header.block_number
        state_root   = block.header.state_root
        tx_batch_hash = keccak256_hex(block.block_hash.encode())

        preimage = (str(l2_num) + state_root + tx_batch_hash).encode()
        seq_sig  = sign(self._privkey, preimage)

        log.info("posting_state_root l2_block=%d state_root=%s", l2_num, state_root[:16])

        import asyncio
        loop = asyncio.get_event_loop()
        try:
            tx_hash = await loop.run_in_executor(None, lambda: self._send_commitment(
                l2_num, state_root, tx_batch_hash, seq_sig
            ))
            self._record_commitment(l2_num, state_root, tx_batch_hash, tx_hash)
            log.info("state_root_committed l2_block=%d l1_tx=%s", l2_num, tx_hash[:16])
        except Exception as exc:
            log.error("rollup_post_failed l2_block=%d err=%s", l2_num, exc, exc_info=True)

    def _send_commitment(
        self,
        l2_block:     int,
        state_root:   str,
        tx_batch_hash: str,
        seq_sig:      str,
    ) -> str:
        nonce    = self._w3.eth.get_transaction_count(self._account.address, "pending")
        base_fee = self._w3.eth.get_block("latest").get("baseFeePerGas", self._w3.eth.gas_price)

        tx = self._contract.functions.commitStateRoot(
            l2_block,
            bytes.fromhex(state_root.removeprefix("0x")),
            bytes.fromhex(tx_batch_hash.removeprefix("0x")),
            bytes.fromhex(seq_sig.removeprefix("0x")),
        ).build_transaction({
            "from":                 self._account.address,
            "nonce":                nonce,
            "gas":                  200_000,
            "maxFeePerGas":         base_fee * 2,
            "maxPriorityFeePerGas": self._w3.to_wei("0.01", "gwei"),
            "type":                 "0x2",
        })

        signed  = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"commitStateRoot tx reverted: {tx_hash.hex()}")
        return tx_hash.hex()

    def _record_commitment(
        self, l2_num: int, state_root: str, tx_batch_hash: str, l1_tx_hash: str
    ) -> None:
        con = sqlite3.connect(self._db_path)
        con.execute(
            "INSERT OR REPLACE INTO commitments "
            "(l2_block_number, state_root, tx_batch_hash, l1_tx_hash, committed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (l2_num, state_root, tx_batch_hash, l1_tx_hash, int(time.time())),
        )
        con.commit()
        con.close()
