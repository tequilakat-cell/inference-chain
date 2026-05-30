"""
Bridge watcher — monitors L1 for DepositInitiated events and mints on L2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

log = logging.getLogger("bridge.watcher")

CREATE = """
CREATE TABLE IF NOT EXISTS deposits (
    nonce       INTEGER PRIMARY KEY,
    l1_tx_hash  TEXT,
    l2_tx_hash  TEXT,
    amount      TEXT,
    l1_sender   TEXT,
    l2_recipient TEXT,
    relayed     INTEGER DEFAULT 0
);
"""


class BridgeWatcher:
    def __init__(
        self,
        l1_rpc_url:     str,
        bridge_address: str,
        bridge_abi:     list,
        l2_rpc_url:     str,
        sequencer_privkey: str,
        l2_chain_id:    int,
        db_path:        str = "bridge.db",
        poll_interval:  int = 12,    # seconds — roughly one L1 block
    ):
        w3 = Web3(Web3.HTTPProvider(l1_rpc_url, request_kwargs={"timeout": 30}))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._w3       = w3
        self._contract = w3.eth.contract(
            address=Web3.to_checksum_address(bridge_address),
            abi=bridge_abi,
        )
        self._l2_rpc       = l2_rpc_url
        self._privkey      = sequencer_privkey
        self._l2_chain_id  = l2_chain_id
        self._poll         = poll_interval
        self._db_path      = db_path

        con = sqlite3.connect(db_path)
        con.execute(CREATE)
        con.commit()
        con.close()
        log.info("bridge_watcher_ready l1=%s bridge=%s", l1_rpc_url[:30], bridge_address[:12])

    async def run(self) -> None:
        while True:
            try:
                await self._poll_l1()
            except Exception as exc:
                log.error("watcher_poll_error err=%s", exc, exc_info=True)
            await asyncio.sleep(self._poll)

    async def _poll_l1(self) -> None:
        last_block = self._last_seen_block()
        current    = self._w3.eth.block_number

        if current <= last_block:
            return

        logs = self._contract.events.DepositInitiated.get_logs(
            from_block=last_block + 1,
            to_block=current,
        )

        for evt in logs:
            args    = evt.args
            nonce   = int(args.depositNonce)
            already = self._is_relayed(nonce)
            if already:
                continue

            log.info(
                "deposit_detected nonce=%d amount=%s l2_recipient=%s",
                nonce, args.amount, args.l2Recipient,
            )

            l2_tx_hash = await self._relay_to_l2(
                nonce=nonce,
                recipient=args.l2Recipient,
                amount=int(args.amount),
                l1_tx_hash=evt.transactionHash.hex(),
            )

            self._record_relay(nonce, evt.transactionHash.hex(), l2_tx_hash,
                               int(args.amount), args.l1Sender, args.l2Recipient)

        self._update_last_seen(current)

    async def _relay_to_l2(self, nonce: int, recipient: str, amount: int, l1_tx_hash: str) -> str:
        """Submit TX_BRIDGE_DEPOSIT to the L2 RPC."""
        import aiohttp, uuid
        from chain.types import TxType
        from chain.crypto import keccak256_hex

        payload_dict = {
            "l1_deposit_nonce": nonce,
            "recipient":        recipient,
            "amount":           amount,
            "l1_tx_hash":       l1_tx_hash,
        }

        tx_dict = {
            "tx_type":   TxType.BRIDGE_DEPOSIT,
            "sender":    "",
            "nonce":     0,
            "payload":   json.dumps(payload_dict, separators=(",", ":")),
            "gas_price": 0,
            "signature": "",
            "tx_hash":   keccak256_hex(("BRIDGE_DEPOSIT" + str(nonce) + recipient).encode()),
        }

        body = {"jsonrpc": "2.0", "method": "eth_sendRawTransaction",
                "params": [json.dumps(tx_dict).encode().hex()], "id": 1}

        async with aiohttp.ClientSession() as session:
            async with session.post(self._l2_rpc, json=body, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                return data.get("result", "")

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _last_seen_block(self) -> int:
        con = sqlite3.connect(self._db_path)
        row = con.execute("SELECT MAX(rowid) FROM deposits").fetchone()
        con.close()
        return row[0] or 0

    def _is_relayed(self, nonce: int) -> bool:
        con = sqlite3.connect(self._db_path)
        row = con.execute("SELECT relayed FROM deposits WHERE nonce=?", (nonce,)).fetchone()
        con.close()
        return bool(row and row[0])

    def _record_relay(self, nonce, l1_tx, l2_tx, amount, l1_sender, l2_recipient) -> None:
        con = sqlite3.connect(self._db_path)
        con.execute(
            "INSERT OR IGNORE INTO deposits (nonce,l1_tx_hash,l2_tx_hash,amount,l1_sender,l2_recipient,relayed) "
            "VALUES (?,?,?,?,?,?,1)",
            (nonce, l1_tx, l2_tx, str(amount), l1_sender, l2_recipient),
        )
        con.commit()
        con.close()

    def _update_last_seen(self, block: int) -> None:
        pass   # last_seen tracked via MAX(nonce) in MVP; swap to a separate cursor table in v2
