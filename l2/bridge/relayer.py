"""
Bridge relayer — finalises L2→L1 withdrawals after state roots are finalised.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

log = logging.getLogger("bridge.relayer")

CREATE = """
CREATE TABLE IF NOT EXISTS withdrawals (
    l2_tx_hash  TEXT PRIMARY KEY,
    l1_recipient TEXT,
    amount       TEXT,
    l2_block     INTEGER,
    finalized    INTEGER DEFAULT 0,
    l1_tx_hash   TEXT
);
"""


class BridgeRelayer:
    def __init__(
        self,
        l1_rpc_url:      str,
        bridge_address:  str,
        bridge_abi:      list,
        rollup_address:  str,
        rollup_abi:      list,
        l2_rpc_url:      str,
        relayer_privkey: str,
        db_path:         str = "bridge.db",
        poll_interval:   int = 300,    # 5 min — L1 finalisation check cadence
    ):
        w3 = Web3(Web3.HTTPProvider(l1_rpc_url, request_kwargs={"timeout": 30}))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._w3      = w3
        self._bridge  = w3.eth.contract(
            address=Web3.to_checksum_address(bridge_address), abi=bridge_abi
        )
        self._rollup  = w3.eth.contract(
            address=Web3.to_checksum_address(rollup_address), abi=rollup_abi
        )
        self._l2_rpc  = l2_rpc_url
        self._account = Account.from_key(relayer_privkey)
        self._poll    = poll_interval
        self._db      = db_path

        con = sqlite3.connect(db_path)
        con.execute(CREATE)
        con.commit()
        con.close()
        log.info("bridge_relayer_ready bridge=%s", bridge_address[:12])

    async def run(self) -> None:
        while True:
            try:
                await self._sync_withdrawals()
                await self._finalize_ready()
            except Exception as exc:
                log.error("relayer_error err=%s", exc, exc_info=True)
            await asyncio.sleep(self._poll)

    async def _sync_withdrawals(self) -> None:
        """Poll L2 RPC for pending withdrawal transactions."""
        import aiohttp
        body = {"jsonrpc": "2.0", "method": "inft_getPendingWithdrawals", "params": [], "id": 1}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._l2_rpc, json=body, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    if "error" in data:
                        log.warning("sync_withdrawals_rpc_error err=%s", data["error"])
                        return
                    for w in data.get("result") or []:
                        self._upsert_withdrawal(w)
        except Exception as exc:
            log.warning("sync_withdrawals_failed err=%s", exc)

    def _upsert_withdrawal(self, w: dict) -> None:
        con = sqlite3.connect(self._db)
        con.execute(
            "INSERT OR IGNORE INTO withdrawals (l2_tx_hash,l1_recipient,amount,l2_block) VALUES (?,?,?,?)",
            (w["l2_tx_hash"], w["l1_recipient"], str(w["amount"]), w["l2_block"]),
        )
        con.commit()
        con.close()

    async def _finalize_ready(self) -> None:
        """Check which pending withdrawals have their L2 block finalised, and submit to L1."""
        con = sqlite3.connect(self._db)
        rows = con.execute(
            "SELECT l2_tx_hash,l1_recipient,amount,l2_block FROM withdrawals WHERE finalized=0"
        ).fetchall()
        con.close()

        for l2_tx_hash, l1_recipient, amount_str, l2_block in rows:
            try:
                is_fin = self._rollup.functions.isFinalized(l2_block).call()
                if not is_fin:
                    continue

                log.info("finalizing_withdrawal l2_tx=%s l2_block=%d", l2_tx_hash[:12], l2_block)
                l1_tx = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._submit_finalization(l2_tx_hash, l1_recipient, int(amount_str), l2_block),
                )

                con = sqlite3.connect(self._db)
                con.execute(
                    "UPDATE withdrawals SET finalized=1, l1_tx_hash=? WHERE l2_tx_hash=?",
                    (l1_tx, l2_tx_hash),
                )
                con.commit()
                con.close()
                log.info("withdrawal_finalized l2_tx=%s l1_tx=%s", l2_tx_hash[:12], l1_tx[:12])
            except Exception as exc:
                log.warning("finalize_failed l2_tx=%s err=%s", l2_tx_hash[:12], exc)

    def _submit_finalization(
        self, l2_tx_hash: str, l1_recipient: str, amount: int, l2_block: int
    ) -> str:
        nonce = self._w3.eth.get_transaction_count(self._account.address, "pending")
        base  = self._w3.eth.get_block("latest").get("baseFeePerGas", self._w3.eth.gas_price)

        # In MVP, merkle proof is empty (single-node chain — root IS the leaf hash).
        # Production would build the actual merkle proof from L2 state.
        leaf = Web3.keccak(
            Web3.to_bytes(hexstr=l2_tx_hash) +
            Web3.to_bytes(hexstr=l1_recipient) +
            amount.to_bytes(32, "big")
        )

        tx = self._bridge.functions.finalizeWithdrawal(
            l2_block,
            Web3.to_checksum_address(l1_recipient),
            amount,
            Web3.to_bytes(hexstr=l2_tx_hash),
            [],  # merkle proof — MVP: empty (single node tree)
        ).build_transaction({
            "from":                 self._account.address,
            "nonce":                nonce,
            "gas":                  200_000,
            "maxFeePerGas":         base * 2,
            "maxPriorityFeePerGas": self._w3.to_wei("0.01", "gwei"),
            "type":                 "0x2",
        })
        signed  = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"finalizeWithdrawal reverted: {tx_hash.hex()}")
        return tx_hash.hex()
