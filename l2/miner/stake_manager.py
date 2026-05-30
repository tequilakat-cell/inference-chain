"""
StakeManager — ensures the miner maintains sufficient L2 INFT stake.

Reads current stake from L2 RPC and submits TX_STAKE if below minimum.
"""

from __future__ import annotations

import logging
import aiohttp
import json

log = logging.getLogger("l2_miner.stake_manager")


class StakeManager:
    def __init__(self, l2_rpc_url: str, address: str, privkey: str):
        self._rpc     = l2_rpc_url
        self._address = address
        self._privkey = privkey

    async def ensure_minimum_stake(self, min_stake: int) -> None:
        """Stake up to min_stake if current stake is below the minimum."""
        try:
            current_stake = await self._get_stake()
            if current_stake >= min_stake:
                log.info("stake_ok address=%s stake=%d", self._address[:10], current_stake)
                return

            needed = min_stake - current_stake
            balance = await self._get_balance()
            if balance < needed:
                log.warning(
                    "insufficient_balance_to_stake address=%s balance=%d needed=%d",
                    self._address[:10], balance, needed,
                )
                return

            await self._submit_stake(needed)
            log.info("stake_submitted address=%s amount=%d", self._address[:10], needed)
        except Exception as exc:
            log.warning("stake_manager_error err=%s", exc)

    async def _rpc_call(self, method: str, params: list) -> dict:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        async with aiohttp.ClientSession() as session:
            async with session.post(self._rpc, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return await resp.json()

    async def _get_stake(self) -> int:
        result = await self._rpc_call("inft_getAccount", [self._address])
        return int(result.get("result", {}).get("stake_inft", 0))

    async def _get_balance(self) -> int:
        result = await self._rpc_call("inft_getAccount", [self._address])
        return int(result.get("result", {}).get("balance_inft", 0))

    async def _submit_stake(self, amount: int) -> None:
        tx_payload = {"jsonrpc": "2.0", "method": "inft_stake", "params": [amount, self._privkey], "id": 1}
        await self._rpc_call("inft_stake", [amount, self._privkey])
