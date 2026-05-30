"""
Async-safe nonce manager for concurrent blockchain transactions.

Without this, parallel claimJob / submitResult / finalizeJob calls race to
read the same on-chain nonce, resulting in "nonce too low" rejections.
This manager serialises nonce allocation behind an asyncio.Lock and
resets from the chain after any reverted transaction.
"""

import asyncio
import logging
from web3 import Web3

log = logging.getLogger("miner.nonce")


class NonceManager:
    def __init__(self, w3: Web3, address: str):
        self._w3      = w3
        self._address = address
        self._nonce: int | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> int:
        """Reserve the next nonce. Thread-safe across concurrent coroutines."""
        async with self._lock:
            if self._nonce is None:
                # Use 'pending' to account for transactions in the mempool
                self._nonce = self._w3.eth.get_transaction_count(
                    self._address, "pending"
                )
                log.debug("Nonce reset from chain: %d", self._nonce)
            n = self._nonce
            self._nonce += 1
            return n

    async def reset(self) -> None:
        """Force-refresh nonce from chain. Call after a reverted or stuck tx."""
        async with self._lock:
            self._nonce = None
        log.warning("Nonce manager reset — will re-read from chain on next tx")

    async def peek(self) -> int:
        """Return current nonce without allocating (for diagnostics)."""
        async with self._lock:
            if self._nonce is None:
                return self._w3.eth.get_transaction_count(self._address, "pending")
            return self._nonce
