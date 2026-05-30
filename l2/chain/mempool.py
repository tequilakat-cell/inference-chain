"""
Transaction mempool.

Priority queue ordered by gas_price DESC, nonce ASC.
Thread-safe via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from typing import Optional

from .types import Transaction
from .crypto import keccak256_hex, encode_transaction, recover, verify_sig

log = logging.getLogger("chain.mempool")


class Mempool:
    def __init__(self, max_size: int = 10_000):
        self._max_size = max_size
        self._lock = asyncio.Lock()
        # heap entries: (-gas_price, nonce, tx_hash, tx)
        self._heap: list[tuple] = []
        self._seen: set[str] = set()
        # sender → list of pending nonces (for nonce gap detection)
        self._pending_nonces: dict[str, set[int]] = {}

    async def add(self, tx: Transaction) -> bool:
        """
        Add a validated transaction to the pool.
        Returns True if added, False if duplicate or pool full.
        """
        async with self._lock:
            if tx.tx_hash in self._seen:
                return False
            if len(self._heap) >= self._max_size:
                log.warning("mempool_full dropping tx=%s", tx.tx_hash[:12])
                return False
            self._seen.add(tx.tx_hash)
            heapq.heappush(self._heap, (-tx.gas_price, tx.nonce, tx.tx_hash, tx))
            if tx.sender:
                self._pending_nonces.setdefault(tx.sender.lower(), set()).add(tx.nonce)
            log.debug("mempool_add tx=%s type=%d size=%d", tx.tx_hash[:12], tx.tx_type, len(self._heap))
            return True

    async def pop_batch(self, max_count: int = 500) -> list[Transaction]:
        """Pop up to max_count highest-priority transactions."""
        async with self._lock:
            batch: list[Transaction] = []
            temp: list[tuple] = []
            seen_senders: dict[str, int] = {}  # sender → last nonce included

            while self._heap and len(batch) < max_count:
                neg_price, nonce, tx_hash, tx = heapq.heappop(self._heap)
                sender = tx.sender.lower() if tx.sender else ""

                # Enforce nonce ordering per sender (skip gaps)
                if sender:
                    last = seen_senders.get(sender)
                    if last is not None and nonce != last + 1:
                        temp.append((neg_price, nonce, tx_hash, tx))
                        continue
                    seen_senders[sender] = nonce

                self._seen.discard(tx_hash)
                if sender:
                    self._pending_nonces.get(sender, set()).discard(nonce)
                batch.append(tx)

            # Put back any txs we couldn't include due to nonce gaps
            for entry in temp:
                heapq.heappush(self._heap, entry)

            return batch

    async def remove(self, tx_hash: str) -> None:
        async with self._lock:
            self._seen.discard(tx_hash)

    async def size(self) -> int:
        async with self._lock:
            return len(self._heap)

    async def pending_for(self, address: str) -> set[int]:
        async with self._lock:
            return self._pending_nonces.get(address.lower(), set()).copy()


def build_transaction(
    tx_type:   int,
    sender:    str,
    nonce:     int,
    payload:   dict,
    chain_id:  int,
    private_key: Optional[str] = None,
    gas_price: int = 1,
) -> Transaction:
    """
    Build and optionally sign a transaction.
    If private_key is None, returns an unsigned tx (for sequencer-synthesized txs).
    """
    import json
    from .crypto import keccak256_hex, encode_transaction, sign

    tx_partial = Transaction(
        tx_type=tx_type,
        sender=sender,
        nonce=nonce,
        payload=json.dumps(payload, separators=(",", ":")),
        gas_price=gas_price,
        signature="",
        tx_hash="",
    )

    # Compute hash over unsigned fields
    preimage = encode_transaction({
        **tx_partial.to_dict(),
        "chain_id": chain_id,
    })
    tx_hash = keccak256_hex(preimage)

    sig = sign(private_key, preimage) if private_key else ""

    return Transaction(
        tx_type=tx_type,
        sender=sender,
        nonce=nonce,
        payload=tx_partial.payload,
        gas_price=gas_price,
        signature=sig,
        tx_hash=tx_hash,
    )


def validate_transaction(tx: Transaction, current_nonce: int, chain_id: int) -> tuple[bool, str]:
    """
    Stateless transaction validation.
    Returns (valid, reason).
    """
    from .crypto import encode_transaction, recover

    if not tx.tx_hash:
        return False, "missing tx_hash"

    # Sequencer-synthesized transactions have no signature
    from .types import TxType
    if tx.tx_type in (TxType.SHARD_COMMIT, TxType.BRIDGE_DEPOSIT, TxType.SLASH, TxType.SLASH_HARD):
        return True, ""

    if not tx.signature:
        return False, "missing signature"

    try:
        preimage = encode_transaction({
            **{k: v for k, v in tx.to_dict().items() if k not in ("signature", "tx_hash")},
            "chain_id": chain_id,
        })
        recovered = recover(preimage, tx.signature)
        if recovered.lower() != tx.sender.lower():
            return False, f"sig mismatch: recovered {recovered}, sender {tx.sender}"
    except Exception as exc:
        return False, f"sig verify failed: {exc}"

    return True, ""
