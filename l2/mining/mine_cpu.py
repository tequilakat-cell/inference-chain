"""
MINE token CPU miner.

Runs continuously, yielding to inference jobs whenever one arrives.
Uses all available CPU cores (configurable via MINE_THREADS env var).

Usage:
    # Standalone
    PRIVATE_KEY=0x... L2_RPC=http://104.197.6.1/rpc python mine_cpu.py

    # The L2 miner starts this automatically when the inference queue is idle.

Algorithm:
    For each nonce in [random_start, random_start + BATCH_SIZE):
        digest = keccak256(challenge || miner_address || nonce)
        if int(digest) < mining_target:
            submit mine_submit(nonce, "0x" + digest.hex(), private_key)
            break
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import time
from pathlib import Path

import aiohttp

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from chain.mine import compute_digest, difficulty_display

log = logging.getLogger("mine_cpu")

BATCH_SIZE   = 50_000       # nonces to try per batch (tune for your CPU)
REPORT_EVERY = 10           # seconds between hashrate reports
RETRY_DELAY  = 3            # seconds to wait after a failed RPC call


class CpuMiner:
    def __init__(
        self,
        l2_rpc:      str,
        private_key: str,
        threads:     int = 0,   # 0 = auto (all cores)
    ):
        self.l2_rpc     = l2_rpc
        self.privkey    = private_key
        self._paused    = asyncio.Event()
        self._paused.set()  # start un-paused
        self._shutdown  = asyncio.Event()
        self._hashes    = 0
        self._t_start   = time.monotonic()
        self._threads   = threads or os.cpu_count() or 1

        # Derive address from key
        from eth_account import Account
        self.address = Account.from_key(private_key).address.lower()
        log.info("mine_cpu address=%s threads=%d rpc=%s", self.address, self._threads, l2_rpc)

    # ── Public control ────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause mining (e.g. while an inference job is running)."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume mining."""
        self._paused.set()

    def stop(self) -> None:
        self._shutdown.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("mine_cpu_started")
        while not self._shutdown.is_set():
            # Wait if paused (inference job in progress)
            await self._paused.wait()

            try:
                info = await self._get_mining_info()
            except Exception as exc:
                log.warning("mine_get_info_failed err=%s", exc)
                await asyncio.sleep(RETRY_DELAY)
                continue

            challenge = bytes.fromhex(info["challenge"].removeprefix("0x"))
            target    = int(info["mining_target"])
            reward    = info["current_reward_human"]
            diff      = info["difficulty"]

            if reward == 0:
                log.info("mine_max_supply_reached — stopping")
                break

            log.info("mine_epoch=%d diff=%s reward=%.4f MINE challenge=%s",
                     info["epoch"], diff, reward, "0x" + challenge[:4].hex() + "…")

            # Mine one solution
            found = await self._mine_batch(challenge, target)
            if found is not None:
                nonce, digest = found
                await self._submit(nonce, "0x" + digest.hex())

    # ── Mining ────────────────────────────────────────────────────────────────

    async def _mine_batch(
        self, challenge: bytes, target: int
    ) -> tuple[int, bytes] | None:
        """
        Try BATCH_SIZE nonces across all CPU threads via run_in_executor.
        Returns (nonce, digest) if a solution is found, else None.
        """
        loop    = asyncio.get_event_loop()
        address = self.address
        t_last  = time.monotonic()

        start = random.getrandbits(64)

        while not self._shutdown.is_set():
            await self._paused.wait()

            batch_start = start  # capture before executor runs
            start += BATCH_SIZE  # advance BEFORE the executor so next batch is different

            def _search(s=batch_start) -> tuple[int, bytes] | None:
                for nonce in range(s, s + BATCH_SIZE):
                    digest = compute_digest(challenge, address, nonce)
                    if int.from_bytes(digest, "big") < target:
                        return nonce, digest
                return None

            result = await loop.run_in_executor(None, _search)
            self._hashes += BATCH_SIZE

            now = time.monotonic()
            if now - t_last >= REPORT_EVERY:
                elapsed  = now - self._t_start
                hashrate = self._hashes / elapsed / 1_000
                log.info("mine_hashrate=%.1f kH/s total_hashes=%s",
                         hashrate, f"{self._hashes:,}")
                t_last = now

            if result is not None:
                nonce, digest = result
                log.info("mine_solution_found nonce=%d digest=0x%s",
                         nonce, digest[:8].hex() + "…")
                return nonce, digest

            # Refresh mining info every batch in case challenge changed
            try:
                fresh = await self._get_mining_info()
                new_challenge = bytes.fromhex(fresh["challenge"].removeprefix("0x"))
                if new_challenge != challenge:
                    log.info("mine_challenge_updated — fetching new puzzle")
                    return None   # let caller refresh
                target = int(fresh["mining_target"])
            except Exception:
                pass   # keep mining with stale info

        return None

    # ── RPC helpers ───────────────────────────────────────────────────────────

    async def _rpc(self, method: str, params: list) -> dict:
        async with aiohttp.ClientSession() as s:
            r = await s.post(self.l2_rpc, json={
                "jsonrpc": "2.0", "method": method, "params": params, "id": 1
            }, timeout=aiohttp.ClientTimeout(total=10))
            d = await r.json()
        if "error" in d:
            raise RuntimeError(d["error"].get("message", str(d["error"])))
        return d["result"]

    async def _get_mining_info(self) -> dict:
        return await self._rpc("mine_getInfo", [])

    async def _submit(self, nonce: int, challenge_digest: str) -> None:
        log.info("mine_submitting nonce=%d", nonce)
        try:
            tx_hash = await self._rpc("mine_submit", [nonce, challenge_digest, self.privkey])
            log.info("mine_submitted tx=%s", tx_hash[:16] + "…")
            # Brief pause to let the block settle before next round
            await asyncio.sleep(2)
        except Exception as exc:
            log.warning("mine_submit_failed err=%s", exc)


# ── Entry point (standalone) ──────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    privkey = os.environ.get("PRIVATE_KEY", "")
    rpc     = os.environ.get("L2_RPC", "http://104.197.6.1/rpc")
    if not privkey:
        print("Set PRIVATE_KEY=0x... env var", file=sys.stderr)
        sys.exit(1)

    miner = CpuMiner(l2_rpc=rpc, private_key=privkey)
    await miner.run()

if __name__ == "__main__":
    asyncio.run(main())
