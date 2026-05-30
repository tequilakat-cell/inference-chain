"""
InferenceChain L2 Python SDK.

High-level client for posting jobs, awaiting results, and bridging tokens.
Mirrors the InferenceClient from inference/sdk/ but targets the L2 RPC.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

log = logging.getLogger("inference_chain_sdk")


@dataclass
class L2InferenceResult:
    job_id:       str
    output:       str
    shard_count:  int
    mode:         str
    elapsed_sec:  float
    output_hash:  str


class InferenceChainClient:
    """
    Client for the InferenceChain L2 JSON-RPC API.

    Example:
        client = InferenceChainClient(
            rpc_url="http://127.0.0.1:8545",
            private_key="0x...",
        )
        result = client.infer(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            prompt="Explain quantum computing in one sentence.",
            n_shards=3,
            shard_mode="parallel_sample",
        )
        print(result.output)
    """

    def __init__(
        self,
        rpc_url:         str,
        private_key:     str,
        l1_rpc_url:      Optional[str]  = None,
        bridge_address:  Optional[str]  = None,
        bridge_abi:      Optional[list] = None,
        l1_inft_address: Optional[str]  = None,
        l1_inft_abi:     Optional[list] = None,
    ):
        self.rpc_url          = rpc_url.rstrip("/")
        self._privkey         = private_key
        self._req_id          = 0
        self._l1_rpc_url      = l1_rpc_url
        self._bridge_address  = bridge_address
        self._bridge_abi      = bridge_abi
        self._l1_inft_address = l1_inft_address
        self._l1_inft_abi     = l1_inft_abi

    # ── Core inference ────────────────────────────────────────────────────────

    def infer(
        self,
        model:      str,
        prompt:     str,
        max_tokens: int = 256,
        n_shards:   int = 1,
        shard_mode: str = "parallel_sample",
        timeout:    float = 120.0,
    ) -> L2InferenceResult:
        """
        Post an inference job and block until it completes.

        Args:
            model:      HuggingFace model ID registered on-chain.
            prompt:     The inference prompt.
            max_tokens: Maximum output tokens per shard.
            n_shards:   Number of parallel miners to use (1-8).
            shard_mode: "parallel_sample" | "context_split" | "speculative"
            timeout:    Seconds to wait for completion.

        Returns:
            L2InferenceResult with the assembled output and metadata.
        """
        start   = time.monotonic()
        job_id  = self.post_job(model, prompt, max_tokens, n_shards, shard_mode)
        log.info("l2_job_posted job_id=%s mode=%s n_shards=%d", job_id, shard_mode, n_shards)

        deadline = time.time() + timeout
        attempt  = 0
        while time.time() < deadline:
            status = self.get_job(job_id)
            if status and status.get("status") == "complete":
                elapsed = time.monotonic() - start
                return L2InferenceResult(
                    job_id=job_id,
                    output=status.get("final_output", ""),
                    shard_count=n_shards,
                    mode=shard_mode,
                    elapsed_sec=elapsed,
                    output_hash=status.get("output_hash", ""),
                )
            sleep_secs = min(10.0, 0.5 * (2 ** min(attempt, 4)))
            time.sleep(sleep_secs)
            attempt += 1

        raise TimeoutError(f"L2 job {job_id} did not complete within {timeout}s")

    # ── Low-level job API ─────────────────────────────────────────────────────

    def post_job(
        self,
        model:      str,
        prompt:     str,
        max_tokens: int = 256,
        n_shards:   int = 1,
        shard_mode: str = "parallel_sample",
    ) -> str:
        """Post a job and return the job_id."""
        result = self._call("inft_postJob", [model, prompt, max_tokens, shard_mode, n_shards, self._privkey])
        return result

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get current job status and shard details."""
        return self._call("inft_getJob", [job_id])

    def await_job(self, job_id: str, timeout: float = 120.0) -> dict:
        """Long-poll the server until the job completes."""
        return self._call("inft_awaitJob", [job_id, timeout])

    # ── Account & chain info ──────────────────────────────────────────────────

    def get_balance(self, address: str) -> int:
        acc = self._call("inft_getAccount", [address])
        return acc.get("balance_inft", 0)

    def get_stake(self, address: str) -> int:
        acc = self._call("inft_getAccount", [address])
        return acc.get("stake_inft", 0)

    def get_chain_info(self) -> dict:
        return self._call("inft_getChainInfo", [])

    def get_miner_info(self, address: str) -> dict:
        return self._call("inft_getMinerInfo", [address])

    def stake(self, amount: int) -> str:
        """Stake INFT as a validator. Returns the L2 tx_hash."""
        return self._call("inft_stake", [amount, self._privkey])

    # ── Bridge ────────────────────────────────────────────────────────────────

    def bridge_deposit(self, l1_amount: int, l1_privkey: str, l2_recipient: Optional[str] = None) -> str:
        """
        Lock INFT on L1 and mint on L2.
        Requires the client to be initialised with L1 bridge info via from_deployment().
        Returns the L1 deposit tx hash.

        Note: This initiates the deposit; the bridge watcher mints on L2 asynchronously
        (typically within 1-2 L1 blocks, ~15 seconds on Sepolia).
        """
        if not all([self._l1_rpc_url, self._bridge_address, self._bridge_abi,
                    self._l1_inft_address, self._l1_inft_abi]):
            raise RuntimeError(
                "bridge_deposit requires L1 bridge info. "
                "Initialise the client with InferenceChainClient.from_deployment(l1_dep_path, ...)."
            )

        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
        from eth_account import Account

        w3 = Web3(Web3.HTTPProvider(self._l1_rpc_url, request_kwargs={"timeout": 30}))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        account = Account.from_key(l1_privkey)

        bridge = w3.eth.contract(
            address=Web3.to_checksum_address(self._bridge_address),
            abi=self._bridge_abi,
        )
        inft = w3.eth.contract(
            address=Web3.to_checksum_address(self._l1_inft_address),
            abi=self._l1_inft_abi,
        )

        recipient = Web3.to_checksum_address(l2_recipient or account.address)

        latest   = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)
        priority = w3.to_wei("0.01", "gwei")
        fees = {
            "maxFeePerGas":         base_fee * 2 + priority,
            "maxPriorityFeePerGas": priority,
            "type":                 "0x2",
        }

        # Step 1 — approve bridge to pull INFT
        nonce = w3.eth.get_transaction_count(account.address, "pending")
        approve_tx = inft.functions.approve(
            Web3.to_checksum_address(self._bridge_address),
            l1_amount,
        ).build_transaction({"from": account.address, "nonce": nonce, "gas": 80_000, **fees})
        signed = account.sign_transaction(approve_tx)
        approve_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(approve_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"INFT approve reverted: {approve_hash.hex()}")
        log.info("bridge_approve_ok tx=%s", approve_hash.hex()[:12])

        # Step 2 — lock INFT in bridge and emit DepositInitiated
        deposit_tx = bridge.functions.depositINFT(
            l1_amount,
            recipient,
        ).build_transaction({"from": account.address, "nonce": nonce + 1, "gas": 150_000, **fees})
        signed = account.sign_transaction(deposit_tx)
        deposit_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(deposit_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"depositINFT reverted: {deposit_hash.hex()}")
        log.info("bridge_deposit_ok tx=%s amount=%d l2_recipient=%s",
                 deposit_hash.hex()[:12], l1_amount, recipient[:10])
        return deposit_hash.hex()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, method: str, params: list):
        self._req_id += 1
        body    = {"jsonrpc": "2.0", "method": method, "params": params, "id": self._req_id}
        resp    = requests.post(self.rpc_url, json=body, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error {method}: {data['error']}")
        return data.get("result")

    @classmethod
    def from_deployment(cls, l1_dep_path: str, l2_rpc_url: str, private_key: str) -> "InferenceChainClient":
        with open(l1_dep_path) as f:
            dep = json.load(f)
        return cls(
            rpc_url=l2_rpc_url,
            private_key=private_key,
            l1_rpc_url=dep.get("l1_rpc_url"),
            bridge_address=dep.get("bridge_address"),
            bridge_abi=dep.get("bridge_abi"),
            l1_inft_address=dep.get("l1_inft_address"),
            l1_inft_abi=dep.get("l1_inft_abi"),
        )
