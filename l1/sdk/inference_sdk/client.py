"""
InferenceToken Python SDK
=========================
Post inference jobs to the contract, retrieve results, register as a miner
or model creator, and check on-chain stats.

Quick start:
    from inference_sdk import InferenceClient

    client = InferenceClient.from_deployment("deployment.json", private_key="0x...")
    response = client.infer(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        prompt="Explain transformers in simple terms",
        max_tokens=512,
    )
    print(response.text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

log = logging.getLogger("inference_sdk")

JOB_FEE    = Web3.to_wei("0.001", "ether")
MINER_BOND = Web3.to_wei("0.005", "ether")

JOB_STATUS = {0: "Open", 1: "Claimed", 2: "Submitted", 3: "Complete", 4: "Disputed", 5: "Expired"}


# ── RSA key helpers (optional) ─────────────────────────────────────────────────

def generate_rsa_keypair(key_dir: Optional[str] = None) -> tuple[bytes, bytes]:
    """
    Generate an RSA-2048 keypair for encrypted inference.
    Returns (private_key_pem, public_key_der).
    If key_dir is given, persists keys to disk.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    if key_dir:
        kd = Path(key_dir)
        kd.mkdir(parents=True, exist_ok=True)
        (kd / "private.pem").write_bytes(priv_pem)
        (kd / "public.der").write_bytes(pub_der)
        log.info("RSA keypair saved to %s", kd)

    return priv_pem, pub_der


def rsa_encrypt(public_key_der: bytes, plaintext: bytes) -> bytes:
    """RSA-OAEP+SHA-256 encrypt with a DER public key."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    pub = serialization.load_der_public_key(public_key_der)
    return pub.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def rsa_decrypt(private_key_pem: bytes, ciphertext: bytes) -> bytes:
    """RSA-OAEP+SHA-256 decrypt with a PEM private key."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    priv = serialization.load_pem_private_key(private_key_pem, password=None)
    return priv.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def load_or_generate_rsa(key_dir: Optional[str] = None) -> tuple[bytes, bytes]:
    """Load RSA keys from disk or generate new ones. Returns (priv_pem, pub_der)."""
    kd = Path(key_dir or os.path.expanduser("~/.inference-sdk/keys"))
    priv_path = kd / "private.pem"
    pub_path  = kd / "public.der"
    if priv_path.exists() and pub_path.exists():
        return priv_path.read_bytes(), pub_path.read_bytes()
    return generate_rsa_keypair(str(kd))


# ── Thread-safe nonce manager for SDK (sync version) ───────────────────────────

class SyncNonceManager:
    """Thread-safe nonce allocation for synchronous SDK usage."""

    def __init__(self, w3: Web3, address: str):
        self._w3 = w3
        self._address = address
        self._nonce: Optional[int] = None
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            if self._nonce is None:
                self._nonce = self._w3.eth.get_transaction_count(self._address, "pending")
            n = self._nonce
            self._nonce += 1
            return n

    def reset(self) -> None:
        with self._lock:
            self._nonce = None


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass
class InferenceResponse:
    job_id:        int
    text:          str
    model:         str
    status:        str
    tokens_minted: int
    elapsed_sec:   float


@dataclass
class JobInfo:
    job_id:         int
    requester:      str
    miner:          str
    challenger:     str
    status:         int
    status_label:   str
    model_id:       str
    max_output_tokens: int
    posted_at:      int
    submitted_at:   int
    input_ref:      str
    output_ref:     str
    prompt:         Optional[str] = None
    output:         Optional[str] = None


# ── Main client ────────────────────────────────────────────────────────────────

class InferenceClient:

    def __init__(
        self,
        rpc_url:          str,
        contract_address: str,
        contract_abi:     list,
        private_key:      str,
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.account  = Account.from_key(private_key)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=contract_abi,
        )
        self._nonces = SyncNonceManager(self.w3, self.account.address)
        self._rsa_priv: Optional[bytes] = None
        self._rsa_pub:  Optional[bytes] = None

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_deployment(
        cls,
        deployment_json: str,
        private_key: str,
        rpc_url: Optional[str] = None,
    ) -> "InferenceClient":
        with open(deployment_json) as f:
            dep = json.load(f)
        return cls(
            rpc_url=rpc_url or dep.get("rpc_url", ""),
            contract_address=dep["address"],
            contract_abi=dep["abi"],
            private_key=private_key,
        )

    # ── RSA helpers for requesters ────────────────────────────────────────

    def load_rsa_keys(self, key_dir: Optional[str] = None) -> None:
        """Load RSA keypair for decrypting miner outputs."""
        self._rsa_priv, self._rsa_pub = load_or_generate_rsa(key_dir)
        log.info("RSA keypair loaded")

    def get_public_key_der(self) -> Optional[bytes]:
        return self._rsa_pub

    # ── Transaction helpers ───────────────────────────────────────────────

    def _eip1559_fees(self) -> dict:
        latest   = self.w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", self.w3.eth.gas_price)
        priority = Web3.to_wei("0.01", "gwei")
        return {
            "maxFeePerGas":         base_fee * 2 + priority,
            "maxPriorityFeePerGas": priority,
            "type":                 "0x2",
        }

    def _build_and_send(self, fn, value: int = 0, gas: int = 400_000, retries: int = 2) -> dict:
        """Build, sign, send a tx, wait for receipt. Retries on nonce failure."""
        for attempt in range(retries + 1):
            try:
                tx = fn.build_transaction({
                    "from":  self.account.address,
                    "value": value,
                    "nonce": self._nonces.get(),
                    "gas":   gas,
                    **self._eip1559_fees(),
                })
                signed  = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt.status != 1:
                    raise RuntimeError(f"tx reverted: {tx_hash.hex()}")
                return receipt
            except Exception as exc:
                log.warning("tx attempt %d/%d failed: %s", attempt + 1, retries + 1, exc)
                self._nonces.reset()
                if attempt == retries:
                    raise
        raise RuntimeError("unreachable")

    # ── Core API ──────────────────────────────────────────────────────────

    def infer(
        self,
        model:         str,
        prompt:        str,
        max_tokens:    int = 512,
        poll_interval: int = 5,
        timeout:       int = 900,
        encrypt:       bool = False,
    ) -> InferenceResponse:
        """
        Post a job, block until a miner completes it, return the result.
        Raises TimeoutError if no miner responds within `timeout` seconds.
        If encrypt=True and RSA keys are loaded, encrypts the prompt.
        """
        start = time.monotonic()

        # Encrypt prompt if requested
        prompt_bytes = prompt.encode("utf-8")
        if encrypt and self._rsa_pub:
            # We need the miner's public key — for now we fetch miners for model
            miners = self.get_miners_for_model(model, limit=1)
            if miners:
                profile = self.get_miner_profile(miners[0])
                if profile.get("pub_key"):
                    prompt_bytes = rsa_encrypt(profile["pub_key"], prompt_bytes)
                    log.info("prompt_encrypted for miner=%s", miners[0][:10])

        job_id = self.post_job(model, prompt_bytes, max_tokens)
        log.info("Job #%s posted — waiting for miner…", job_id)

        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_id)

            if job["status"] == 3:  # Complete
                text = self._fetch_output(job_id, decrypt=encrypt)
                elapsed = time.monotonic() - start
                return InferenceResponse(
                    job_id=job_id, text=text, model=model, status="Complete",
                    tokens_minted=self._calc_reward(max_tokens), elapsed_sec=elapsed,
                )
            if job["status"] == 4:
                raise RuntimeError(f"Job #{job_id} disputed — await owner arbitration.")
            if job["status"] == 5:
                raise RuntimeError(f"Job #{job_id} expired without a miner. Try again.")

            log.debug("Job #%s → %s…", job_id, JOB_STATUS.get(job["status"], "?"))
            time.sleep(poll_interval)

        raise TimeoutError(f"Job #{job_id} did not complete within {timeout}s.")

    def post_job(
        self,
        model: str,
        prompt: str | bytes,
        max_tokens: int = 512,
    ) -> int:
        """
        Post a job and return job_id.
        prompt can be str (plaintext) or bytes (pre-encrypted).
        """
        if isinstance(prompt, str):
            encrypted_input = prompt.encode("utf-8")
        else:
            encrypted_input = prompt
        max_tokens_u56 = min(max_tokens, 2**56 - 1)

        receipt = self._build_and_send(
            self.contract.functions.postJob(model, encrypted_input, max_tokens_u56),
            value=JOB_FEE,
        )

        for log_entry in receipt.logs:
            try:
                evt = self.contract.events.JobPosted().process_log(log_entry)
                return int(evt.args.jobId)
            except Exception:
                continue
        raise RuntimeError("Could not parse JobPosted event")

    def get_job(self, job_id: int) -> dict:
        raw = self.contract.functions.getJob(job_id).call()
        return {
            "requester":       raw[0],
            "maxOutputTokens": int(raw[1]),
            "status":          int(raw[2]),
            "miner":           raw[3],
            "postedAt":        int(raw[4]),
            "challenger":      raw[5],
            "submittedAt":     int(raw[6]),
            "inputRef":        raw[7].hex(),
            "outputRef":       raw[8].hex(),
            "modelId":         raw[9],
        }

    def get_job_info(self, job_id: int) -> JobInfo:
        """Return a rich JobInfo with prompt/output from event logs."""
        raw = self.get_job(job_id)
        prompt = self._fetch_input(job_id)
        output = self._fetch_output(job_id) if raw["status"] >= 2 else None
        return JobInfo(
            job_id=job_id,
            requester=raw["requester"],
            miner=raw["miner"],
            challenger=raw["challenger"],
            status=raw["status"],
            status_label=JOB_STATUS.get(raw["status"], "Unknown"),
            model_id=raw["modelId"],
            max_output_tokens=raw["maxOutputTokens"],
            posted_at=raw["postedAt"],
            submitted_at=raw["submittedAt"],
            input_ref=raw["inputRef"],
            output_ref=raw["outputRef"],
            prompt=prompt,
            output=output,
        )

    # ── Claim / Submit / Finalize (for miners) ────────────────────────────

    def claim_job(self, job_id: int) -> str:
        """Claim a job as a miner. Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.claimJob(job_id),
            value=MINER_BOND,
        )
        return receipt.transactionHash.hex()

    def submit_result(self, job_id: int, output: str | bytes) -> str:
        """Submit inference output for a claimed job. Returns tx hash."""
        if isinstance(output, str):
            output_bytes = output.encode("utf-8")
        else:
            output_bytes = output
        receipt = self._build_and_send(
            self.contract.functions.submitResult(job_id, output_bytes),
        )
        return receipt.transactionHash.hex()

    def finalize_job(self, job_id: int) -> str:
        """Finalize a job after challenge window. Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.finalizeJob(job_id),
        )
        return receipt.transactionHash.hex()

    def challenge_result(self, job_id: int) -> str:
        """Challenge a submitted result. Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.challengeResult(job_id),
            value=MINER_BOND,
        )
        return receipt.transactionHash.hex()

    def reclaim_expired_job(self, job_id: int) -> str:
        """Reclaim ETH from an expired job (requester only). Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.reclaimExpiredJob(job_id),
        )
        return receipt.transactionHash.hex()

    # ── Miners and models ─────────────────────────────────────────────────

    def register_miner(
        self,
        models: list[str],
        public_key_der: Optional[bytes] = None,
    ) -> str:
        """Register as a miner on-chain. Returns tx hash."""
        pub_key = public_key_der or self._rsa_pub or b""
        receipt = self._build_and_send(
            self.contract.functions.registerMiner(models, pub_key),
        )
        return receipt.transactionHash.hex()

    def deactivate_miner(self) -> str:
        """Deactivate yourself as a miner. Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.deactivateMiner(),
        )
        return receipt.transactionHash.hex()

    def register_model(self, model_id: str, royalty_bps: int = 0) -> str:
        """Register a model as a creator. Returns tx hash."""
        receipt = self._build_and_send(
            self.contract.functions.registerModel(model_id, royalty_bps),
        )
        return receipt.transactionHash.hex()

    def get_miner_profile(self, address: str) -> dict:
        addr = Web3.to_checksum_address(address)
        raw = self.contract.functions.getMinerProfile(addr).call()
        return {
            "models":    raw[0],
            "pub_key":   raw[1],
            "active":    raw[2],
            "registered_at": raw[3],
        }

    def get_miners_for_model(self, model_id: str, limit: int = 10) -> list[str]:
        raw = self.contract.functions.minersForModel(model_id, limit).call()
        return [addr for addr in raw if addr != "0x0000000000000000000000000000000000000000"]

    def get_miner_count(self) -> int:
        return self.contract.functions.getRegisteredMinerCount().call()

    # ── Fetch output from event log ───────────────────────────────────────

    def _fetch_input(self, job_id: int) -> Optional[str]:
        """Retrieve prompt from the JobPosted event log."""
        try:
            events = self.contract.events.JobPosted.get_logs(
                argument_filters={"jobId": job_id},
                fromBlock=0,
            )
            if events:
                raw_bytes = events[0].args.encryptedInput
                return raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            log.warning("Could not fetch prompt from logs: %s", exc)
        return None

    def _fetch_output(self, job_id: int, decrypt: bool = False) -> str:
        """Retrieve output from the ResultSubmitted event log."""
        try:
            events = self.contract.events.ResultSubmitted.get_logs(
                argument_filters={"jobId": job_id},
                fromBlock=0,
            )
            if events:
                raw_bytes = events[0].args.encryptedOutput
                if decrypt and self._rsa_priv:
                    try:
                        raw_bytes = rsa_decrypt(self._rsa_priv, raw_bytes)
                    except Exception:
                        log.warning("RSA decrypt failed for job %s, returning raw", job_id)
                return raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            log.warning("Could not fetch output from logs: %s", exc)
        return ""

    # ── Introspection ─────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        count = self.contract.functions.getModelCount().call()
        return [self.contract.functions.modelList(i).call() for i in range(count)]

    def get_model_info(self, model_id: str) -> dict:
        raw = self.contract.functions.getModelInfo(model_id).call()
        return {
            "creator":     raw[0],
            "royalty_bps": raw[1],
            "exists":      raw[2],
        }

    def miner_stats(self, address: str) -> dict:
        addr = Web3.to_checksum_address(address)
        return {
            "address":      addr,
            "reputation":   self.contract.functions.minerReputation(addr).call(),
            "total_earned": Web3.from_wei(
                self.contract.functions.totalEarned(addr).call(), "ether"
            ),
            "inft_balance": Web3.from_wei(
                self.contract.functions.balanceOf(addr).call(), "ether"
            ),
        }

    def model_creator_stats(self, address: str) -> dict:
        addr = Web3.to_checksum_address(address)
        return {
            "address":              addr,
            "total_creator_earned": Web3.from_wei(
                self.contract.functions.totalCreatorEarned(addr).call(), "ether"
            ),
        }

    def token_stats(self) -> dict:
        return {
            "total_minted":      Web3.from_wei(self.contract.functions.totalMinted().call(), "ether"),
            "max_supply":        "21000000",
            "jobs_completed":    self.contract.functions.jobsCompleted().call(),
            "next_job_id":       self.contract.functions.nextJobId().call(),
            "protocol_fee_bps":  self.contract.functions.protocolFeeBps().call(),
            "model_count":       self.contract.functions.getModelCount().call(),
            "registered_miners": self.contract.functions.getRegisteredMinerCount().call(),
            "protocol_fees_eth": Web3.from_wei(
                self.contract.functions.protocolFees().call(), "ether"
            ),
        }

    # ── Reward calculation ────────────────────────────────────────────────

    def _calc_reward(self, max_tokens: int) -> int:
        completed = self.contract.functions.jobsCompleted().call()
        halvings  = min(completed // 1_000_000, 20)
        if   max_tokens < 512:  base = 10
        elif max_tokens < 2048: base = 25
        else:                    base = 50
        return base >> halvings
