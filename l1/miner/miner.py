"""
InferenceToken Miner Node
=========================
Serves AI inference jobs from the on-chain marketplace, earns INFT tokens.
Auto-detects best GPU backend: CUDA (NVIDIA), MLX (Apple Silicon), Vulkan (AMD/Intel).

Configuration via environment variables (preferred) or config.json:
  PRIVATE_KEY           Ethereum wallet private key (0x...)
  RPC_URL               JSON-RPC endpoint (e.g. https://sepolia.base.org)
  CONTRACT_ADDRESS      Deployed InferenceToken address
  MODELS_DIR            Directory containing GGUF model files
  KEY_DIR               Directory for RSA keypair (default ~/.inference-miner/keys)
  MAX_JOBS              Max concurrent jobs (default 3)
  CHALLENGE_WAIT        Seconds to wait before finalising (default 602)
  N_GPU_LAYERS          GPU layers for llama.cpp (default -1 = all)
  N_CTX                 Context window size (default 4096)
  HEALTH_PORT           Port for health + metrics server (default 9090)
  LOG_LEVEL             DEBUG / INFO / WARNING (default INFO)
  ENCRYPTION_ENABLED    true/false – encrypt prompts/outputs (default false for MVP)
  BACKEND               Force GPU backend: cuda, mlx, vulkan, llama, mock (default auto-detect)

Usage:
    pip install -r requirements.txt
    PRIVATE_KEY=0x... RPC_URL=https://... CONTRACT_ADDRESS=0x... python miner.py
    # OR
    python miner.py --config config.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aiohttp import web
from eth_account import Account
from eth_account.signers.local import LocalAccount
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

import keys
import nonce as nonce_mod

# GPU inference backends (CUDA, MLX, Vulkan, llama.cpp)
from backends import get_backend, get_available_backends
from backends.base import InferenceBackend

log = logging.getLogger("miner")

# ── Prometheus metrics ────────────────────────────────────────────────────────

JOBS_SEEN      = Counter("miner_jobs_seen_total",      "Jobs seen on-chain")
JOBS_CLAIMED   = Counter("miner_jobs_claimed_total",   "Jobs successfully claimed")
JOBS_COMPLETED = Counter("miner_jobs_completed_total", "Jobs finalized (tokens earned)")
JOBS_FAILED    = Counter("miner_jobs_failed_total",    "Jobs that errored")
INFT_EARNED    = Gauge  ("miner_inft_balance",         "Current INFT token balance")
ETH_BALANCE    = Gauge  ("miner_eth_balance_wei",      "Current ETH balance in wei")
INFER_SECONDS  = Histogram("miner_inference_seconds",  "Inference wall-clock time",
                           buckets=[1, 2, 5, 10, 30, 60, 120, 300])

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MinerConfig:
    private_key:         str
    rpc_url:             str
    contract_address:    str
    contract_abi:        list
    models:              dict[str, str]   # hf_model_id → /path/to/model.gguf
    key_dir:             Path  = field(default_factory=lambda: Path.home() / ".inference-miner" / "keys")
    max_jobs:            int   = 3
    challenge_wait:      int   = 602
    n_gpu_layers:        int   = -1
    n_ctx:               int   = 4096
    health_port:         int   = 9090
    log_level:           str   = "INFO"
    encryption_enabled:  bool  = False
    poll_interval:       int   = 3
    backend:             str   = ""  # auto-detect by default


def load_config(args: argparse.Namespace) -> MinerConfig:
    abi: list = []
    raw: dict = {}

    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        abi_src = raw.get("abi")
        if isinstance(abi_src, str):
            config_dir = Path(args.config).parent
            loaded = json.load(open((config_dir / abi_src).resolve()))
            abi = loaded["abi"] if isinstance(loaded, dict) else loaded
        else:
            abi = abi_src or []
    else:
        dep_path = Path("deployment.json")
        if dep_path.exists():
            loaded = json.load(open(dep_path))
            abi = loaded.get("abi", [])

    # Helper: env var takes precedence over config file
    def env(key: str, default: str = "") -> str:
        return os.environ.get(key, raw.get(key.lower(), default) if args.config else default)

    models_dir = env("MODELS_DIR", "models")
    models: dict[str, str] = {}
    if args.config and "models" in raw:
        models = raw["models"]
    elif Path(models_dir).is_dir():
        for f in Path(models_dir).glob("*.gguf"):
            models[f.stem] = str(f)

    return MinerConfig(
        private_key=os.environ.get("PRIVATE_KEY") or raw.get("private_key", ""),
        rpc_url=os.environ.get("RPC_URL") or raw.get("rpc_url", ""),
        contract_address=os.environ.get("CONTRACT_ADDRESS") or raw.get("contract_address", ""),
        contract_abi=abi,
        models=models,
        key_dir=Path(os.environ.get("KEY_DIR", raw.get("key_dir", str(Path.home() / ".inference-miner" / "keys")))),
        max_jobs=int(os.environ.get("MAX_JOBS", raw.get("max_jobs", 3))),
        challenge_wait=int(os.environ.get("CHALLENGE_WAIT", raw.get("challenge_wait", 602))),
        n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", raw.get("n_gpu_layers", -1))),
        n_ctx=int(os.environ.get("N_CTX", raw.get("n_ctx", 4096))),
        health_port=int(os.environ.get("HEALTH_PORT", raw.get("health_port", 9090))),
        log_level=os.environ.get("LOG_LEVEL", raw.get("log_level", "INFO")),
        encryption_enabled=os.environ.get("ENCRYPTION_ENABLED", str(raw.get("encryption_enabled", False))).lower() == "true",
        backend=os.environ.get("BACKEND", raw.get("backend", "")),
    )


# ── GPU Backend using auto-detect system ─────────────────────────────────────

class GPUModelPool:
    """
    Auto-detects the best GPU inference backend:
    1. CUDA  — NVIDIA GPU (transformers + PyTorch)
    2. MLX   — Apple Silicon (mlx-lm)
    3. Vulkan — AMD/Intel/NVIDIA (llama.cpp + Vulkan)
    4. llama.cpp — CPU fallback
    5. mock — No hardware available
    """
    def __init__(self, model_map: dict[str, str], preferred_backend: str = ""):
        self._map       = model_map
        self._backend: InferenceBackend = get_backend(
            model_map, preferred=preferred_backend or None
        )
        log.info("gpu_backend_selected backend=%s info=%s",
                 self._backend.name, self._backend.info)

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    async def load(self, model_id: str) -> None:
        await self._backend.load(model_id)

    async def run(self, model_id: str, prompt: str, max_tokens: int) -> str:
        if not self.supports(model_id):
            raise RuntimeError(f"Unsupported model: {model_id}")
        return await self._backend.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=min(max_tokens, 2048),
            temperature=0.7,
        )


# ── Chain client ──────────────────────────────────────────────────────────────

class ChainClient:
    def __init__(self, cfg: MinerConfig):
        self.w3 = Web3(Web3.HTTPProvider(cfg.rpc_url, request_kwargs={"timeout": 30}))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.account: LocalAccount = Account.from_key(cfg.private_key)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(cfg.contract_address),
            abi=cfg.contract_abi,
        )
        self.nonces = nonce_mod.NonceManager(self.w3, self.account.address)
        log.info("chain_connected chain_id=%s address=%s",
                 self.w3.eth.chain_id, self.account.address)

    async def _build_tx(self, fn, value: int = 0) -> dict:
        latest = self.w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", self.w3.eth.gas_price)
        priority = self.w3.to_wei("0.01", "gwei")
        max_fee  = base_fee * 2 + priority
        return fn.build_transaction({
            "from":                 self.account.address,
            "value":                value,
            "nonce":                await self.nonces.get(),
            "gas":                  500_000,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority,
            "type":                 "0x2",
        })

    async def send(self, fn, value: int = 0, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                tx     = await self._build_tx(fn, value)
                signed = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt.status != 1:
                    raise RuntimeError(f"tx reverted: {tx_hash.hex()}")
                return tx_hash.hex()
            except Exception as exc:
                log.warning("tx_failed attempt=%d/%d err=%s", attempt + 1, retries, exc)
                await self.nonces.reset()
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def eth_balance(self) -> int:
        return self.w3.eth.get_balance(self.account.address)

    def inft_balance(self) -> int:
        return self.contract.functions.balanceOf(self.account.address).call()

    def get_job(self, job_id: int) -> dict:
        raw = self.contract.functions.getJob(job_id).call()
        return {
            "requester":       raw[0],
            "maxOutputTokens": raw[1],
            "status":          raw[2],
            "miner":           raw[3],
            "postedAt":        raw[4],
            "challenger":      raw[5],
            "submittedAt":     raw[6],
            "inputRef":        raw[7].hex(),
            "outputRef":       raw[8].hex(),
            "modelId":         raw[9],
        }

    def next_job_id(self) -> int:
        return self.contract.functions.nextJobId().call()

    def jobs_completed(self) -> int:
        return self.contract.functions.jobsCompleted().call()

    def get_job_posted_event(self, job_id: int) -> Optional[bytes]:
        events = self.contract.events.JobPosted.get_logs(
            fromBlock=0,
            argument_filters={"jobId": job_id},
        )
        if events:
            return events[0].args.encryptedInput
        return None


# ── Health + metrics server ───────────────────────────────────────────────────

async def health_server(miner: "Miner", port: int) -> None:
    async def handle_health(request: web.Request) -> web.Response:
        bal = miner.chain.eth_balance()
        return web.json_response({
            "status":      "ok",
            "address":     miner.chain.account.address,
            "eth_wei":     bal,
            "active_jobs": len(miner._active_jobs),
            "models":      list(miner.cfg.models.keys()),
            "backend":     miner.models._backend.name,
            "backend_info": miner.models._backend.info,
        })

    async def handle_metrics(request: web.Request) -> web.Response:
        try:
            INFT_EARNED.set(miner.chain.inft_balance())
            ETH_BALANCE.set(miner.chain.eth_balance())
        except Exception:
            pass
        return web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)

    app = web.Application()
    app.router.add_get("/health",  handle_health)
    app.router.add_get("/metrics", handle_metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("health_server_started port=%d", port)


# ── Miner ─────────────────────────────────────────────────────────────────────

class Miner:
    def __init__(self, cfg: MinerConfig, priv_pem: bytes, pub_der: bytes):
        self.cfg       = cfg
        self.chain     = ChainClient(cfg)
        self.models    = GPUModelPool(cfg.models, cfg.backend)
        self._priv_pem = priv_pem
        self._pub_der  = pub_der
        self._active_jobs: set[int] = set()
        self._seen_jobs:   set[int] = set()
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        await asyncio.gather(
            self._register_on_chain(),
            health_server(self, self.cfg.health_port),
        )
        log.info("miner_ready address=%s models=%s backend=%s",
                 self.chain.account.address, list(self.cfg.models),
                 self.models._backend.name)

    async def _register_on_chain(self) -> None:
        pub_hex = bytes.fromhex(keys.pubkey_hex(self._pub_der).lstrip("0x"))
        try:
            tx = await self.chain.send(
                self.chain.contract.functions.registerMiner(
                    list(self.cfg.models.keys()), pub_hex
                )
            )
            log.info("registered_on_chain tx=%s", tx[:12] + "…")
        except Exception as exc:
            log.warning("register_skipped reason=%s", exc)

    async def run(self) -> None:
        await self.start()
        while not self._shutdown.is_set():
            try:
                await self._scan()
            except Exception as exc:
                log.error("scan_error err=%s", exc, exc_info=True)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.cfg.poll_interval
                )
            except asyncio.TimeoutError:
                pass

    async def _scan(self) -> None:
        next_id = self.chain.next_job_id()
        for job_id in range(next_id):
            if job_id in self._active_jobs or job_id in self._seen_jobs:
                continue
            if len(self._active_jobs) >= self.cfg.max_jobs:
                break

            job = self.chain.get_job(job_id)
            if job["status"] == 0 and self.models.supports(job["modelId"]):
                JOBS_SEEN.inc()
                log.info("job_found job_id=%d model=%s", job_id, job["modelId"])
                self._active_jobs.add(job_id)
                asyncio.create_task(self._process(job_id, job))
            elif job["status"] in (3, 4, 5):
                self._seen_jobs.add(job_id)

    async def _process(self, job_id: int, job: dict) -> None:
        try:
            await self._do_process(job_id, job)
            JOBS_COMPLETED.inc()
        except Exception as exc:
            JOBS_FAILED.inc()
            log.error("job_failed job_id=%d err=%s", job_id, exc, exc_info=True)
        finally:
            self._active_jobs.discard(job_id)
            self._seen_jobs.add(job_id)

    async def _do_process(self, job_id: int, job: dict) -> None:
        # 1. Claim
        log.info("claiming job_id=%d", job_id)
        tx = await self.chain.send(
            self.chain.contract.functions.claimJob(job_id),
            value=Web3.to_wei(0.005, "ether"),
        )
        log.info("claimed job_id=%d tx=%s", job_id, tx[:12] + "…")
        JOBS_CLAIMED.inc()

        # 2. Retrieve prompt from event log
        raw_input = self.chain.get_job_posted_event(job_id)
        if raw_input is None:
            raise RuntimeError(f"could not retrieve encryptedInput for job {job_id}")

        # 3. Decrypt prompt (or pass through)
        if self.cfg.encryption_enabled and raw_input:
            try:
                prompt = keys.decrypt(self._priv_pem, raw_input).decode("utf-8")
            except Exception as exc:
                log.error("decrypt_failed job_id=%d err=%s — aborting job", job_id, exc)
                raise RuntimeError(f"failed to decrypt job {job_id}") from exc
        else:
            prompt = raw_input.decode("utf-8", errors="replace")

        # 4. Run inference on GPU backend
        max_tokens = min(int(job["maxOutputTokens"]), 2048)
        log.info("inference_start job_id=%d model=%s max_tokens=%d backend=%s",
                 job_id, job["modelId"], max_tokens, self.models._backend.name)
        t0 = time.monotonic()
        output_text = await self.models.run(job["modelId"], prompt, max_tokens)
        elapsed = time.monotonic() - t0
        INFER_SECONDS.observe(elapsed)
        log.info("inference_done job_id=%d elapsed=%.1fs len=%d",
                 job_id, elapsed, len(output_text))

        # 5. Encrypt output (or pass through)
        if self.cfg.encryption_enabled:
            try:
                req_profile = self.chain.contract.functions.getMinerProfile(
                    job["requester"]
                ).call()
                req_pubkey = req_profile[1]
                if req_pubkey:
                    encrypted_output = keys.encrypt(req_pubkey, output_text.encode())
                else:
                    encrypted_output = output_text.encode()
            except Exception:
                encrypted_output = output_text.encode()
        else:
            encrypted_output = output_text.encode("utf-8")

        # 6. Submit result
        log.info("submitting_result job_id=%d", job_id)
        tx = await self.chain.send(
            self.chain.contract.functions.submitResult(job_id, encrypted_output)
        )
        log.info("result_submitted job_id=%d tx=%s", job_id, tx[:12] + "…")

        # 7. Wait out challenge window
        log.info("challenge_wait job_id=%d seconds=%d", job_id, self.cfg.challenge_wait)
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=self.cfg.challenge_wait)
        except asyncio.TimeoutError:
            pass
        if self._shutdown.is_set():
            log.warning("shutdown_during_wait job_id=%d — will not finalize", job_id)
            return

        # 8. Finalize → tokens minted
        log.info("finalizing job_id=%d", job_id)
        tx = await self.chain.send(
            self.chain.contract.functions.finalizeJob(job_id)
        )
        inft = Web3.from_wei(self.chain.inft_balance(), "ether")
        log.info("finalized job_id=%d tx=%s inft_balance=%s", job_id, tx[:12] + "…", inft)


# ── Periodic stats ────────────────────────────────────────────────────────────

async def stats_loop(miner: Miner) -> None:
    while not miner._shutdown.is_set():
        try:
            inft = Web3.from_wei(miner.chain.inft_balance(), "ether")
            eth  = Web3.from_wei(miner.chain.eth_balance(), "ether")
            rep  = miner.chain.contract.functions.minerReputation(
                miner.chain.account.address
            ).call()
            log.info("stats address=%s eth=%.4f inft=%s rep=%d active_jobs=%d backend=%s",
                     miner.chain.account.address, eth, inft, rep,
                     len(miner._active_jobs), miner.models._backend.name)
        except Exception as exc:
            log.warning("stats_error err=%s", exc)
        await asyncio.sleep(60)


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def install_signal_handlers(miner: Miner, loop: asyncio.AbstractEventLoop) -> None:
    def _handle(sig_name: str) -> None:
        log.info("signal_received sig=%s — shutting down gracefully", sig_name)
        miner._shutdown.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig.name: _handle(s))


# ── Entry point ───────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    cfg = load_config(args)
    setup_logging(cfg.log_level)

    if not cfg.private_key:
        log.error("No PRIVATE_KEY set. Export PRIVATE_KEY=0x... or use --config.")
        sys.exit(1)
    if not cfg.rpc_url:
        log.error("No RPC_URL set.")
        sys.exit(1)
    if not cfg.contract_address:
        log.error("No CONTRACT_ADDRESS set.")
        sys.exit(1)

    # Log available backends at startup
    available = get_available_backends()
    log.info("available_gpu_backends count=%d list=%s", len(available), available)

    # Load or generate RSA keypair
    priv_pem, pub_der = keys.load_or_generate(cfg.key_dir)
    log.info("rsa_pubkey_loaded dir=%s", cfg.key_dir)

    miner = Miner(cfg, priv_pem, pub_der)

    loop = asyncio.get_event_loop()
    install_signal_handlers(miner, loop)

    await asyncio.gather(
        miner.run(),
        stats_loop(miner),
    )
    log.info("miner_stopped")


def setup_logging(level: str = "INFO") -> None:
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level":   record.levelname,
                "logger":  record.name,
                "msg":     record.getMessage(),
                **({"exc": self.formatException(record.exc_info)} if record.exc_info else {}),
            })

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        handlers=[handler])


def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceToken miner node")
    parser.add_argument("--config", default="", help="Optional config.json path")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
