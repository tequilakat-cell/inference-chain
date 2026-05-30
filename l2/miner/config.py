"""L2MinerConfig — extends L1 MinerConfig with L2-specific fields."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "l1" / "miner"))
from miner import MinerConfig


@dataclass
class L2MinerConfig:
    # ── Inherited L1 fields ───────────────────────────────────────────────────
    private_key:       str
    rpc_url:           str
    contract_address:  str
    contract_abi:      list
    models:            dict           # hf_model_id → /path/to/model.gguf
    key_dir:           str = "~/.inference-miner/keys"
    max_jobs:          int = 2
    challenge_wait:    int = 602
    health_port:       int = 9090
    log_level:         str = "INFO"
    encryption_enabled: bool = False
    backend:           str = ""

    # ── L2-specific fields ────────────────────────────────────────────────────
    l2_rpc_url:          str = "ws://127.0.0.1:8546"
    l2_chain_id:         int = 2026
    l2_private_key:      str = ""    # defaults to private_key if empty
    bootstrap_peers:     list = field(default_factory=list)
    p2p_host:            str = "0.0.0.0"
    p2p_port:            int = 9001
    min_stake_inft:      int = 100
    max_concurrent_shards: int = 4
    shard_timeout_ms:    int = 35_000

    def to_l1_config(self) -> MinerConfig:
        """Convert to the MinerConfig expected by the L1 Miner parent class."""
        from pathlib import Path
        return MinerConfig(
            private_key=self.private_key,
            rpc_url=self.rpc_url,
            contract_address=self.contract_address,
            contract_abi=self.contract_abi,
            models=self.models,
            key_dir=Path(self.key_dir),
            max_jobs=self.max_jobs,
            challenge_wait=self.challenge_wait,
            health_port=self.health_port,
            log_level=self.log_level,
            encryption_enabled=self.encryption_enabled,
            backend=self.backend,
        )


def load_l2_config(path: str) -> L2MinerConfig:
    with open(path) as f:
        raw = json.load(f)

    # Load ABI from deployment.json if abi field is a file path
    abi = raw.get("abi", [])
    if isinstance(abi, str):
        dep_path = Path(path).parent / abi
        with open(dep_path) as f:
            dep = json.load(f)
        abi = dep.get("abi", dep) if isinstance(dep, dict) else dep

    return L2MinerConfig(
        private_key=os.environ.get("PRIVATE_KEY", raw.get("private_key", "")),
        rpc_url=os.environ.get("L1_RPC_URL", raw.get("rpc_url", "")),
        contract_address=os.environ.get("L1_CONTRACT_ADDRESS", raw.get("contract_address", "")),
        contract_abi=abi,
        models=raw.get("models", {}),
        key_dir=raw.get("key_dir", "~/.inference-miner/keys"),
        max_jobs=int(raw.get("max_jobs", 2)),
        challenge_wait=int(raw.get("challenge_wait", 602)),
        health_port=int(raw.get("health_port", 9090)),
        log_level=raw.get("log_level", "INFO"),
        encryption_enabled=raw.get("encryption_enabled", False),
        backend=raw.get("backend", ""),
        l2_rpc_url=os.environ.get("L2_RPC_URL", raw.get("l2_rpc_url", "http://127.0.0.1:8545")),
        l2_chain_id=int(raw.get("l2_chain_id", 2026)),
        l2_private_key=os.environ.get("L2_PRIVATE_KEY", raw.get("l2_private_key", "")),
        bootstrap_peers=raw.get("bootstrap_peers", []),
        p2p_host=raw.get("p2p_host", "0.0.0.0"),
        p2p_port=int(raw.get("p2p_port", 9001)),
        min_stake_inft=int(raw.get("min_stake_inft", 100)),
        max_concurrent_shards=int(raw.get("max_concurrent_shards", 4)),
        shard_timeout_ms=int(raw.get("shard_timeout_ms", 35_000)),
    )
