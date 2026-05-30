"""
PeerbitClient — Python wrapper around the Peerbit sidecar REST API.

The sidecar runs as a Node.js subprocess on port 7700. This client provides
a clean async Python interface so the L2 miner can:
  - Announce itself to the distributed miner registry on startup
  - Discover other miners by model
  - Publish job records when accepting shards
  - Log reputation events after shard completion
  - Deregister on clean shutdown

If the sidecar is not running, all calls degrade gracefully (log + return None).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger("l2_miner.peerbit")

SIDECAR_PORT = int(os.environ.get("ORBITDB_PORT", "7700"))
SIDECAR_URL  = f"http://127.0.0.1:{SIDECAR_PORT}"
SIDECAR_DIR  = Path(__file__).parent.parent / "peerbit"  # renamed to orbitdb sidecar dir


class PeerbitClient:
    """
    Async client for the Peerbit sidecar REST API.
    All methods are no-ops if the sidecar is unavailable.
    """

    def __init__(self, address: str, models: list[str], backend: str,
                 p2p_addr: str = "", l2_chain_id: int = 2026,
                 max_shards: int = 4):
        self._address      = address
        self._models       = models
        self._backend      = backend
        self._p2p_addr     = p2p_addr
        self._chain_id     = str(l2_chain_id)
        self._max_shards   = max_shards
        self._session: Optional[aiohttp.ClientSession] = None
        self._available    = False
        self._proc: Optional[subprocess.Popen] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, start_sidecar: bool = True) -> None:
        """
        Open HTTP session and optionally start the Node.js sidecar subprocess.
        Registers this miner in the distributed registry.
        """
        if start_sidecar:
            self._proc = await self._start_sidecar()

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        )

        # Wait for sidecar to be ready (up to 30s)
        for _ in range(30):
            try:
                async with self._session.get(f"{SIDECAR_URL}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._available = True
                        log.info("peerbit_connected peer_id=%s", data.get("peerId", "?")[:20])
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        if not self._available:
            log.warning("peerbit_unavailable sidecar did not start in 30s — continuing without it")
            return

        await self.register_miner()

    async def stop(self) -> None:
        """Deregister and close connections."""
        if self._available:
            await self._deregister()
        if self._session:
            await self._session.close()
        if self._proc:
            self._proc.terminate()
            log.info("peerbit_sidecar_stopped")

    # ── Miner registry ────────────────────────────────────────────────────────

    async def register_miner(self) -> None:
        """Announce this miner to the distributed registry."""
        await self._put(f"/miners/{self._address}", {
            "address":   self._address,
            "models":    json.dumps(self._models),
            "backend":   self._backend,
            "p2pAddr":   self._p2p_addr,
            "l2ChainId": self._chain_id,
            "maxShards": self._max_shards,
            "active":    True,
        })
        log.info("peerbit_registered address=%s models=%s", self._address[:12], self._models)

    async def heartbeat(self) -> None:
        """Update last_seen timestamp. Call every 60s."""
        if self._available:
            await self._put(f"/miners/{self._address}", {
                "address":   self._address,
                "models":    json.dumps(self._models),
                "backend":   self._backend,
                "p2pAddr":   self._p2p_addr,
                "l2ChainId": self._chain_id,
                "maxShards": self._max_shards,
                "active":    True,
            })

    async def get_miners_for_model(self, model_id: str) -> list[dict]:
        """Return all active miners that support a given model."""
        return await self._get(f"/miners?model={model_id}") or []

    async def list_miners(self) -> list[dict]:
        return await self._get("/miners") or []

    async def _deregister(self) -> None:
        await self._delete(f"/miners/{self._address}")
        log.info("peerbit_deregistered address=%s", self._address[:12])

    # ── Job board ─────────────────────────────────────────────────────────────

    async def announce_job_accepted(self, job_id: str, model_id: str,
                                     mode: str, n_shards: int) -> None:
        """Announce that this miner accepted a shard for a job."""
        await self._put(f"/jobs/{job_id}", {
            "jobId":    job_id,
            "modelId":  model_id,
            "mode":     mode,
            "nShards":  n_shards,
            "postedAt": int(time.time() * 1000),
            "status":   "partial",
            "requester": "",
        })

    async def announce_job_complete(self, job_id: str, output_hash: str,
                                     latency_ms: int) -> None:
        """Update job record when a shard is successfully completed."""
        await self._put(f"/jobs/{job_id}", {
            "jobId":       job_id,
            "status":      "complete",
            "outputHash":  output_hash,
            "completedAt": int(time.time() * 1000),
            "latencyMs":   latency_ms,
        })

    # ── Reputation events ─────────────────────────────────────────────────────

    async def log_shard_complete(self, job_id: str, shard_idx: int,
                                  latency_ms: int) -> None:
        await self._post("/events", {
            "eventId":   str(uuid.uuid4()),
            "miner":     self._address,
            "eventType": "shard_complete",
            "delta":     1,
            "jobId":     job_id,
            "shardIdx":  shard_idx,
        })

    async def log_shard_failed(self, job_id: str, shard_idx: int) -> None:
        await self._post("/events", {
            "eventId":   str(uuid.uuid4()),
            "miner":     self._address,
            "eventType": "shard_failed",
            "delta":     -5,
            "jobId":     job_id,
            "shardIdx":  shard_idx,
        })

    async def get_reputation_events(self, address: str) -> list[dict]:
        return await self._get(f"/events/{address}") or []

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    async def _get(self, path: str) -> Optional[list | dict]:
        if not self._available or not self._session:
            return None
        try:
            async with self._session.get(SIDECAR_URL + path) as resp:
                return await resp.json()
        except Exception as exc:
            log.debug("peerbit_get_failed path=%s err=%s", path, exc)
            return None

    async def _put(self, path: str, data: dict) -> bool:
        if not self._available or not self._session:
            return False
        try:
            async with self._session.put(SIDECAR_URL + path, json=data) as resp:
                return resp.status == 200
        except Exception as exc:
            log.debug("peerbit_put_failed path=%s err=%s", path, exc)
            return False

    async def _post(self, path: str, data: dict) -> bool:
        if not self._available or not self._session:
            return False
        try:
            async with self._session.post(SIDECAR_URL + path, json=data) as resp:
                return resp.status == 200
        except Exception as exc:
            log.debug("peerbit_post_failed path=%s err=%s", path, exc)
            return False

    async def _delete(self, path: str) -> bool:
        if not self._available or not self._session:
            return False
        try:
            async with self._session.delete(SIDECAR_URL + path) as resp:
                return resp.status == 200
        except Exception as exc:
            log.debug("peerbit_delete_failed path=%s err=%s", path, exc)
            return False

    # ── Sidecar subprocess ────────────────────────────────────────────────────

    async def _start_sidecar(self) -> Optional[subprocess.Popen]:
        node_modules = SIDECAR_DIR / "node_modules"
        if not node_modules.exists():
            log.warning("orbitdb_node_modules_missing — run 'npm install' in %s", SIDECAR_DIR)
            return None

        # Use Node.js 22 from nvm if available (OrbitDB requires v22+)
        nvm_node = Path.home() / ".nvm" / "versions" / "node"
        node_bin = "node"
        if nvm_node.exists():
            node22_bins = sorted(nvm_node.glob("v22*/bin/node"))
            if node22_bins:
                node_bin = str(node22_bins[-1])

        dist_file = SIDECAR_DIR / "dist" / "index.js"
        if not dist_file.exists():
            log.warning("orbitdb_dist_missing — run 'npm run build' in %s", SIDECAR_DIR)
            return None

        env = {**os.environ, "ORBITDB_PORT": str(SIDECAR_PORT)}
        proc = subprocess.Popen(
            [node_bin, str(dist_file)],
            cwd=str(SIDECAR_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("orbitdb_sidecar_started pid=%d port=%d node=%s", proc.pid, SIDECAR_PORT, node_bin)
        return proc
