"""
Persistent KV cache manager for context prefixes (Phase 4).

Miners use llama-cli's --prompt-cache flag to save KV activations after
processing a context prefix. This file manages those cache files:
  - lookup(context_hash, model_id) → path if cached, else None
  - register(context_hash, model_id, path) → record a newly saved file
  - invalidate(context_hash) → evict all model variants for this context
  - evict_expired(max_age_s) → clean up stale files

Cache files are stored as:
  {cache_dir}/{model_slug}/{context_hash[:24]}.bin

The model_slug is the model_id with slashes replaced by underscores.

Cache hit: coordinator miner reports cache_hit=True in ContextLoadResult.
The sequencer records this in the CONTEXT_LOAD_COMMIT tx and skips the
context load wait on future jobs with the same context_hash (the event
fires immediately when a cache-hit coordinator responds).
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("chain.shard.context_cache")


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)[:64]


class ContextKVCache:
    """
    File-based KV cache directory for prompt-cache files produced by llama-cli.

    Thread-safety: asyncio single-threaded — all mutations happen in the event
    loop; no locking needed.
    """

    def __init__(self, cache_dir: str = "/tmp/inft_kv", ttl_s: int = 3600):
        self._root  = Path(cache_dir)
        self._ttl_s = ttl_s
        # In-memory index: (context_hash, model_id) → abs path
        self._index: dict[tuple[str, str], str] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        self._scan_existing()

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, context_hash: str, model_id: str) -> Optional[str]:
        """Return path to existing cache file, or None if not cached / expired."""
        key = (context_hash, model_id)
        path = self._index.get(key)
        if path is None:
            # Try scanning disk in case another process wrote it
            path = self._disk_path(context_hash, model_id)
            if Path(path).exists():
                self._index[key] = path
            else:
                return None
        if not Path(path).exists():
            self._index.pop(key, None)
            return None
        # Check TTL
        age = time.time() - Path(path).stat().st_mtime
        if age > self._ttl_s:
            log.info("kv_cache_expired context=%s model=%s age=%.0fs", context_hash[:12], model_id, age)
            self._evict_file(path, key)
            return None
        return path

    def register(self, context_hash: str, model_id: str, path: str) -> None:
        """Record that path is the KV cache file for (context_hash, model_id)."""
        self._index[(context_hash, model_id)] = path
        log.info("kv_cache_registered context=%s model=%s path=%s",
                 context_hash[:12], model_id, path)

    def cache_path(self, context_hash: str, model_id: str) -> str:
        """Return the canonical path for a cache file (may not exist yet)."""
        return self._disk_path(context_hash, model_id)

    def invalidate(self, context_hash: str) -> int:
        """Delete all cache files for context_hash across all models. Returns count."""
        evicted = 0
        keys_to_remove = [k for k in self._index if k[0] == context_hash]
        for key in keys_to_remove:
            path = self._index.pop(key)
            evicted += self._evict_file(path, key)
        # Also scan disk for orphaned files (index may have been cold)
        for model_dir in self._root.iterdir():
            if not model_dir.is_dir():
                continue
            fname = f"{context_hash[:24]}.bin"
            candidate = model_dir / fname
            if candidate.exists():
                try:
                    candidate.unlink()
                    evicted += 1
                    log.info("kv_cache_invalidated path=%s", candidate)
                except OSError:
                    pass
        return evicted

    def evict_expired(self) -> int:
        """Remove all cache files older than ttl_s. Returns count evicted."""
        cutoff = time.time() - self._ttl_s
        evicted = 0
        for key, path in list(self._index.items()):
            p = Path(path)
            if not p.exists() or p.stat().st_mtime < cutoff:
                evicted += self._evict_file(path, key)
        # Also sweep disk for files not in index
        for model_dir in self._root.iterdir():
            if not model_dir.is_dir():
                continue
            for f in model_dir.glob("*.bin"):
                if f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        evicted += 1
                    except OSError:
                        pass
        if evicted:
            log.info("kv_cache_evict_expired count=%d", evicted)
        return evicted

    def stats(self) -> dict:
        return {
            "entries": len(self._index),
            "cache_dir": str(self._root),
            "ttl_s": self._ttl_s,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _disk_path(self, context_hash: str, model_id: str) -> str:
        slug = _model_slug(model_id)
        model_dir = self._root / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        # Use first 24 chars of context_hash as filename — long enough to avoid collision
        safe_hash = context_hash.lstrip("0x")[:24]
        return str(model_dir / f"{safe_hash}.bin")

    def _evict_file(self, path: str, key: tuple[str, str]) -> int:
        self._index.pop(key, None)
        try:
            Path(path).unlink(missing_ok=True)
            log.info("kv_cache_evicted path=%s", path)
            return 1
        except OSError:
            return 0

    def _scan_existing(self) -> None:
        """Index any cache files already on disk from a previous run."""
        if not self._root.exists():
            return
        for model_dir in self._root.iterdir():
            if not model_dir.is_dir():
                continue
            for f in model_dir.glob("*.bin"):
                # We don't know the full context_hash or model_id from filename alone,
                # so we just record the path by its partial hash key.
                # Full lookup will still work because lookup() checks _disk_path().
                pass
        log.debug("kv_cache_scanned dir=%s", self._root)
