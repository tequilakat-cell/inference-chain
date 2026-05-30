"""
Peer discovery.

MVP: static bootstrap peer list from genesis.json / environment variable.
Maintains active peer set via periodic ping/pong over WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

log = logging.getLogger("chain.p2p.discovery")


class PeerDiscovery:
    def __init__(self, bootstrap_peers: list[str], ping_interval: float = 10.0):
        """
        Args:
            bootstrap_peers: List of ws://host:port strings.
            ping_interval:   How often to ping known peers (seconds).
        """
        self._bootstrap  = list(bootstrap_peers)
        self._active:    set[str] = set(bootstrap_peers)
        self._failed:    dict[str, int] = {}   # peer → consecutive failure count
        self._interval   = ping_interval
        self._max_failures = 3

    async def run(self, node) -> None:
        """Continuously refresh the active peer set."""
        while True:
            await asyncio.sleep(self._interval)
            await self._refresh(node)

    async def _refresh(self, node) -> None:
        for peer in list(self._active):
            try:
                await node.ping(peer)
                self._failed.pop(peer, None)
            except Exception:
                n = self._failed.get(peer, 0) + 1
                self._failed[peer] = n
                if n >= self._max_failures:
                    self._active.discard(peer)
                    log.info("peer_removed addr=%s (failed %d times)", peer, n)

        # Re-add bootstrap peers that may have recovered
        for peer in self._bootstrap:
            if peer not in self._active:
                self._active.add(peer)

    def active_peers(self) -> list[str]:
        return list(self._active)

    def add_peer(self, addr: str) -> None:
        self._active.add(addr)
        log.info("peer_added addr=%s", addr)

    def remove_peer(self, addr: str) -> None:
        self._active.discard(addr)
