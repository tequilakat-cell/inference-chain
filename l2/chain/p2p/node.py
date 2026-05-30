"""
P2P gossip node.

Asyncio WebSocket-based. Each node maintains persistent connections to
known peers and fans out messages to all connected peers.
Topics: blocks, shard_offers, shard_results, speculative_draft, heartbeats.

A simple seen-cache (LRU) prevents duplicate message processing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Callable, Coroutine, Optional

import aiohttp
from aiohttp import web

from .messages import Envelope, TOPICS
from .discovery import PeerDiscovery

log = logging.getLogger("chain.p2p.node")


class LRUCache:
    def __init__(self, max_size: int = 10_000):
        self._d: OrderedDict[str, bool] = OrderedDict()
        self._max = max_size

    def contains(self, key: str) -> bool:
        if key in self._d:
            self._d.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        self._d[key] = True
        self._d.move_to_end(key)
        if len(self._d) > self._max:
            self._d.popitem(last=False)


class P2PNode:
    def __init__(
        self,
        host:             str = "0.0.0.0",
        port:             int = 9000,
        bootstrap_peers:  list[str] = None,
        privkey:          str = "",
        sender_address:   str = "",
    ):
        self.host    = host
        self.port    = port
        self._privkey = privkey
        self._sender  = sender_address

        # topic → list of async handler coroutines
        self._handlers: dict[str, list[Callable]] = {t: [] for t in TOPICS.values()}

        # Connected outbound WebSocket connections: peer_url → ws
        self._connections: dict[str, aiohttp.ClientWebSocketResponse] = {}

        # Inbound WebSocket connections (managed by aiohttp server)
        self._inbound: set[web.WebSocketResponse] = set()

        # Dedup cache — prevents processing the same message twice
        self._seen = LRUCache(10_000)

        self.discovery = PeerDiscovery(bootstrap_peers or [])
        self._session: Optional[aiohttp.ClientSession] = None
        # Optional callback fired when a new inbound peer connects (used for job recovery)
        self._on_peer_connected: Optional[Callable] = None

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        # Connect to bootstrap peers
        for peer in self.discovery.active_peers():
            asyncio.create_task(self._connect_peer(peer))
        # Start discovery refresh
        asyncio.create_task(self.discovery.run(self))
        log.info("p2p_node_started host=%s port=%d peers=%d",
                 self.host, self.port, len(self.discovery.active_peers()))

    async def serve(self) -> None:
        """Start the inbound WebSocket server."""
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info("p2p_server_listening port=%d", self.port)

    # ── Publish ───────────────────────────────────────────────────────────────

    async def broadcast(self, topic: str, payload: dict) -> None:
        """
        Broadcast a message to all connected peers and local subscribers.
        Signs the message if a private key is configured.
        """
        from .messages import make_envelope
        env = make_envelope(topic, payload, privkey=self._privkey, sender=self._sender)
        msg_json = env.to_json()
        # Use the full envelope JSON (includes timestamp) so periodic heartbeats
        # and other repeated messages are not silently dropped after the first send.
        msg_key = hashlib.sha256(msg_json.encode()).hexdigest()

        if self._seen.contains(msg_key):
            return
        self._seen.add(msg_key)

        # Deliver to local subscribers first
        await self._deliver(env)

        # Fan out to all peers
        dead = []
        for peer_url, ws in list(self._connections.items()):
            try:
                await ws.send_str(msg_json)
            except Exception as exc:
                log.debug("peer_send_failed peer=%s err=%s", peer_url, exc)
                dead.append(peer_url)

        for peer_url in dead:
            self._connections.pop(peer_url, None)

        # Also fan out to inbound connections
        dead_inbound = []
        for ws in list(self._inbound):
            try:
                await ws.send_str(msg_json)
            except Exception:
                dead_inbound.append(ws)
        for ws in dead_inbound:
            self._inbound.discard(ws)

    # ── Subscribe ─────────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: Callable) -> None:
        """Register an async handler for messages on a given topic."""
        self._handlers.setdefault(topic, []).append(handler)

    # ── Peer management ───────────────────────────────────────────────────────

    async def _connect_peer(self, peer_url: str) -> None:
        if peer_url in self._connections:
            return
        try:
            ws = await self._session.ws_connect(peer_url + "/ws", heartbeat=15)
            self._connections[peer_url] = ws
            log.info("peer_connected addr=%s", peer_url)
            asyncio.create_task(self._receive_loop(peer_url, ws))
        except Exception as exc:
            log.debug("peer_connect_failed addr=%s err=%s", peer_url, exc)

    async def ping(self, peer_url: str) -> None:
        """Ping a peer. Raises if unreachable."""
        ws = self._connections.get(peer_url)
        if ws and not ws.closed:
            await ws.ping()
        else:
            await self._connect_peer(peer_url)

    # ── Message receiving ─────────────────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        self._inbound.add(ws)
        log.info("inbound_peer_connected remote=%s", request.remote)
        # Trigger recovery of any pending jobs that lost their miner
        if self._on_peer_connected:
            asyncio.create_task(self._on_peer_connected())
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._on_raw(msg.data)
        self._inbound.discard(ws)
        return ws

    async def _receive_loop(
        self, peer_url: str, ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._on_raw(msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
        except Exception as exc:
            log.debug("receive_loop_error peer=%s err=%s", peer_url, exc)
        finally:
            self._connections.pop(peer_url, None)
            log.debug("peer_disconnected addr=%s", peer_url)

    async def _on_raw(self, raw: str) -> None:
        try:
            d = json.loads(raw)
            env = Envelope.from_dict(d)
        except Exception:
            return

        # Dedup by hash of full raw message
        msg_key = hashlib.sha256(raw.encode()).hexdigest()
        if self._seen.contains(msg_key):
            return
        self._seen.add(msg_key)

        # Verify signature
        if not env.verify():
            log.warning("invalid_signature msg_type=%s sender=%s", env.msg_type, env.sender[:10])
            return

        await self._deliver(env)

        # Re-gossip to other peers (simple flood — replace with epidemic protocol for scale)
        dead = []
        for peer_url, ws in list(self._connections.items()):
            try:
                await ws.send_str(raw)
            except Exception:
                dead.append(peer_url)
        for peer_url in dead:
            self._connections.pop(peer_url, None)

    async def _deliver(self, env: Envelope) -> None:
        for handler in self._handlers.get(env.msg_type, []):
            try:
                await handler(env.payload)
            except Exception as exc:
                log.error("handler_error topic=%s err=%s", env.msg_type, exc, exc_info=True)
