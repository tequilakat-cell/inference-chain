"""
P2P thought gossip protocol for pg_inft distributed LLM memory rollup.

Message types:
  THOUGHT_BROADCAST    (type=20): gossip a completed inference thought to all peers.
  THOUGHT_SYNC_REQUEST (type=21): request recent thoughts from a peer (cold-start).
  THOUGHT_SYNC_RESPONSE (type=22): response with a batch of thoughts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any

from ..shard.thought_store import ThoughtStore

log = logging.getLogger("thought_protocol")


# ── Message type constants ────────────────────────────────────────────────────

THOUGHT_BROADCAST     = 20
THOUGHT_SYNC_REQUEST  = 21
THOUGHT_SYNC_RESPONSE = 22


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ThoughtBroadcast:
    """
    A single completed inference record broadcast to the P2P network.
    proof_sig is a 0x-prefixed 65-byte ECDSA signature in hex.
    """
    type: int           = THOUGHT_BROADCAST
    job_id: str         = ""
    miner_address: str  = ""
    model_id: str       = ""
    question_text: str  = ""
    thinking_text: str  = ""
    answer_text: str    = ""
    proof_sig: str      = ""      # 0x-prefixed hex, 65 bytes = "0x" + 130 chars
    block_number: int   = 0
    tx_hash: str        = ""
    embedding: list     = field(default_factory=list)  # 768-dim query embedding (pg_inft >=1.4)
    timestamp: float    = field(default_factory=time.time)


@dataclass
class ThoughtSyncRequest:
    """
    Request recent thoughts from a peer (used at cold-start to bootstrap
    the local pg_inft store from a peer that has already synced).
    """
    type: int           = THOUGHT_SYNC_REQUEST
    since_timestamp: float = 0.0    # Unix epoch; 0 = "give me everything"
    limit: int          = 50        # maximum number of thoughts to receive
    model_id: str       = ""        # filter by model (empty = any)
    requester: str      = ""        # requesting peer's address


@dataclass
class ThoughtSyncResponse:
    """
    Batch of thoughts returned in response to a ThoughtSyncRequest.
    thoughts is a list of ThoughtBroadcast dicts (serialised).
    """
    type: int               = THOUGHT_SYNC_RESPONSE
    thoughts: list          = field(default_factory=list)
    responder: str          = ""
    has_more: bool          = False   # True if there are more thoughts beyond `limit`
    next_since_timestamp: float = 0.0


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_json(obj) -> str:
    """Serialise a dataclass instance to a compact JSON string."""
    return json.dumps(asdict(obj), separators=(",", ":"))


def _from_dict_broadcast(data: dict) -> ThoughtBroadcast:
    return ThoughtBroadcast(
        type          = int(data.get("type", THOUGHT_BROADCAST)),
        job_id        = str(data.get("job_id", "")),
        miner_address = str(data.get("miner_address", "")),
        model_id      = str(data.get("model_id", "")),
        question_text = str(data.get("question_text", "")),
        thinking_text = str(data.get("thinking_text", "")),
        answer_text   = str(data.get("answer_text", "")),
        proof_sig     = str(data.get("proof_sig", "")),
        block_number  = int(data.get("block_number", 0)),
        tx_hash       = str(data.get("tx_hash", "")),
        embedding     = list(data.get("embedding", []) or []),
        timestamp     = float(data.get("timestamp", 0.0)),
    )


def _from_dict_sync_request(data: dict) -> ThoughtSyncRequest:
    return ThoughtSyncRequest(
        type              = int(data.get("type", THOUGHT_SYNC_REQUEST)),
        since_timestamp   = float(data.get("since_timestamp", 0.0)),
        limit             = int(data.get("limit", 50)),
        model_id          = str(data.get("model_id", "")),
        requester         = str(data.get("requester", "")),
    )


def _from_dict_sync_response(data: dict) -> ThoughtSyncResponse:
    return ThoughtSyncResponse(
        type                  = int(data.get("type", THOUGHT_SYNC_RESPONSE)),
        thoughts              = list(data.get("thoughts", [])),
        responder             = str(data.get("responder", "")),
        has_more              = bool(data.get("has_more", False)),
        next_since_timestamp  = float(data.get("next_since_timestamp", 0.0)),
    )


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def broadcast_thought(thought: ThoughtBroadcast, peers: list) -> None:
    """
    Serialise `thought` to JSON and send it to all connected peers.

    `peers` is a list of objects that expose a `send(data: str | bytes)` coroutine
    or a `send_message(data: str)` method (as used by the P2PNode websocket layer).
    Errors per-peer are logged and suppressed so a single bad peer does not block
    the broadcast to others.
    """
    payload = _to_json(thought)
    payload_bytes = payload.encode("utf-8")

    for peer in peers:
        try:
            if hasattr(peer, "send"):
                await peer.send(payload_bytes)
            elif hasattr(peer, "send_message"):
                await peer.send_message(payload)
            else:
                log.debug("broadcast_thought: peer %r has no send method", peer)
        except Exception as exc:
            log.debug(
                "broadcast_thought_peer_err peer=%r err=%s", peer, exc
            )


# ── Handle incoming broadcast ─────────────────────────────────────────────────

async def handle_thought_broadcast(
    data: dict,
    peer_addr: str,
    store: ThoughtStore,
) -> None:
    """
    Handle an incoming THOUGHT_BROADCAST message from a peer.

    1. Deserialise the dict into a ThoughtBroadcast.
    2. Call store.ingest(...) to persist into pg_inft.
    3. Call store.record_peer(...) to update peer statistics.

    All errors are logged and suppressed — a bad message never crashes the node.
    """
    try:
        thought = _from_dict_broadcast(data)

        if not thought.job_id:
            log.debug("handle_thought_broadcast: empty job_id from peer=%s — ignored", peer_addr)
            return

        success = await store.ingest(
            job_id        = thought.job_id,
            miner_address = thought.miner_address,
            model_id      = thought.model_id,
            question      = thought.question_text,
            thinking      = thought.thinking_text,
            answer        = thought.answer_text,
            proof_sig_hex = thought.proof_sig,
            block_number  = thought.block_number if thought.block_number else None,
            tx_hash       = thought.tx_hash if thought.tx_hash else None,
            peer_origin   = peer_addr,
        )

        await store.record_peer(
            peer_address = peer_addr,
            job_id       = thought.job_id,
            rejected     = not success,
        )

        # Apply the gossiped embedding so every replica's semantic index matches
        # the producer's (embeddings are not re-computed per node). Done even when
        # ingest reported a duplicate — set_embedding is an idempotent UPDATE keyed
        # by job_id, so it also backfills a thought that arrived before its vector.
        if thought.embedding:
            await store.set_embedding(thought.job_id, thought.embedding)

        if success:
            log.debug(
                "thought_received_ok job=%s model=%s peer=%s",
                thought.job_id[:12], thought.model_id, peer_addr,
            )
        else:
            log.debug(
                "thought_received_rejected job=%s peer=%s",
                thought.job_id[:12], peer_addr,
            )

    except Exception as exc:
        log.warning(
            "handle_thought_broadcast_err peer=%s err=%s", peer_addr, exc
        )


# ── Sync request / response helpers ──────────────────────────────────────────

async def send_sync_request(
    requester_address: str,
    peer: Any,
    since_timestamp: float = 0.0,
    limit: int = 50,
    model_id: str = "",
) -> None:
    """
    Send a THOUGHT_SYNC_REQUEST to a single peer to bootstrap our local store.
    `peer` must expose a `send` or `send_message` coroutine.
    """
    req = ThoughtSyncRequest(
        since_timestamp = since_timestamp,
        limit           = limit,
        model_id        = model_id,
        requester       = requester_address,
    )
    payload = _to_json(req).encode("utf-8")
    try:
        if hasattr(peer, "send"):
            await peer.send(payload)
        elif hasattr(peer, "send_message"):
            await peer.send_message(payload.decode("utf-8"))
    except Exception as exc:
        log.debug("send_sync_request_err peer=%r err=%s", peer, exc)


async def handle_sync_request(
    data: dict,
    peer: Any,
    responder_address: str,
    store: ThoughtStore,
) -> None:
    """
    Handle a THOUGHT_SYNC_REQUEST from a peer.

    Queries pg_inft for recent thoughts and sends a THOUGHT_SYNC_RESPONSE.
    Uses store.search() as a proxy — in a full implementation one would
    query by timestamp directly; here we return the top-50 matching any
    model as a best-effort bootstrap.
    """
    try:
        req = _from_dict_sync_request(data)
        model_id = req.model_id or ""
        lim      = min(req.limit, 200)  # server-side cap

        # Use an intentionally broad question to fetch recent thoughts.
        # A production implementation would use a direct timestamp range query.
        results = await store.search(question="the", model_id=model_id, limit=lim)

        thoughts_dicts = [
            {
                "type":          THOUGHT_BROADCAST,
                "job_id":        r.job_id,
                "miner_address": r.miner_address,
                "model_id":      r.model_id,
                "question_text": r.question_text,
                "thinking_text": r.thinking_text,
                "answer_text":   r.answer_text,
                "proof_sig":     "",   # omit sig for gossip (re-signed at source)
                "block_number":  0,
                "tx_hash":       "",
                "timestamp":     0.0,
            }
            for r in results
        ]

        response = ThoughtSyncResponse(
            thoughts   = thoughts_dicts,
            responder  = responder_address,
            has_more   = False,
            next_since_timestamp = 0.0,
        )
        payload = _to_json(response).encode("utf-8")
        if hasattr(peer, "send"):
            await peer.send(payload)
        elif hasattr(peer, "send_message"):
            await peer.send_message(payload.decode("utf-8"))

    except Exception as exc:
        log.warning("handle_sync_request_err err=%s", exc)


async def handle_sync_response(
    data: dict,
    peer_addr: str,
    store: ThoughtStore,
) -> None:
    """
    Handle a THOUGHT_SYNC_RESPONSE: ingest each bundled thought into the local store.
    """
    try:
        resp = _from_dict_sync_response(data)
        ingested = 0
        for t_dict in resp.thoughts:
            t = _from_dict_broadcast(t_dict)
            if not t.job_id:
                continue
            ok = await store.ingest(
                job_id        = t.job_id,
                miner_address = t.miner_address,
                model_id      = t.model_id,
                question      = t.question_text,
                thinking      = t.thinking_text,
                answer        = t.answer_text,
                proof_sig_hex = t.proof_sig,
                peer_origin   = peer_addr,
            )
            if ok:
                ingested += 1

        log.info(
            "sync_response_ingested peer=%s count=%d/%d",
            peer_addr, ingested, len(resp.thoughts),
        )
    except Exception as exc:
        log.warning("handle_sync_response_err peer=%s err=%s", peer_addr, exc)
