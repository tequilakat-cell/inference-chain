"""
P2P message envelopes.

All messages are JSON-serialisable dicts with a mandatory "type" field.
The sender signs the payload so recipients can verify authenticity.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any


TOPICS = {
    "blocks":               "blocks",               # new block announcements
    "shard_offers":         "shard_offers",         # sequencer → miners: work available
    "shard_results":        "shard_results",        # miners → sequencer: work done
    "speculative_draft":    "speculative_draft",    # draft miner → verifier: streaming tokens
    "tensor_activations":   "tensor_activations",  # pipeline stage N → stage N+1: hidden state
    "heartbeats":           "heartbeats",           # miner liveness pings (include backend + TP capability)
    "peer_announce":        "peer_announce",        # peer discovery
    "context_load_offers":  "context_load_offers",  # sequencer → miners: pre-load context chunk
    "context_load_results": "context_load_results", # miners → sequencer: chunk loaded (Option B parallel)
    "thought_broadcast":    "thought_broadcast",    # pg_inft: gossip completed inference thoughts
    "thought_sync":         "thought_sync",         # pg_inft: cold-start sync request/response
    "rollup_broadcast":     "rollup_broadcast",     # pg_inft: gossip consolidated rollup memories
    "benchmark_challenges": "benchmark_challenges", # sequencer → miners: run benchmark challenge
    "benchmark_responses":  "benchmark_responses",  # miners → sequencer: benchmark result
}

# ── pg_inft thought gossip message type constants ─────────────────────────────
THOUGHT_BROADCAST     = 20
THOUGHT_SYNC_REQUEST  = 21
THOUGHT_SYNC_RESPONSE = 22


@dataclass
class Envelope:
    """Wrapper around any P2P message. Always signed by the sender."""
    msg_type:  str
    sender:    str       # checksummed address
    timestamp: int       # unix ms
    payload:   dict
    signature: str = "" # ECDSA over keccak(msg_type + sender + timestamp + payload_json)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(
            msg_type=d["msg_type"],
            sender=d.get("sender", ""),
            timestamp=d.get("timestamp", 0),
            payload=d.get("payload", {}),
            signature=d.get("signature", ""),
        )

    def sign_preimage(self) -> bytes:
        payload_str = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return (self.msg_type + self.sender + str(self.timestamp) + payload_str).encode()

    def sign(self, privkey: str) -> "Envelope":
        from ..crypto import sign, address_from_key
        preimage = self.sign_preimage()
        sig = sign(privkey, preimage)
        return Envelope(
            msg_type=self.msg_type,
            sender=address_from_key(privkey),
            timestamp=self.timestamp,
            payload=self.payload,
            signature=sig,
        )

    def verify(self) -> bool:
        if not self.signature or not self.sender:
            return True   # unsigned messages (e.g. from sequencer self-calls) are trusted
        from ..crypto import verify_sig
        return verify_sig(self.sign_preimage(), self.signature, self.sender)


def make_envelope(
    msg_type: str,
    payload:  dict,
    privkey:  str = "",
    sender:   str = "",
) -> Envelope:
    env = Envelope(
        msg_type=msg_type,
        sender=sender,
        timestamp=int(time.time() * 1000),
        payload=payload,
    )
    if privkey:
        env = env.sign(privkey)
    return env
