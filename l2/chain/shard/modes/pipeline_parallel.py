"""
Pipeline parallel mode — llama.cpp RPC layer distribution.

Model layers are split across miners using the llama.cpp RPC backend:
  - Shard 0 (coordinator): loads the GGUF, connects to worker RPC servers,
    runs full generation. The llama.cpp runtime distributes layer weights
    proportionally by available memory on each node.
  - Shard 1..N-1 (workers): run rpc-server as an always-on sidecar,
    receive layer weights from the coordinator, and compute their portion.

The coordinator produces the final text. Workers submit placeholder results
immediately after accepting the offer (their compute contribution happens
transparently via the RPC protocol, not through the shard result message).

Assembly: wait for all shards (coordinator + workers), return coordinator's output.
"""

from __future__ import annotations

from ...types import ShardResult


def split_prompt(prompt: str, n_shards: int) -> list[str]:
    """
    Coordinator (shard 0) gets the full prompt.
    Workers get empty string — they don't need the prompt, only RPC connections.
    """
    slices = [""] * n_shards
    slices[0] = prompt
    return slices


def assemble(results: list[ShardResult]) -> str:
    """
    Return the coordinator's (shard 0) output.
    Worker placeholder outputs ("[pipeline_worker:N]") are ignored.
    """
    if not results:
        return ""
    ordered = sorted(results, key=lambda r: r.shard_index)
    coordinator = ordered[0]
    output = coordinator.output.strip()
    # Guard: if coordinator somehow sent a placeholder, try next shard
    if output.startswith("[pipeline_worker:"):
        for r in ordered[1:]:
            if not r.output.startswith("[pipeline_worker:"):
                return r.output.strip()
        return ""
    return output
