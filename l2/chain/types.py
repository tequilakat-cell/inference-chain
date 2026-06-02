"""
Core data types for InferenceChain L2.

Every other module imports from here. Stabilise this file before touching anything else.
All dataclasses are frozen (immutable) to make hashing safe and state mutations explicit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional

# ── Transaction types ─────────────────────────────────────────────────────────

class TxType(IntEnum):
    JOB_POST         = 1   # user requests inference job
    SHARD_COMMIT     = 2   # sequencer records completed shard + pays miner
    STAKE            = 3   # miner locks INFT into validator stake
    UNSTAKE          = 4   # miner begins 86400-block unbonding
    TRANSFER         = 5   # INFT transfer between addresses
    BRIDGE_DEPOSIT   = 6   # sequencer-synthesized: L1→L2 credit
    BRIDGE_WITHDRAW  = 7   # user initiates L2→L1 withdrawal
    SLASH            = 8   # sequencer-synthesized: offer-timeout slash (10%)
    SLASH_HARD       = 9   # sequencer-synthesized: result-timeout slash (30%)
    MINE_SUBMIT          = 10  # miner submits PoW solution, receives MINE tokens
    MINE_BRIDGE          = 11  # burn MINE on L2 to release wrapped MINE on L1
    MODEL_REGISTER       = 12  # miner commits Merkle root of model weights
    HISTORY_COMMIT       = 13  # sequencer-synthesized: records completed Q&A in per-wallet history
    CONTEXT_LOAD_COMMIT  = 14  # sequencer-synthesized: miner confirmed context chunk pre-load
    BENCHMARK_COMMIT     = 15  # sequencer-synthesized: records miner hardware benchmark score

# ── Shard modes ───────────────────────────────────────────────────────────────

class ShardMode:
    PARALLEL_SAMPLE   = "parallel_sample"    # N miners, same prompt, race/vote
    CONTEXT_SPLIT     = "context_split"      # prompt chunked, results concatenated
    SPECULATIVE       = "speculative"        # draft miner + verifier, 3-4x speedup
    PIPELINE_PARALLEL = "pipeline_parallel"  # llama.cpp RPC: both nodes in every forward pass
    TENSOR_PARALLEL   = "tensor_parallel"    # alias for PIPELINE_PARALLEL (same RPC mechanism)
    #
    # PIPELINE_PARALLEL / TENSOR_PARALLEL both use llama.cpp RPC + --tensor-split.
    # With equal fractions (0.5 / 0.5) this is true cross-node Tensor Parallelism:
    #   • Both nodes participate in EVERY layer of EVERY forward pass
    #   • Node 1 computes columns [0 .. N/2] of attention/FFN weight matrices
    #   • Node 2 computes columns [N/2 .. N] simultaneously
    #   • Results are all-reduced before the next layer
    # The TP groups span both nodes — they are tightly interdependent.

# ── Job and shard status ──────────────────────────────────────────────────────

class JobStatus:
    WAITING    = "waiting"     # blocked on parent job in a compounding chain
    PENDING    = "pending"     # posted, shards being dispatched
    PARTIAL    = "partial"     # some shards submitted
    ASSEMBLING = "assembling"  # all shards in, building final output
    COMPLETE   = "complete"    # final result committed to chain
    FAILED     = "failed"      # timed out or all miners slashed

class ShardStatus:
    UNASSIGNED = "unassigned"
    OFFERED    = "offered"     # ShardOffer sent over P2P
    ACCEPTED   = "accepted"    # miner acknowledged
    SUBMITTED  = "submitted"   # result received
    TIMEOUT    = "timeout"     # offer or result window expired
    SLASHED    = "slashed"     # miner penalised, shard being reassigned

# ── Core data structures ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShardSpec:
    shard_index:     int
    total_shards:    int
    mode:            str
    assigned_miner:  str          # checksummed Ethereum address
    prompt_slice:    str          # full prompt for coordinator; empty for pipeline workers
    max_tokens:      int
    timeout_ms:      int = 35_000
    backend_hint:    str = ""     # preferred backend type ("metal", "vulkan", …)
    # Pipeline parallel fields
    role:            str = ""     # "coordinator" | "worker" | "" (non-pipeline modes)
    rpc_peers:       str = ""     # JSON list of "host:port" strings for coordinator to connect to
    rpc_addr:        str = ""     # this miner's advertised RPC address (for on-chain audit)
    rpc_memory_gb:   str = ""     # JSON list of ints: memory budget (GB) per worker (coordinator only)
    tensor_split:    str = ""     # JSON list of floats: layer fraction per miner (coordinator only)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardSpec":
        return cls(**d)


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    miner:       str
    output:      str
    latency_ms:  int
    signature:   str             # hex-encoded ECDSA over keccak(shard_index||job_id||output)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardResult":
        return cls(**d)


@dataclass(frozen=True)
class TensorActivation:
    """
    Intermediate hidden state passed between pipeline stages in TENSOR_PARALLEL mode.
    Broadcast over the tensor_activations P2P topic; never stored on-chain.
    """
    job_id:    str
    stage:     int      # which pipeline stage produced this (0 = embed + first layers)
    n_stages:  int      # total stages in this job's pipeline
    shape:     str      # JSON list: [batch, seq_len, hidden_size]
    dtype:     str      # "float16" or "bfloat16"
    data_b64:  str      # base64-encoded hidden state bytes
    producer:  str      # miner address
    signature: str = "" # ECDSA over keccak(job_id || str(stage) || data_b64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TensorActivation":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class Transaction:
    tx_type:   int
    sender:    str
    nonce:     int
    payload:   str               # JSON-encoded type-specific fields
    gas_price: int = 0           # INFT-wei; 0 for sequencer-synthesized txs
    signature: str = ""          # hex; empty for sequencer-synthesized txs
    tx_hash:   str = ""          # filled by sign() or synthesize()

    def payload_dict(self) -> dict:
        return json.loads(self.payload)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transaction":
        return cls(**d)


@dataclass(frozen=True)
class BlockHeader:
    block_number:  int
    parent_hash:   str           # hex bytes32
    timestamp:     int           # unix milliseconds
    sequencer:     str           # checksummed address
    tx_root:       str           # merkle root of tx hashes
    state_root:    str           # merkle root of (address, account_state) pairs
    shard_root:    str           # merkle root of (job_id, shard_index, output_hash)
    gas_used:      int = 0
    extra_data:    str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BlockHeader":
        return cls(**d)


@dataclass(frozen=True)
class Block:
    header:       BlockHeader
    transactions: tuple           # tuple[Transaction, ...]
    sequencer_sig: str = ""       # ECDSA over block_hash
    block_hash:    str = ""       # keccak(rlp(header))

    def to_dict(self) -> dict:
        return {
            "header":        self.header.to_dict(),
            "transactions":  [tx.to_dict() for tx in self.transactions],
            "sequencer_sig": self.sequencer_sig,
            "block_hash":    self.block_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        return cls(
            header=BlockHeader.from_dict(d["header"]),
            transactions=tuple(Transaction.from_dict(t) for t in d["transactions"]),
            sequencer_sig=d.get("sequencer_sig", ""),
            block_hash=d.get("block_hash", ""),
        )


@dataclass(frozen=True)
class AccountState:
    balance_inft:  int = 0
    stake_inft:    int = 0
    unlock_block:  int = 0       # block at which unstake completes (0 = not unbonding)
    nonce:         int = 0
    reputation:    int = 500     # 0-1000; starts at 500 for new validators

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AccountState":
        return cls(**d)


@dataclass
class JobState:
    """Mutable in-memory tracking for an in-flight parallel job."""
    job_id:        str
    requester:     str
    model_id:      str
    prompt:        str
    mode:          str
    n_shards:      int
    max_tokens:    int
    fee_inft:      int
    block_number:  int           # block in which TX_JOB_POST was confirmed
    status:        str = JobStatus.PENDING
    specs:         dict = field(default_factory=dict)    # shard_index → ShardSpec
    results:       dict = field(default_factory=dict)    # shard_index → ShardResult
    shard_status:  dict = field(default_factory=dict)    # shard_index → ShardStatus.*
    final_output:  Optional[str] = None
    output_hash:   Optional[str] = None
    deadline_ms:   int = 0
    # Compounding inference — set when this job is part of a chain
    parent_job_id:      Optional[str] = None   # job whose output feeds this job's prompt
    prompt_template:    str = ""               # raw template; "{prev_output}" is replaced at dispatch
    parent_output_hash: Optional[str] = None   # assembled hash of the parent job
    chain_step:         int = 0                # 0 = root of chain
    # Context assembly — history injected as a prefix before the user's prompt
    original_prompt: str = ""           # user's actual prompt before context prefix was prepended
    context_hash:    Optional[str] = None  # merkle root of history entry job_ids used
    context_entries: int = 0            # count of history Q&A pairs prepended

    def all_shards_in(self) -> bool:
        return len(self.results) == self.n_shards

    def ready_to_assemble(self) -> bool:
        if self.mode == ShardMode.PARALLEL_SAMPLE:
            return len(self.results) >= 1
        if self.mode in (ShardMode.PIPELINE_PARALLEL, ShardMode.TENSOR_PARALLEL):
            # Only the coordinator (shard 0) produces real output; workers submit
            # liveness placeholders and their layer compute already happened (via the
            # RPC session) before the coordinator returns. Assemble the instant the
            # coordinator result lands instead of blocking on every worker placeholder
            # round-trip — a lost/late worker placeholder must never stall a job that
            # already has its answer.
            return 0 in self.results
        return self.all_shards_in()


@dataclass(frozen=True)
class StateRootCommitment:
    """Posted to L1 every state_root_interval blocks."""
    l2_block_number: int
    state_root:      str
    tx_batch_hash:   str
    timestamp:       int
    sequencer_sig:   str = ""

    def to_dict(self) -> dict:
        return asdict(self)
