import { field, option, vec } from "@dao-xyz/borsh";

export class ThoughtDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) job_id: string;
  @field({ type: "string" }) miner_address: string;
  @field({ type: "string" }) model_id: string;
  @field({ type: option("string") }) question_text?: string;
  @field({ type: option("string") }) thinking_text?: string;
  @field({ type: option("string") }) answer_text?: string;
  @field({ type: "u64" }) block_number: bigint;
  @field({ type: "u64" }) ingested_at: bigint;
  @field({ type: option(vec("f32")) }) embedding?: number[];
  @field({ type: option("string") }) proof_sig?: string;
  @field({ type: option("string") }) tx_hash?: string;
  @field({ type: option("string") }) peer_origin?: string;

  constructor(props?: {
    job_id: string;
    miner_address: string;
    model_id: string;
    question_text?: string;
    thinking_text?: string;
    answer_text?: string;
    block_number?: number | bigint;
    embedding?: number[];
    proof_sig?: string;
    tx_hash?: string;
    peer_origin?: string;
  }) {
    if (!props) {
      this.id = "";
      this.job_id = "";
      this.miner_address = "";
      this.model_id = "";
      this.block_number = 0n;
      this.ingested_at = 0n;
      return;
    }
    this.id = props.job_id;
    this.job_id = props.job_id;
    this.miner_address = props.miner_address;
    this.model_id = props.model_id;
    this.question_text = props.question_text;
    this.thinking_text = props.thinking_text;
    this.answer_text = props.answer_text;
    this.block_number = BigInt(props.block_number ?? 0);
    this.ingested_at = BigInt(Date.now());
    this.embedding = props.embedding;
    this.proof_sig = props.proof_sig;
    this.tx_hash = props.tx_hash;
    this.peer_origin = props.peer_origin;
  }
}

export class RollupDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) rollup_id: string;
  @field({ type: "string" }) topic: string;
  @field({ type: "string" }) model_id: string;
  @field({ type: "string" }) summary_text: string;
  @field({ type: "u32" }) source_count: number;
  @field({ type: vec("string") }) source_job_ids: string[];
  @field({ type: "u64" }) created_at: bigint;
  @field({ type: option(vec("f32")) }) embedding?: number[];

  constructor(props?: {
    rollup_id: string;
    topic: string;
    model_id: string;
    summary_text: string;
    source_count: number;
    source_job_ids: string[];
    embedding?: number[];
  }) {
    if (!props) {
      this.id = "";
      this.rollup_id = "";
      this.topic = "";
      this.model_id = "";
      this.summary_text = "";
      this.source_count = 0;
      this.source_job_ids = [];
      this.created_at = 0n;
      return;
    }
    this.id = props.rollup_id;
    this.rollup_id = props.rollup_id;
    this.topic = props.topic;
    this.model_id = props.model_id;
    this.summary_text = props.summary_text;
    this.source_count = props.source_count;
    this.source_job_ids = props.source_job_ids;
    this.created_at = BigInt(Date.now());
    this.embedding = props.embedding;
  }
}

export class BenchmarkDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) miner_address: string;
  @field({ type: "string" }) model_id: string;
  @field({ type: "f64" }) tokens_per_sec: number;
  @field({ type: "f64" }) live_tps: number;
  @field({ type: "u32" }) live_sample_count: number;
  @field({ type: "u32" }) n_tokens: number;
  @field({ type: "u32" }) elapsed_ms: number;
  @field({ type: "string" }) nonce: string;
  @field({ type: "u64" }) block_number: bigint;
  @field({ type: "u64" }) expires_at_block: bigint;
  @field({ type: "u64" }) last_updated: bigint;

  constructor(props?: {
    miner_address: string;
    model_id: string;
    tokens_per_sec: number;
    live_tps?: number;
    live_sample_count?: number;
    n_tokens: number;
    elapsed_ms: number;
    nonce: string;
    block_number: number;
    expires_at_block: number;
  }) {
    if (!props) {
      this.id = "";
      this.miner_address = "";
      this.model_id = "";
      this.tokens_per_sec = 0;
      this.live_tps = 0;
      this.live_sample_count = 0;
      this.n_tokens = 0;
      this.elapsed_ms = 0;
      this.nonce = "";
      this.block_number = 0n;
      this.expires_at_block = 0n;
      this.last_updated = 0n;
      return;
    }
    this.id = `${props.miner_address}:${props.model_id}`;
    this.miner_address = props.miner_address;
    this.model_id = props.model_id;
    this.tokens_per_sec = props.tokens_per_sec;
    this.live_tps = props.live_tps ?? 0;
    this.live_sample_count = props.live_sample_count ?? 0;
    this.n_tokens = props.n_tokens;
    this.elapsed_ms = props.elapsed_ms;
    this.nonce = props.nonce;
    this.block_number = BigInt(props.block_number);
    this.expires_at_block = BigInt(props.expires_at_block);
    this.last_updated = BigInt(Date.now());
  }
}

export class MinerDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) address: string;
  @field({ type: vec("string") }) models: string[];
  @field({ type: "string" }) backend: string;
  @field({ type: "string" }) p2p_addr: string;
  @field({ type: "string" }) chain_id: string;
  @field({ type: "u32" }) max_shards: number;
  @field({ type: "f64" }) reputation: number;
  @field({ type: "bool" }) active: boolean;
  @field({ type: "u64" }) last_seen: bigint;

  constructor(props?: {
    address: string;
    models: string[];
    backend: string;
    p2p_addr: string;
    chain_id: string;
    max_shards: number;
    reputation?: number;
  }) {
    if (!props) {
      this.id = "";
      this.address = "";
      this.models = [];
      this.backend = "";
      this.p2p_addr = "";
      this.chain_id = "";
      this.max_shards = 0;
      this.reputation = 0;
      this.active = false;
      this.last_seen = 0n;
      return;
    }
    this.id = props.address;
    this.address = props.address;
    this.models = props.models;
    this.backend = props.backend;
    this.p2p_addr = props.p2p_addr;
    this.chain_id = props.chain_id;
    this.max_shards = props.max_shards;
    this.reputation = props.reputation ?? 0;
    this.active = true;
    this.last_seen = BigInt(Date.now());
  }
}

export class JobDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) job_id: string;
  @field({ type: "string" }) model_id: string;
  @field({ type: "string" }) mode: string;
  @field({ type: "u32" }) n_shards: number;
  @field({ type: "string" }) status: string;
  @field({ type: "string" }) output_hash: string;
  @field({ type: "u32" }) latency_ms: number;
  @field({ type: "u64" }) updated_at: bigint;

  constructor(props?: {
    job_id: string;
    model_id?: string;
    mode?: string;
    n_shards?: number;
    status: string;
    output_hash?: string;
    latency_ms?: number;
  }) {
    if (!props) {
      this.id = "";
      this.job_id = "";
      this.model_id = "";
      this.mode = "";
      this.n_shards = 0;
      this.status = "";
      this.output_hash = "";
      this.latency_ms = 0;
      this.updated_at = 0n;
      return;
    }
    this.id = props.job_id;
    this.job_id = props.job_id;
    this.model_id = props.model_id ?? "";
    this.mode = props.mode ?? "";
    this.n_shards = props.n_shards ?? 0;
    this.status = props.status;
    this.output_hash = props.output_hash ?? "";
    this.latency_ms = props.latency_ms ?? 0;
    this.updated_at = BigInt(Date.now());
  }
}

export class ReputationEventDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) event_id: string;
  @field({ type: "string" }) miner: string;
  @field({ type: "string" }) event_type: string;
  @field({ type: "i32" }) delta: number;
  @field({ type: "string" }) job_id: string;
  @field({ type: "u32" }) shard_idx: number;
  @field({ type: "u64" }) created_at: bigint;

  constructor(props?: {
    event_id: string;
    miner: string;
    event_type: string;
    delta: number;
    job_id: string;
    shard_idx: number;
  }) {
    if (!props) {
      this.id = "";
      this.event_id = "";
      this.miner = "";
      this.event_type = "";
      this.delta = 0;
      this.job_id = "";
      this.shard_idx = 0;
      this.created_at = 0n;
      return;
    }
    this.id = props.event_id;
    this.event_id = props.event_id;
    this.miner = props.miner;
    this.event_type = props.event_type;
    this.delta = props.delta;
    this.job_id = props.job_id;
    this.shard_idx = props.shard_idx;
    this.created_at = BigInt(Date.now());
  }
}

export class JobContextDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) job_id: string;
  @field({ type: "string" }) query_text: string;
  @field({ type: "string" }) context_text: string;
  @field({ type: "string" }) context_hash: string;
  @field({ type: "string" }) model_id: string;
  @field({ type: "u32" }) n_entries: number;
  @field({ type: "u64" }) expires_at: bigint;

  constructor(props?: {
    job_id: string;
    query_text: string;
    context_text: string;
    context_hash: string;
    model_id: string;
    n_entries: number;
    ttl_ms?: number;
  }) {
    if (!props) {
      this.id = "";
      this.job_id = "";
      this.query_text = "";
      this.context_text = "";
      this.context_hash = "";
      this.model_id = "";
      this.n_entries = 0;
      this.expires_at = 0n;
      return;
    }
    this.id = props.job_id;
    this.job_id = props.job_id;
    this.query_text = props.query_text;
    this.context_text = props.context_text;
    this.context_hash = props.context_hash;
    this.model_id = props.model_id;
    this.n_entries = props.n_entries;
    this.expires_at = BigInt(Date.now() + (props.ttl_ms ?? 600_000));
  }
}

export class PeerSyncDocument {
  @field({ type: "string" }) id: string;
  @field({ type: "string" }) peer_address: string;
  @field({ type: "u64" }) last_seen: bigint;
  @field({ type: "u32" }) thoughts_received: number;
  @field({ type: "u32" }) proofs_rejected: number;
  @field({ type: "string" }) last_job_id: string;

  constructor(props?: {
    peer_address: string;
    thoughts_received?: number;
    proofs_rejected?: number;
    last_job_id?: string;
  }) {
    if (!props) {
      this.id = "";
      this.peer_address = "";
      this.last_seen = 0n;
      this.thoughts_received = 0;
      this.proofs_rejected = 0;
      this.last_job_id = "";
      return;
    }
    this.id = props.peer_address;
    this.peer_address = props.peer_address;
    this.last_seen = BigInt(Date.now());
    this.thoughts_received = props.thoughts_received ?? 0;
    this.proofs_rejected = props.proofs_rejected ?? 0;
    this.last_job_id = props.last_job_id ?? "";
  }
}
