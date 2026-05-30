/**
 * Plain TypeScript interfaces for the three InferenceChain Peerbit collections.
 * No decorator magic needed — these are JSON-serialised over Peerbit pubsub.
 */

export interface MinerProfile {
    address:    string;   // checksummed Ethereum address (primary key)
    models:     string;   // JSON array: '["Qwen/..."]'
    backend:    string;   // cpu | cuda | mlx | vulkan
    reputation: number;   // 0-1000
    lastSeen:   number;   // unix ms
    p2pAddr:    string;   // ws://host:port
    l2ChainId:  string;
    active:     boolean;
    maxShards:  number;
}

export interface JobRecord {
    jobId:       string;
    modelId:     string;
    mode:        string;  // parallel_sample | context_split | speculative
    nShards:     number;
    postedAt:    number;  // unix ms
    status:      string;  // pending | partial | complete | failed
    outputHash:  string;
    requester:   string;
    completedAt: number;  // unix ms, 0 if not yet
    latencyMs:   number;
}

export interface ReputationEvent {
    eventId:   string;   // uuid
    miner:     string;
    eventType: string;   // shard_complete | shard_failed | shard_slash
    delta:     number;   // reputation change
    jobId:     string;
    shardIdx:  number;
    timestamp: number;   // unix ms
    signature: string;
}
