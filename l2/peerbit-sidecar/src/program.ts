import { field, variant } from "@dao-xyz/borsh";
import { Documents, SearchRequest, Sort, SortDirection } from "@peerbit/document";
import { Program } from "@peerbit/program";
import {
  BenchmarkDocument,
  JobContextDocument,
  JobDocument,
  MinerDocument,
  PeerSyncDocument,
  ReputationEventDocument,
  RollupDocument,
  ThoughtDocument,
} from "./types.js";

@variant("inference-chain-v1")
export class InferenceChainProgram extends Program {
  @field({ type: Documents }) thoughts: Documents<ThoughtDocument>;
  @field({ type: Documents }) rollups: Documents<RollupDocument>;
  @field({ type: Documents }) benchmarks: Documents<BenchmarkDocument>;
  @field({ type: Documents }) miners: Documents<MinerDocument>;
  @field({ type: Documents }) jobs: Documents<JobDocument>;
  @field({ type: Documents }) reputation: Documents<ReputationEventDocument>;
  @field({ type: Documents }) contexts: Documents<JobContextDocument>;
  @field({ type: Documents }) peers: Documents<PeerSyncDocument>;

  constructor() {
    super();
    this.thoughts = new Documents();
    this.rollups = new Documents();
    this.benchmarks = new Documents();
    this.miners = new Documents();
    this.jobs = new Documents();
    this.reputation = new Documents();
    this.contexts = new Documents();
    this.peers = new Documents();
  }

  async open() {
    await Promise.all([
      this.thoughts.open({ type: ThoughtDocument }),
      this.rollups.open({ type: RollupDocument }),
      this.benchmarks.open({ type: BenchmarkDocument }),
      this.miners.open({ type: MinerDocument }),
      this.jobs.open({ type: JobDocument }),
      this.reputation.open({ type: ReputationEventDocument }),
      this.contexts.open({ type: JobContextDocument }),
      this.peers.open({ type: PeerSyncDocument }),
    ]);
  }

  // ── Convenience fetch helpers ───────────────────────────────────────────────

  async allThoughts(limit = 2000): Promise<ThoughtDocument[]> {
    return this.thoughts.index.search(
      new SearchRequest({
        sort: [new Sort({ key: "ingested_at", direction: SortDirection.DESC })],
        fetch: limit,
      }),
    ) as Promise<ThoughtDocument[]>;
  }

  async allRollups(limit = 500): Promise<RollupDocument[]> {
    return this.rollups.index.search(
      new SearchRequest({
        sort: [new Sort({ key: "created_at", direction: SortDirection.DESC })],
        fetch: limit,
      }),
    ) as Promise<RollupDocument[]>;
  }

  async allBenchmarks(): Promise<BenchmarkDocument[]> {
    return this.benchmarks.index.search(
      new SearchRequest({ fetch: 10000 }),
    ) as Promise<BenchmarkDocument[]>;
  }

  async allMiners(): Promise<MinerDocument[]> {
    return this.miners.index.search(
      new SearchRequest({ fetch: 1000 }),
    ) as Promise<MinerDocument[]>;
  }

  async allReputation(miner: string): Promise<ReputationEventDocument[]> {
    return this.reputation.index.search(
      new SearchRequest({ fetch: 1000 }),
    ).then((rows) =>
      (rows as ReputationEventDocument[])
        .filter((r) => r.miner === miner)
        .sort((a, b) => Number(b.created_at - a.created_at)),
    );
  }
}
