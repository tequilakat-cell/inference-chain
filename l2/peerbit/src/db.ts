/**
 * InferenceChainDB — OrbitDB distributed miner database.
 *
 * Three databases replicated via libp2p gossipsub:
 *   miners  (documents) — miner profiles, keyed by Ethereum address
 *   jobs    (documents) — lightweight job board, keyed by job_id
 *   events  (events)    — append-only reputation event log
 *
 * Two sidecar nodes that open the same database address will auto-sync.
 * Persistence: LevelDB under ./data/orbitdb. Fast replay on restart.
 */

import { createOrbitDB } from "@orbitdb/core";
import { createHelia }   from "helia";
import type { MinerProfile, JobRecord, ReputationEvent } from "./schema.js";

export type { MinerProfile, JobRecord, ReputationEvent };

const DB_MINERS = "inferencechain.miners.v1";
const DB_JOBS   = "inferencechain.jobs.v1";
const DB_EVENTS = "inferencechain.events.v1";

export class InferenceChainDB {
    private orbitdb!: Awaited<ReturnType<typeof createOrbitDB>>;
    private ipfs!:    Awaited<ReturnType<typeof createHelia>>;
    private _miners!: any;
    private _jobs!:   any;
    private _events!: any;

    async open(_dataDir?: string, bootstrapAddrs: string[] = []): Promise<void> {
        // OrbitDB requires gossipsub — configure libp2p explicitly
        const { createLibp2p } = await import("libp2p");
        const { gossipsub }    = await import("@chainsafe/libp2p-gossipsub");
        const { identify }     = await import("@libp2p/identify");
        const { tcp }          = await import("@libp2p/tcp");

        const libp2p = await createLibp2p({
            addresses: { listen: ["/ip4/0.0.0.0/tcp/0"] },
            transports: [tcp()],
            services: {
                pubsub:   gossipsub({ allowPublishToZeroTopicPeers: true }) as any,
                identify: identify() as any,
            },
        } as any);

        this.ipfs = await createHelia({ libp2p: libp2p as any });

        // Dial any bootstrap peers
        if (bootstrapAddrs.length) {
            const { multiaddr } = await import("@multiformats/multiaddr");
            for (const addr of bootstrapAddrs) {
                try {
                    await (this.ipfs.libp2p as any).dial(multiaddr(addr));
                    console.log("[orbitdb] dialed:", addr);
                } catch (err) {
                    console.warn("[orbitdb] dial failed:", addr, (err as Error).message);
                }
            }
        }

        this.orbitdb = await createOrbitDB({ ipfs: this.ipfs as any });

        this._miners = await this.orbitdb.open(DB_MINERS, { type: "documents" });
        this._jobs   = await this.orbitdb.open(DB_JOBS,   { type: "documents" });
        this._events = await this.orbitdb.open(DB_EVENTS, { type: "events"    });

        console.log("[orbitdb] ready");
        console.log("  miners →", (this._miners as any).address);
        console.log("  jobs   →", (this._jobs   as any).address);
        console.log("  events →", (this._events as any).address);
    }

    // ── Miners ────────────────────────────────────────────────────────────────

    async upsertMiner(profile: MinerProfile): Promise<void> {
        await this._miners.put({ ...profile, _id: profile.address, lastSeen: Date.now() });
    }

    async getMiner(address: string): Promise<MinerProfile | undefined> {
        const results = await this._miners.get(address);
        return (results as any[])[0] as MinerProfile | undefined;
    }

    async listMiners(activeOnly = true): Promise<MinerProfile[]> {
        const all: Array<{ value: MinerProfile }> = await this._miners.all();
        const miners = all.map(e => e.value);
        return activeOnly ? miners.filter(m => m.active !== false) : miners;
    }

    async minersForModel(modelId: string): Promise<MinerProfile[]> {
        const all = await this.listMiners(true);
        return all.filter(m => {
            try { return (JSON.parse(m.models || "[]") as string[]).includes(modelId); }
            catch { return false; }
        });
    }

    async deleteMiner(address: string): Promise<void> {
        const existing = await this.getMiner(address);
        await this._miners.put({ ...(existing ?? {}), _id: address, address, active: false });
    }

    // ── Jobs ──────────────────────────────────────────────────────────────────

    async upsertJob(record: JobRecord): Promise<void> {
        await this._jobs.put({ ...record, _id: record.jobId });
    }

    async getJob(jobId: string): Promise<JobRecord | undefined> {
        const results = await this._jobs.get(jobId);
        return (results as any[])[0] as JobRecord | undefined;
    }

    async listJobs(status?: string): Promise<JobRecord[]> {
        const all: Array<{ value: JobRecord }> = await this._jobs.all();
        const jobs = all.map(e => e.value);
        return status ? jobs.filter(j => j.status === status) : jobs;
    }

    // ── Events ────────────────────────────────────────────────────────────────

    async appendEvent(event: ReputationEvent): Promise<void> {
        await this._events.add({ ...event, timestamp: event.timestamp || Date.now() });
    }

    async eventsForMiner(address: string): Promise<ReputationEvent[]> {
        const all: Array<{ value: ReputationEvent }> = await this._events.all();
        return all
            .map(e => e.value)
            .filter(e => e.miner?.toLowerCase() === address.toLowerCase());
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    async close(): Promise<void> {
        try { await this._miners?.close?.(); } catch { /**/ }
        try { await this._jobs?.close?.();   } catch { /**/ }
        try { await this._events?.close?.(); } catch { /**/ }
        try { await this.orbitdb?.stop?.();  } catch { /**/ }
        try { await this.ipfs?.stop?.();     } catch { /**/ }
    }

    peerInfo(): { peerId: string; addresses: string[]; databases: Record<string, string> } {
        return {
            peerId:    (this.ipfs?.libp2p as any)?.peerId?.toString() ?? "(starting)",
            addresses: ((this.ipfs?.libp2p as any)?.getMultiaddrs?.() ?? []).map((a: any) => a.toString()),
            databases: {
                miners: (this._miners as any)?.address ?? "",
                jobs:   (this._jobs   as any)?.address ?? "",
                events: (this._events as any)?.address ?? "",
            },
        };
    }
}
