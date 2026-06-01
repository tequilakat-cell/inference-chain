import { Router } from "express";
import type { Request, Response } from "express";
import { SearchRequest } from "@peerbit/document";
import type { InferenceChainProgram } from "./program.js";
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
import {
  semanticSearchRollups,
  semanticSearchThoughts,
  textMatchThoughts,
} from "./search.js";

const ok = (res: Response, data: unknown) => res.json({ ok: true, data });
const err = (res: Response, msg: string, status = 500) =>
  res.status(status).json({ ok: false, error: msg });

export function buildRouter(program: InferenceChainProgram): Router {
  const r = Router();

  // ── Health ────────────────────────────────────────────────────────────────

  r.get("/health", (_req, res) => res.json({ ok: true }));

  r.get("/status", async (_req, res) => {
    try {
      const [thoughts, rollups, benchmarks, miners, jobs, reputation] = await Promise.all([
        program.thoughts.index.search(new SearchRequest({ fetch: 50000 })),
        program.rollups.index.search(new SearchRequest({ fetch: 50000 })),
        program.allBenchmarks(),
        program.allMiners(),
        program.jobs.index.search(new SearchRequest({ fetch: 50000 })),
        program.reputation.index.search(new SearchRequest({ fetch: 50000 })),
      ]);
      const staleThreshold = BigInt(Date.now() - 5 * 60 * 1000);
      const activeMiners = (miners as MinerDocument[]).filter(
        (m) => m.active && m.last_seen > staleThreshold,
      );
      ok(res, {
        thoughts: thoughts.length,
        rollups: rollups.length,
        benchmarks: benchmarks.length,
        miners_active: activeMiners.length,
        miners_total: miners.length,
        jobs: jobs.length,
        reputation_events: reputation.length,
      });
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Thoughts ──────────────────────────────────────────────────────────────

  r.post("/thoughts", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      await program.thoughts.put(
        new ThoughtDocument({
          job_id: b.job_id,
          miner_address: b.miner_address,
          model_id: b.model_id,
          question_text: b.question,
          thinking_text: b.thinking,
          answer_text: b.answer,
          block_number: b.block_number,
          proof_sig: b.proof_sig,
          tx_hash: b.tx_hash,
          peer_origin: b.peer_origin,
        }),
      );
      ok(res, { ingested: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/thoughts/search", async (req: Request, res: Response) => {
    try {
      const q = String(req.query.q ?? "");
      const model_id = String(req.query.model_id ?? "");
      const limit = parseInt(String(req.query.limit ?? "5"));
      const all = await program.allThoughts();
      const results = textMatchThoughts(all, q, model_id, limit);
      ok(res, results.map(thoughtRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/thoughts/recent", async (req: Request, res: Response) => {
    try {
      const model_id = String(req.query.model_id ?? "");
      const limit = parseInt(String(req.query.limit ?? "20"));
      const all = await program.allThoughts(limit * 2);
      const filtered = model_id ? all.filter((d) => d.model_id === model_id) : all;
      ok(res, filtered.slice(0, limit).map(thoughtRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/thoughts/embedding", async (req: Request, res: Response) => {
    try {
      const { job_id, embedding } = req.body;
      const doc = await program.thoughts.index.get(job_id) as ThoughtDocument | undefined;
      if (!doc) { ok(res, { updated: false }); return; }
      const updated = Object.assign(Object.create(Object.getPrototypeOf(doc)), doc);
      updated.embedding = embedding;
      await program.thoughts.put(updated);
      ok(res, { updated: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/thoughts/search/semantic", async (req: Request, res: Response) => {
    try {
      const { embedding, model_id = "", limit = 20, min_score = 0.0 } = req.body;
      const all = await program.allThoughts();
      const results = semanticSearchThoughts(all, embedding, model_id, limit, min_score);
      ok(res, results.map((d) => ({ ...thoughtRow(d), score: d.score })));
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Rollups ───────────────────────────────────────────────────────────────

  r.post("/rollups", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      await program.rollups.put(
        new RollupDocument({
          rollup_id: b.rollup_id,
          topic: b.topic,
          model_id: b.model_id,
          summary_text: b.summary,
          source_count: b.source_count,
          source_job_ids: b.source_job_ids ?? [],
          embedding: b.embedding,
        }),
      );
      ok(res, { upserted: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/rollups/search", async (req: Request, res: Response) => {
    try {
      const { embedding, model_id = "", limit = 5, min_score = 0.0 } = req.body;
      const all = await program.allRollups();
      const results = semanticSearchRollups(all, embedding, model_id, limit, min_score);
      ok(res, results.map((d) => ({ ...rollupRow(d), score: d.score })));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/rollups", async (req: Request, res: Response) => {
    try {
      const model_id = String(req.query.model_id ?? "");
      const limit = parseInt(String(req.query.limit ?? "20"));
      const all = await program.allRollups(limit * 2);
      const filtered = model_id ? all.filter((d) => d.model_id === model_id) : all;
      ok(res, filtered.slice(0, limit).map(rollupRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Benchmarks ────────────────────────────────────────────────────────────

  r.get("/benchmarks", async (_req: Request, res: Response) => {
    try {
      const all = await program.allBenchmarks();
      ok(res, all.map(benchRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/benchmarks/:miner/:model", async (req: Request, res: Response) => {
    try {
      const id = `${req.params.miner}:${decodeURIComponent(req.params.model)}`;
      const doc = await program.benchmarks.index.get(id) as BenchmarkDocument | undefined;
      ok(res, doc ? benchRow(doc) : null);
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/benchmarks", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      const id = `${b.miner_address}:${b.model_id}`;
      const existing = await program.benchmarks.index.get(id) as BenchmarkDocument | undefined;
      await program.benchmarks.put(
        new BenchmarkDocument({
          miner_address: b.miner_address,
          model_id: b.model_id,
          tokens_per_sec: b.tokens_per_sec,
          live_tps: existing?.live_tps ?? 0,
          live_sample_count: existing?.live_sample_count ?? 0,
          n_tokens: b.n_tokens,
          elapsed_ms: b.elapsed_ms,
          nonce: b.nonce ?? "",
          block_number: b.block_number,
          expires_at_block: b.expires_at_block,
        }),
      );
      ok(res, { upserted: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/benchmarks/live-tps", async (req: Request, res: Response) => {
    try {
      const { miner_address, model_id, actual_tps } = req.body;
      const id = `${miner_address}:${model_id}`;
      const doc = await program.benchmarks.index.get(id) as BenchmarkDocument | undefined;
      if (!doc) { ok(res, { updated: false }); return; }
      const ALPHA = 0.3;
      const updated = Object.assign(Object.create(Object.getPrototypeOf(doc)), doc);
      updated.live_tps = doc.live_sample_count === 0
        ? actual_tps
        : ALPHA * actual_tps + (1 - ALPHA) * doc.live_tps;
      updated.live_sample_count = doc.live_sample_count + 1;
      updated.last_updated = BigInt(Date.now());
      await program.benchmarks.put(updated);
      ok(res, { updated: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Miners ────────────────────────────────────────────────────────────────

  r.post("/miners", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      const existing = await program.miners.index.get(b.address) as MinerDocument | undefined;
      const doc = new MinerDocument({
        address: b.address,
        models: b.models,
        backend: b.backend,
        p2p_addr: b.p2p_addr ?? "",
        chain_id: String(b.chain_id ?? "2026"),
        max_shards: b.max_shards ?? 4,
        reputation: existing?.reputation ?? 0,
      });
      await program.miners.put(doc);
      ok(res, { registered: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.delete("/miners/:address", async (req: Request, res: Response) => {
    try {
      const doc = await program.miners.index.get(req.params.address) as MinerDocument | undefined;
      if (doc) {
        const updated = Object.assign(Object.create(Object.getPrototypeOf(doc)), doc);
        updated.active = false;
        await program.miners.put(updated);
      }
      ok(res, { deregistered: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/miners", async (req: Request, res: Response) => {
    try {
      const model = String(req.query.model ?? "");
      const staleMs = 5 * 60 * 1000;
      const staleThreshold = BigInt(Date.now() - staleMs);
      const all = await program.allMiners();
      const active = all.filter(
        (m) => m.active && m.last_seen > staleThreshold,
      );
      const filtered = model ? active.filter((m) => m.models.includes(model)) : active;
      ok(res, filtered.map(minerRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/miners/cleanup", async (req: Request, res: Response) => {
    try {
      const staleMs = parseInt(String(req.body.stale_ms ?? 300_000));
      const threshold = BigInt(Date.now() - staleMs);
      const all = await program.allMiners();
      let cleaned = 0;
      for (const m of all) {
        if (m.active && m.last_seen < threshold) {
          const updated = Object.assign(Object.create(Object.getPrototypeOf(m)), m);
          updated.active = false;
          await program.miners.put(updated);
          cleaned++;
        }
      }
      ok(res, { cleaned });
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Jobs ──────────────────────────────────────────────────────────────────

  r.get("/jobs", async (req: Request, res: Response) => {
    try {
      const limit = parseInt(String(req.query.limit ?? "100"));
      const rows = await program.jobs.index.search(
        new SearchRequest({
          sort: [new Sort({ key: "updated_at", direction: SortDirection.DESC })],
          fetch: limit,
        }),
      ) as JobDocument[];
      ok(res, rows.map(j => ({
        job_id: j.job_id,
        model_id: j.model_id,
        mode: j.mode,
        n_shards: j.n_shards,
        status: j.status,
        output_hash: j.output_hash,
        latency_ms: j.latency_ms,
        updated_at: new Date(Number(j.updated_at)).toISOString(),
      })));
    } catch (e) {
      err(res, String(e));
    }
  });

  r.post("/jobs", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      await program.jobs.put(
        new JobDocument({
          job_id: b.job_id,
          model_id: b.model_id,
          mode: b.mode,
          n_shards: b.n_shards,
          status: b.status,
          output_hash: b.output_hash,
          latency_ms: b.latency_ms,
        }),
      );
      ok(res, { upserted: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Reputation ────────────────────────────────────────────────────────────

  r.post("/reputation", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      await program.reputation.put(
        new ReputationEventDocument({
          event_id: b.event_id,
          miner: b.miner,
          event_type: b.event_type,
          delta: b.delta,
          job_id: b.job_id,
          shard_idx: b.shard_idx ?? 0,
        }),
      );
      // Update miner's reputation score in the registry
      const minerDoc = await program.miners.index.get(b.miner) as MinerDocument | undefined;
      if (minerDoc) {
        const updated = Object.assign(Object.create(Object.getPrototypeOf(minerDoc)), minerDoc);
        updated.reputation = (minerDoc.reputation ?? 0) + b.delta;
        await program.miners.put(updated);
      }
      ok(res, { logged: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/reputation/:address", async (req: Request, res: Response) => {
    try {
      const events = await program.allReputation(req.params.address);
      ok(res, events.map(repRow));
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Job contexts ──────────────────────────────────────────────────────────

  r.post("/contexts", async (req: Request, res: Response) => {
    try {
      const b = req.body;
      await program.contexts.put(
        new JobContextDocument({
          job_id: b.job_id,
          query_text: b.query_text,
          context_text: b.context_text,
          context_hash: b.context_hash,
          model_id: b.model_id,
          n_entries: b.n_entries,
          ttl_ms: b.ttl_ms,
        }),
      );
      ok(res, { set: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.get("/contexts/:job_id", async (req: Request, res: Response) => {
    try {
      const doc = await program.contexts.index.get(req.params.job_id) as JobContextDocument | undefined;
      if (!doc || doc.expires_at < BigInt(Date.now())) { ok(res, null); return; }
      ok(res, {
        query_text: doc.query_text,
        context_text: doc.context_text,
        context_hash: doc.context_hash,
        model_id: doc.model_id,
        n_entries: doc.n_entries,
      });
    } catch (e) {
      err(res, String(e));
    }
  });

  r.delete("/contexts/expired", async (_req: Request, res: Response) => {
    try {
      const now = BigInt(Date.now());
      const all = await program.contexts.index.search(
        new SearchRequest({ fetch: 10000 }),
      ) as JobContextDocument[];
      let deleted = 0;
      for (const c of all) {
        if (c.expires_at < now) {
          await program.contexts.del(c.id);
          deleted++;
        }
      }
      ok(res, { deleted });
    } catch (e) {
      err(res, String(e));
    }
  });

  // ── Peers ─────────────────────────────────────────────────────────────────

  r.post("/peers", async (req: Request, res: Response) => {
    try {
      const { peer_address, job_id, rejected } = req.body;
      const existing = await program.peers.index.get(peer_address) as PeerSyncDocument | undefined;
      const doc = new PeerSyncDocument({
        peer_address,
        thoughts_received: (existing?.thoughts_received ?? 0) + 1,
        proofs_rejected: (existing?.proofs_rejected ?? 0) + (rejected ? 1 : 0),
        last_job_id: job_id,
      });
      await program.peers.put(doc);
      ok(res, { recorded: true });
    } catch (e) {
      err(res, String(e));
    }
  });

  return r;
}

// ── Row serialisers (BigInt → string for JSON) ─────────────────────────────

function thoughtRow(d: ThoughtDocument & { score?: number }) {
  return {
    id: Number(d.ingested_at), // stable numeric id approximation
    job_id: d.job_id,
    miner_address: d.miner_address,
    model_id: d.model_id,
    question_text: d.question_text ?? "",
    thinking_text: d.thinking_text ?? "",
    answer_text: d.answer_text ?? "",
    score: d.score ?? 0,
  };
}

function rollupRow(d: RollupDocument & { score?: number }) {
  return {
    rollup_id: d.rollup_id,
    topic: d.topic,
    model_id: d.model_id,
    summary_text: d.summary_text,
    source_count: d.source_count,
    source_job_ids: d.source_job_ids,
    created_at: new Date(Number(d.created_at)).toISOString(),
    score: d.score ?? 0,
  };
}

function benchRow(d: BenchmarkDocument) {
  return {
    miner_address: d.miner_address,
    model_id: d.model_id,
    tokens_per_sec: d.tokens_per_sec,
    live_tps: d.live_tps,
    live_sample_count: d.live_sample_count,
    n_tokens: d.n_tokens,
    elapsed_ms: d.elapsed_ms,
    nonce: d.nonce,
    block_number: Number(d.block_number),
    expires_at_block: Number(d.expires_at_block),
    last_updated: new Date(Number(d.last_updated)).toISOString(),
  };
}

function minerRow(d: MinerDocument) {
  return {
    address: d.address,
    models: d.models,
    backend: d.backend,
    p2p_addr: d.p2p_addr,
    max_shards: d.max_shards,
    reputation: d.reputation,
    last_seen: new Date(Number(d.last_seen)).toISOString(),
  };
}

function repRow(d: ReputationEventDocument) {
  return {
    event_id: d.event_id,
    miner: d.miner,
    event_type: d.event_type,
    delta: d.delta,
    job_id: d.job_id,
    shard_idx: d.shard_idx,
    created_at: new Date(Number(d.created_at)).toISOString(),
  };
}
