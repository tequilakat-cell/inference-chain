import { Router } from "express";
import type { Request, Response } from "express";
import type Database from "better-sqlite3";
import { cosineSim } from "./search.js";

const ok = (res: Response, data: unknown) => res.json({ ok: true, data });
const err = (res: Response, msg: string, status = 500) =>
  res.status(status).json({ ok: false, error: msg });

export function buildRouter(db: Database.Database): Router {
  const r = Router();

  // ── Health / Status ───────────────────────────────────────────────────────

  r.get("/health", (_req, res) => res.json({ ok: true }));

  r.get("/status", (_req: Request, res: Response) => {
    try {
      const stale = Date.now() - 5 * 60 * 1000;
      ok(res, {
        thoughts:         (db.prepare("SELECT COUNT(*) AS n FROM thoughts").get() as any).n,
        rollups:          (db.prepare("SELECT COUNT(*) AS n FROM rollups").get() as any).n,
        benchmarks:       (db.prepare("SELECT COUNT(*) AS n FROM benchmarks").get() as any).n,
        miners_active:    (db.prepare("SELECT COUNT(*) AS n FROM miners WHERE active=1 AND last_seen>?").get(stale) as any).n,
        miners_total:     (db.prepare("SELECT COUNT(*) AS n FROM miners").get() as any).n,
        jobs:             (db.prepare("SELECT COUNT(*) AS n FROM jobs").get() as any).n,
        reputation_events:(db.prepare("SELECT COUNT(*) AS n FROM reputation").get() as any).n,
      });
    } catch (e) { err(res, String(e)); }
  });

  // ── Thoughts ──────────────────────────────────────────────────────────────

  r.post("/thoughts", (req: Request, res: Response) => {
    try {
      const b = req.body;
      db.prepare(`
        INSERT INTO thoughts(id,job_id,miner_addr,model_id,question,thinking,answer,
          block_num,ingested_at,proof_sig,tx_hash,peer_origin)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          miner_addr=excluded.miner_addr, model_id=excluded.model_id,
          question=excluded.question, thinking=excluded.thinking, answer=excluded.answer,
          proof_sig=excluded.proof_sig, tx_hash=excluded.tx_hash, peer_origin=excluded.peer_origin
      `).run(
        b.job_id, b.job_id, b.miner_address ?? "", b.model_id ?? "",
        b.question ?? "", b.thinking ?? "", b.answer ?? "",
        b.block_number ?? 0, Date.now(),
        b.proof_sig ?? null, b.tx_hash ?? null, b.peer_origin ?? null,
      );
      ok(res, { ingested: true });
    } catch (e) { err(res, String(e)); }
  });

  r.get("/thoughts/search", (req: Request, res: Response) => {
    try {
      const q = String(req.query.q ?? "").trim();
      const model = String(req.query.model_id ?? "");
      const limit = Math.min(parseInt(String(req.query.limit ?? "5")), 200);
      if (!q) { ok(res, []); return; }
      const ftsQ = q.split(/\s+/).map(w => `"${w.replace(/"/g, '')}"`).join(" OR ");
      let rows: any[];
      if (model) {
        rows = db.prepare(`
          SELECT t.id,t.job_id,t.miner_addr AS miner_address,t.model_id,
                 t.question AS question_text,t.thinking AS thinking_text,
                 t.answer AS answer_text,t.ingested_at,
                 bm25(thoughts_fts) AS score
          FROM thoughts_fts f
          JOIN thoughts t ON t.rowid = f.rowid
          WHERE thoughts_fts MATCH ? AND t.model_id = ?
          ORDER BY score LIMIT ?
        `).all(ftsQ, model, limit);
      } else {
        rows = db.prepare(`
          SELECT t.id,t.job_id,t.miner_addr AS miner_address,t.model_id,
                 t.question AS question_text,t.thinking AS thinking_text,
                 t.answer AS answer_text,t.ingested_at,
                 bm25(thoughts_fts) AS score
          FROM thoughts_fts f
          JOIN thoughts t ON t.rowid = f.rowid
          WHERE thoughts_fts MATCH ?
          ORDER BY score LIMIT ?
        `).all(ftsQ, limit);
      }
      ok(res, rows.map(thoughtRow));
    } catch (e) { err(res, String(e)); }
  });

  r.get("/thoughts/recent", (req: Request, res: Response) => {
    try {
      const model = String(req.query.model_id ?? "");
      const limit = Math.min(parseInt(String(req.query.limit ?? "20")), 200);
      const rows = model
        ? db.prepare("SELECT * FROM thoughts WHERE model_id=? ORDER BY ingested_at DESC LIMIT ?").all(model, limit)
        : db.prepare("SELECT * FROM thoughts ORDER BY ingested_at DESC LIMIT ?").all(limit);
      ok(res, (rows as any[]).map(thoughtRow));
    } catch (e) { err(res, String(e)); }
  });

  r.post("/thoughts/embedding", (req: Request, res: Response) => {
    try {
      const { job_id, embedding } = req.body;
      const info = db.prepare("UPDATE thoughts SET embedding=? WHERE id=?")
        .run(JSON.stringify(embedding), job_id);
      ok(res, { updated: info.changes > 0 });
    } catch (e) { err(res, String(e)); }
  });

  r.post("/thoughts/search/semantic", (req: Request, res: Response) => {
    try {
      const { embedding, model_id = "", limit = 20, min_score = 0.0 } = req.body;
      const rows = (model_id
        ? db.prepare("SELECT * FROM thoughts WHERE embedding IS NOT NULL AND model_id=?").all(model_id)
        : db.prepare("SELECT * FROM thoughts WHERE embedding IS NOT NULL").all()
      ) as any[];
      const scored = rows
        .map(r => ({ ...r, _score: cosineSim(embedding, JSON.parse(r.embedding)) }))
        .filter(r => r._score >= min_score)
        .sort((a, b) => b._score - a._score)
        .slice(0, limit);
      ok(res, scored.map(r => ({ ...thoughtRow(r), score: r._score })));
    } catch (e) { err(res, String(e)); }
  });

  // ── Rollups ───────────────────────────────────────────────────────────────

  r.post("/rollups", (req: Request, res: Response) => {
    try {
      const b = req.body;
      db.prepare(`
        INSERT INTO rollups(id,rollup_id,topic,model_id,summary_text,source_count,source_jobs,created_at,embedding)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          topic=excluded.topic, summary_text=excluded.summary_text,
          source_count=excluded.source_count, source_jobs=excluded.source_jobs,
          embedding=excluded.embedding
      `).run(
        b.rollup_id, b.rollup_id, b.topic ?? "", b.model_id ?? "",
        b.summary ?? "", b.source_count ?? 0,
        JSON.stringify(b.source_job_ids ?? []), Date.now(),
        b.embedding ? JSON.stringify(b.embedding) : null,
      );
      ok(res, { upserted: true });
    } catch (e) { err(res, String(e)); }
  });

  r.post("/rollups/search", (req: Request, res: Response) => {
    try {
      const { embedding, model_id = "", limit = 5, min_score = 0.0 } = req.body;
      const rows = (model_id
        ? db.prepare("SELECT * FROM rollups WHERE embedding IS NOT NULL AND model_id=?").all(model_id)
        : db.prepare("SELECT * FROM rollups WHERE embedding IS NOT NULL").all()
      ) as any[];
      const scored = rows
        .map(r => ({ ...r, _score: cosineSim(embedding, JSON.parse(r.embedding)) }))
        .filter(r => r._score >= min_score)
        .sort((a, b) => b._score - a._score)
        .slice(0, limit);
      ok(res, scored.map(r => ({ ...rollupRow(r), score: r._score })));
    } catch (e) { err(res, String(e)); }
  });

  r.get("/rollups", (req: Request, res: Response) => {
    try {
      const model = String(req.query.model_id ?? "");
      const limit = Math.min(parseInt(String(req.query.limit ?? "20")), 500);
      const rows = model
        ? db.prepare("SELECT * FROM rollups WHERE model_id=? ORDER BY created_at DESC LIMIT ?").all(model, limit)
        : db.prepare("SELECT * FROM rollups ORDER BY created_at DESC LIMIT ?").all(limit);
      ok(res, (rows as any[]).map(rollupRow));
    } catch (e) { err(res, String(e)); }
  });

  // ── Benchmarks ────────────────────────────────────────────────────────────

  r.get("/benchmarks", (_req: Request, res: Response) => {
    try {
      ok(res, (db.prepare("SELECT * FROM benchmarks ORDER BY last_updated DESC").all() as any[]).map(benchRow));
    } catch (e) { err(res, String(e)); }
  });

  r.get("/benchmarks/:miner/:model", (req: Request, res: Response) => {
    try {
      const id = `${req.params.miner}:${decodeURIComponent(req.params.model)}`;
      const row = db.prepare("SELECT * FROM benchmarks WHERE id=?").get(id);
      ok(res, row ? benchRow(row as any) : null);
    } catch (e) { err(res, String(e)); }
  });

  r.post("/benchmarks", (req: Request, res: Response) => {
    try {
      const b = req.body;
      const id = `${b.miner_address}:${b.model_id}`;
      const existing = db.prepare("SELECT live_tps,live_sample_count FROM benchmarks WHERE id=?").get(id) as any;
      db.prepare(`
        INSERT INTO benchmarks(id,miner_addr,model_id,tokens_per_sec,live_tps,live_sample_count,
          n_tokens,elapsed_ms,nonce,block_number,expires_at_block,last_updated)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          tokens_per_sec=excluded.tokens_per_sec, n_tokens=excluded.n_tokens,
          elapsed_ms=excluded.elapsed_ms, nonce=excluded.nonce,
          block_number=excluded.block_number, expires_at_block=excluded.expires_at_block,
          last_updated=excluded.last_updated
      `).run(
        id, b.miner_address, b.model_id, b.tokens_per_sec ?? 0,
        existing?.live_tps ?? 0, existing?.live_sample_count ?? 0,
        b.n_tokens ?? 0, b.elapsed_ms ?? 0, b.nonce ?? "",
        b.block_number ?? 0, b.expires_at_block ?? 0, Date.now(),
      );
      ok(res, { upserted: true });
    } catch (e) { err(res, String(e)); }
  });

  r.post("/benchmarks/live-tps", (req: Request, res: Response) => {
    try {
      const { miner_address, model_id, actual_tps } = req.body;
      const id = `${miner_address}:${model_id}`;
      const row = db.prepare("SELECT live_tps,live_sample_count FROM benchmarks WHERE id=?").get(id) as any;
      if (!row) { ok(res, { updated: false }); return; }
      const ALPHA = 0.3;
      const newTps = row.live_sample_count === 0
        ? actual_tps
        : ALPHA * actual_tps + (1 - ALPHA) * row.live_tps;
      db.prepare("UPDATE benchmarks SET live_tps=?,live_sample_count=?,last_updated=? WHERE id=?")
        .run(newTps, row.live_sample_count + 1, Date.now(), id);
      ok(res, { updated: true });
    } catch (e) { err(res, String(e)); }
  });

  // ── Miners ────────────────────────────────────────────────────────────────

  r.post("/miners", (req: Request, res: Response) => {
    try {
      const b = req.body;
      const existing = db.prepare("SELECT reputation FROM miners WHERE id=?").get(b.address) as any;
      db.prepare(`
        INSERT INTO miners(id,address,models,backend,p2p_addr,chain_id,max_shards,reputation,active,last_seen)
        VALUES(?,?,?,?,?,?,?,?,1,?)
        ON CONFLICT(id) DO UPDATE SET
          models=excluded.models, backend=excluded.backend, p2p_addr=excluded.p2p_addr,
          chain_id=excluded.chain_id, max_shards=excluded.max_shards,
          active=1, last_seen=excluded.last_seen
      `).run(
        b.address, b.address, JSON.stringify(b.models ?? []), b.backend ?? "",
        b.p2p_addr ?? "", String(b.chain_id ?? "2026"), b.max_shards ?? 4,
        existing?.reputation ?? 0, Date.now(),
      );
      ok(res, { registered: true });
    } catch (e) { err(res, String(e)); }
  });

  r.delete("/miners/:address", (req: Request, res: Response) => {
    try {
      db.prepare("UPDATE miners SET active=0 WHERE id=?").run(req.params.address);
      ok(res, { deregistered: true });
    } catch (e) { err(res, String(e)); }
  });

  r.get("/miners", (req: Request, res: Response) => {
    try {
      const model = String(req.query.model ?? "");
      const stale = Date.now() - 5 * 60 * 1000;
      const rows = (db.prepare("SELECT * FROM miners WHERE active=1 AND last_seen>?").all(stale) as any[])
        .filter(m => !model || JSON.parse(m.models).includes(model));
      ok(res, rows.map(minerRow));
    } catch (e) { err(res, String(e)); }
  });

  r.post("/miners/cleanup", (req: Request, res: Response) => {
    try {
      const staleMs = req.body?.stale_ms ?? 300_000;
      const threshold = Date.now() - staleMs;
      const info = db.prepare("UPDATE miners SET active=0 WHERE active=1 AND last_seen<?").run(threshold);
      ok(res, { cleaned: info.changes });
    } catch (e) { err(res, String(e)); }
  });

  // ── Jobs ──────────────────────────────────────────────────────────────────

  r.get("/jobs", (req: Request, res: Response) => {
    try {
      const limit = Math.min(parseInt(String(req.query.limit ?? "100")), 500);
      const rows = db.prepare("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?").all(limit);
      ok(res, (rows as any[]).map(jobRow));
    } catch (e) { err(res, String(e)); }
  });

  r.post("/jobs", (req: Request, res: Response) => {
    try {
      const b = req.body;
      db.prepare(`
        INSERT INTO jobs(id,job_id,model_id,mode,n_shards,status,output_hash,latency_ms,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          status=excluded.status, output_hash=excluded.output_hash,
          latency_ms=excluded.latency_ms, updated_at=excluded.updated_at
      `).run(
        b.job_id, b.job_id, b.model_id ?? "", b.mode ?? "",
        b.n_shards ?? 0, b.status ?? "", b.output_hash ?? "",
        b.latency_ms ?? 0, Date.now(),
      );
      ok(res, { upserted: true });
    } catch (e) { err(res, String(e)); }
  });

  // ── Reputation ────────────────────────────────────────────────────────────

  r.post("/reputation", (req: Request, res: Response) => {
    try {
      const b = req.body;
      db.prepare(`
        INSERT OR IGNORE INTO reputation(id,event_id,miner,event_type,delta,job_id,shard_idx,created_at)
        VALUES(?,?,?,?,?,?,?,?)
      `).run(b.event_id, b.event_id, b.miner, b.event_type, b.delta ?? 0, b.job_id ?? "", b.shard_idx ?? 0, Date.now());
      db.prepare("UPDATE miners SET reputation=reputation+? WHERE id=?").run(b.delta ?? 0, b.miner);
      ok(res, { logged: true });
    } catch (e) { err(res, String(e)); }
  });

  r.get("/reputation/:address", (req: Request, res: Response) => {
    try {
      const rows = db.prepare("SELECT * FROM reputation WHERE miner=? ORDER BY created_at DESC").all(req.params.address);
      ok(res, (rows as any[]).map(repRow));
    } catch (e) { err(res, String(e)); }
  });

  // ── Job contexts ──────────────────────────────────────────────────────────

  r.post("/contexts", (req: Request, res: Response) => {
    try {
      const b = req.body;
      const ttl = b.ttl_ms ?? 600_000;
      db.prepare(`
        INSERT INTO contexts(id,job_id,query_text,context_text,context_hash,model_id,n_entries,expires_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          query_text=excluded.query_text, context_text=excluded.context_text,
          context_hash=excluded.context_hash, model_id=excluded.model_id,
          n_entries=excluded.n_entries, expires_at=excluded.expires_at
      `).run(b.job_id, b.job_id, b.query_text ?? "", b.context_text ?? "",
             b.context_hash ?? "", b.model_id ?? "", b.n_entries ?? 0, Date.now() + ttl);
      ok(res, { set: true });
    } catch (e) { err(res, String(e)); }
  });

  r.get("/contexts/:job_id", (req: Request, res: Response) => {
    try {
      const row = db.prepare("SELECT * FROM contexts WHERE id=? AND expires_at>?")
        .get(req.params.job_id, Date.now()) as any;
      ok(res, row ? { query_text: row.query_text, context_text: row.context_text,
        context_hash: row.context_hash, model_id: row.model_id, n_entries: row.n_entries } : null);
    } catch (e) { err(res, String(e)); }
  });

  r.delete("/contexts/expired", (_req: Request, res: Response) => {
    try {
      const info = db.prepare("DELETE FROM contexts WHERE expires_at<?").run(Date.now());
      ok(res, { deleted: info.changes });
    } catch (e) { err(res, String(e)); }
  });

  // ── Peers ─────────────────────────────────────────────────────────────────

  r.post("/peers", (req: Request, res: Response) => {
    try {
      const { peer_address, job_id, rejected } = req.body;
      db.prepare(`
        INSERT INTO peers(id,peer_address,last_seen,thoughts_received,proofs_rejected,last_job_id)
        VALUES(?,?,?,1,?,?)
        ON CONFLICT(id) DO UPDATE SET
          last_seen=excluded.last_seen, last_job_id=excluded.last_job_id,
          thoughts_received=thoughts_received+1,
          proofs_rejected=proofs_rejected+excluded.proofs_rejected
      `).run(peer_address, peer_address, Date.now(), rejected ? 1 : 0, job_id ?? "");
      ok(res, { recorded: true });
    } catch (e) { err(res, String(e)); }
  });

  return r;
}

// ── Row serialisers ────────────────────────────────────────────────────────

function thoughtRow(r: any) {
  return {
    id: r.ingested_at ?? 0,
    job_id: r.job_id ?? "",
    miner_address: r.miner_addr ?? r.miner_address ?? "",
    model_id: r.model_id ?? "",
    question_text: r.question ?? r.question_text ?? "",
    thinking_text: r.thinking ?? r.thinking_text ?? "",
    answer_text: r.answer ?? r.answer_text ?? "",
    score: typeof r.score === "number" ? r.score : 0,
  };
}

function rollupRow(r: any) {
  return {
    rollup_id: r.rollup_id,
    topic: r.topic ?? "",
    model_id: r.model_id ?? "",
    summary_text: r.summary_text ?? "",
    source_count: r.source_count ?? 0,
    source_job_ids: JSON.parse(r.source_jobs ?? "[]"),
    created_at: new Date(r.created_at ?? 0).toISOString(),
    score: typeof r.score === "number" ? r.score : 0,
  };
}

function benchRow(r: any) {
  return {
    miner_address: r.miner_addr,
    model_id: r.model_id,
    tokens_per_sec: r.tokens_per_sec ?? 0,
    live_tps: r.live_tps ?? 0,
    live_sample_count: r.live_sample_count ?? 0,
    n_tokens: r.n_tokens ?? 0,
    elapsed_ms: r.elapsed_ms ?? 0,
    nonce: r.nonce ?? "",
    block_number: r.block_number ?? 0,
    expires_at_block: r.expires_at_block ?? 0,
    last_updated: new Date(r.last_updated ?? 0).toISOString(),
  };
}

function minerRow(r: any) {
  return {
    address: r.address,
    models: JSON.parse(r.models ?? "[]"),
    backend: r.backend ?? "",
    p2p_addr: r.p2p_addr ?? "",
    max_shards: r.max_shards ?? 4,
    reputation: r.reputation ?? 0,
    last_seen: new Date(r.last_seen ?? 0).toISOString(),
  };
}

function jobRow(r: any) {
  return {
    job_id: r.job_id,
    model_id: r.model_id ?? "",
    mode: r.mode ?? "",
    n_shards: r.n_shards ?? 0,
    status: r.status ?? "",
    output_hash: r.output_hash ?? "",
    latency_ms: r.latency_ms ?? 0,
    updated_at: new Date(r.updated_at ?? 0).toISOString(),
  };
}

function repRow(r: any) {
  return {
    event_id: r.event_id,
    miner: r.miner,
    event_type: r.event_type,
    delta: r.delta ?? 0,
    job_id: r.job_id ?? "",
    shard_idx: r.shard_idx ?? 0,
    created_at: new Date(r.created_at ?? 0).toISOString(),
  };
}
