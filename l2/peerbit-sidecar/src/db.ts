/**
 * SQLite storage layer. Replaces Peerbit as the local store;
 * P2P sync is handled by the existing inference-chain gossip layer.
 */
import Database from "better-sqlite3";
import fs from "fs";
import path from "path";

export function openDb(dir: string): Database.Database {
  fs.mkdirSync(dir, { recursive: true });
  const db = new Database(path.join(dir, "sidecar.db"));
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS thoughts (
      id          TEXT PRIMARY KEY,
      job_id      TEXT NOT NULL,
      miner_addr  TEXT NOT NULL DEFAULT '',
      model_id    TEXT NOT NULL DEFAULT '',
      question    TEXT DEFAULT '',
      thinking    TEXT DEFAULT '',
      answer      TEXT DEFAULT '',
      block_num   INTEGER DEFAULT 0,
      ingested_at INTEGER NOT NULL,
      embedding   TEXT DEFAULT NULL,
      proof_sig   TEXT DEFAULT NULL,
      tx_hash     TEXT DEFAULT NULL,
      peer_origin TEXT DEFAULT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_th_time  ON thoughts(ingested_at DESC);
    CREATE INDEX IF NOT EXISTS idx_th_model ON thoughts(model_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS thoughts_fts
      USING fts5(job_id UNINDEXED, question, thinking, answer,
                 content='thoughts', content_rowid='rowid');
    CREATE TRIGGER IF NOT EXISTS th_ai AFTER INSERT ON thoughts BEGIN
      INSERT INTO thoughts_fts(rowid, job_id, question, thinking, answer)
      VALUES (new.rowid, new.job_id, new.question, new.thinking, new.answer);
    END;
    CREATE TRIGGER IF NOT EXISTS th_ad AFTER DELETE ON thoughts BEGIN
      INSERT INTO thoughts_fts(thoughts_fts, rowid, job_id, question, thinking, answer)
      VALUES ('delete', old.rowid, old.job_id, old.question, old.thinking, old.answer);
    END;

    CREATE TABLE IF NOT EXISTS rollups (
      id           TEXT PRIMARY KEY,
      rollup_id    TEXT NOT NULL,
      topic        TEXT NOT NULL DEFAULT '',
      model_id     TEXT NOT NULL DEFAULT '',
      summary_text TEXT NOT NULL DEFAULT '',
      source_count INTEGER NOT NULL DEFAULT 0,
      source_jobs  TEXT NOT NULL DEFAULT '[]',
      created_at   INTEGER NOT NULL,
      embedding    TEXT DEFAULT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rl_time  ON rollups(created_at DESC);

    CREATE TABLE IF NOT EXISTS benchmarks (
      id               TEXT PRIMARY KEY,
      miner_addr       TEXT NOT NULL,
      model_id         TEXT NOT NULL,
      tokens_per_sec   REAL NOT NULL DEFAULT 0,
      live_tps         REAL NOT NULL DEFAULT 0,
      live_sample_count INTEGER NOT NULL DEFAULT 0,
      n_tokens         INTEGER NOT NULL DEFAULT 0,
      elapsed_ms       INTEGER NOT NULL DEFAULT 0,
      nonce            TEXT NOT NULL DEFAULT '',
      block_number     INTEGER NOT NULL DEFAULT 0,
      expires_at_block INTEGER NOT NULL DEFAULT 0,
      last_updated     INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS miners (
      id        TEXT PRIMARY KEY,
      address   TEXT NOT NULL,
      models    TEXT NOT NULL DEFAULT '[]',
      backend   TEXT NOT NULL DEFAULT '',
      p2p_addr  TEXT NOT NULL DEFAULT '',
      chain_id  TEXT NOT NULL DEFAULT '2026',
      max_shards INTEGER NOT NULL DEFAULT 4,
      reputation REAL NOT NULL DEFAULT 0,
      active    INTEGER NOT NULL DEFAULT 1,
      last_seen INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS jobs (
      id          TEXT PRIMARY KEY,
      job_id      TEXT NOT NULL,
      model_id    TEXT NOT NULL DEFAULT '',
      mode        TEXT NOT NULL DEFAULT '',
      n_shards    INTEGER NOT NULL DEFAULT 0,
      status      TEXT NOT NULL DEFAULT '',
      output_hash TEXT NOT NULL DEFAULT '',
      latency_ms  INTEGER NOT NULL DEFAULT 0,
      updated_at  INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jb_time ON jobs(updated_at DESC);

    CREATE TABLE IF NOT EXISTS reputation (
      id          TEXT PRIMARY KEY,
      event_id    TEXT NOT NULL,
      miner       TEXT NOT NULL,
      event_type  TEXT NOT NULL,
      delta       REAL NOT NULL DEFAULT 0,
      job_id      TEXT NOT NULL DEFAULT '',
      shard_idx   INTEGER NOT NULL DEFAULT 0,
      created_at  INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rep_miner ON reputation(miner);

    CREATE TABLE IF NOT EXISTS contexts (
      id           TEXT PRIMARY KEY,
      job_id       TEXT NOT NULL,
      query_text   TEXT NOT NULL DEFAULT '',
      context_text TEXT NOT NULL DEFAULT '',
      context_hash TEXT NOT NULL DEFAULT '',
      model_id     TEXT NOT NULL DEFAULT '',
      n_entries    INTEGER NOT NULL DEFAULT 0,
      expires_at   INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS peers (
      id                TEXT PRIMARY KEY,
      peer_address      TEXT NOT NULL,
      last_seen         INTEGER NOT NULL,
      thoughts_received INTEGER NOT NULL DEFAULT 0,
      proofs_rejected   INTEGER NOT NULL DEFAULT 0,
      last_job_id       TEXT NOT NULL DEFAULT ''
    );
  `);
  return db;
}
