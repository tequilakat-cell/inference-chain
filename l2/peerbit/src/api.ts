/**
 * REST API that the Python miners call to read/write the Peerbit DB.
 *
 * All endpoints are local-only (bind to 127.0.0.1 by default).
 * No authentication needed since this is a local sidecar.
 *
 * Base URL: http://127.0.0.1:7700
 */

import express, { Request, Response } from "express";
import { InferenceChainDB } from "./db.js";
import type { MinerProfile, JobRecord, ReputationEvent } from "./schema.js";

export function buildApi(db: InferenceChainDB): express.Application {
    const app = express();
    app.use(express.json());
    // CORS — allow browser dashboards on any origin to read the sidecar
    app.use((_req, res, next) => {
        res.setHeader("Access-Control-Allow-Origin",  "*");
        res.setHeader("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE, OPTIONS");
        res.setHeader("Access-Control-Allow-Headers", "Content-Type");
        next();
    });
    app.options("*", (_req, res) => res.sendStatus(204));

    // ── Health ────────────────────────────────────────────────────────────────
    app.get("/health", (_req: Request, res: Response) => {
        res.json({ status: "ok", ...db.peerInfo() });
    });

    // ── Miners ────────────────────────────────────────────────────────────────

    // List all (optionally filter by model)
    app.get("/miners", async (req: Request, res: Response) => {
        try {
            const model = req.query.model as string | undefined;
            const miners = model
                ? await db.minersForModel(model)
                : await db.listMiners();
            res.json(miners);
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    // Get one
    app.get("/miners/:address", async (req: Request, res: Response) => {
        try {
            const miner = await db.getMiner(req.params.address);
            if (!miner) { res.status(404).json({ error: "not found" }); return; }
            res.json(miner);
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    // Upsert (miner calls this on startup + periodic heartbeat)
    app.put("/miners/:address", async (req: Request, res: Response) => {
        try {
            const profile: MinerProfile = {
                address: req.params.address, models: "", backend: "cpu",
                reputation: 500, lastSeen: Date.now(), p2pAddr: "",
                l2ChainId: "2026", active: true, maxShards: 4,
                ...req.body,
            };
            await db.upsertMiner(profile);
            res.json({ ok: true });
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    // Deactivate (miner calls this on shutdown)
    app.delete("/miners/:address", async (req: Request, res: Response) => {
        try {
            await db.deleteMiner(req.params.address);
            res.json({ ok: true });
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    // ── Jobs ──────────────────────────────────────────────────────────────────

    // List jobs, optionally filter by status
    app.get("/jobs", async (req: Request, res: Response) => {
        try {
            const status = req.query.status as string | undefined;
            const jobs = await db.listJobs(status);
            res.json(jobs);
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    app.get("/jobs/:jobId", async (req: Request, res: Response) => {
        try {
            const job = await db.getJob(req.params.jobId);
            if (!job) { res.status(404).json({ error: "not found" }); return; }
            res.json(job);
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    app.put("/jobs/:jobId", async (req: Request, res: Response) => {
        try {
            const record: JobRecord = {
                jobId: req.params.jobId, modelId: "", mode: "parallel_sample",
                nShards: 1, postedAt: Date.now(), status: "pending",
                outputHash: "", requester: "", completedAt: 0, latencyMs: 0,
                ...req.body,
            };
            await db.upsertJob(record);
            res.json({ ok: true });
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    // ── Reputation events ─────────────────────────────────────────────────────

    app.get("/events/:address", async (req: Request, res: Response) => {
        try {
            const events = await db.eventsForMiner(req.params.address);
            res.json(events);
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    app.post("/events", async (req: Request, res: Response) => {
        try {
            const { eventId, miner } = req.body;
            if (!eventId || !miner) {
                res.status(400).json({ error: "eventId and miner required" });
                return;
            }
            const event: ReputationEvent = {
                eventId, miner, eventType: "shard_complete", delta: 0,
                jobId: "", shardIdx: 0, timestamp: Date.now(), signature: "",
                ...req.body,
            };
            await db.appendEvent(event);
            res.json({ ok: true });
        } catch (err) {
            res.status(500).json({ error: String(err) });
        }
    });

    return app;
}
