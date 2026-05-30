/**
 * InferenceChain OrbitDB sidecar — distributed miner registry and job board.
 *
 * Starts an OrbitDB node backed by libp2p + Helia (IPFS).
 * Data is automatically replicated to all other sidecar nodes via gossipsub.
 * Exposes a local REST API on port 7700 for the Python miners to call.
 *
 * Requirements: Node.js v22+
 *
 * Usage:
 *   ORBITDB_PORT=7700 BOOTSTRAP_ADDRS=/ip4/... node dist/index.js
 */

import { InferenceChainDB } from "./db.js";
import { buildApi }         from "./api.js";

const PORT       = parseInt(process.env.ORBITDB_PORT ?? "7700", 10);
const DATA_DIR   = process.env.ORBITDB_DATA ?? "./data/orbitdb";
const BOOTSTRAP  = (process.env.BOOTSTRAP_ADDRS ?? "").split(",").filter(Boolean);

async function main(): Promise<void> {
    console.log("[orbitdb-sidecar] starting (Node.js", process.version + ")");
    console.log("[orbitdb-sidecar] data dir:", DATA_DIR);
    console.log("[orbitdb-sidecar] REST port:", PORT);
    console.log("[orbitdb-sidecar] bootstrap peers:", BOOTSTRAP.length || "(none — mDNS auto-discovery active)");

    const db = new InferenceChainDB();
    await db.open(DATA_DIR, BOOTSTRAP);

    const app = buildApi(db);
    const server = app.listen(PORT, () => {
        console.log(`[orbitdb-sidecar] REST API → http://127.0.0.1:${PORT}`);
        const info = db.peerInfo();
        console.log("[orbitdb-sidecar] peer id:", info.peerId.slice(0, 20) + "…");
        console.log("[orbitdb-sidecar] listening on:", info.addresses.join(", ") || "(none)");
    });

    const shutdown = async (signal: string): Promise<void> => {
        console.log(`[orbitdb-sidecar] ${signal} received — closing...`);
        server.close();
        await db.close();
        process.exit(0);
    };
    process.on("SIGTERM", () => shutdown("SIGTERM"));
    process.on("SIGINT",  () => shutdown("SIGINT"));
}

main().catch(err => {
    console.error("[orbitdb-sidecar] fatal:", err);
    process.exit(1);
});
