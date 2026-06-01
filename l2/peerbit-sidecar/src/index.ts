import express from "express";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { openDb } from "./db.js";
import { buildRouter } from "./routes.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTTP_PORT = parseInt(process.env.PEERBIT_HTTP_PORT ?? "7731");
const DATA_DIR = process.env.PEERBIT_DATA_DIR ?? path.join(os.homedir(), ".peerbit-sidecar");

const db = openDb(DATA_DIR);
const app = express();

app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "../public")));
app.use(buildRouter(db));

const server = app.listen(HTTP_PORT, "127.0.0.1", () => {
  console.log(`[sidecar] listening on http://127.0.0.1:${HTTP_PORT}`);
  console.log(`[sidecar] data dir: ${DATA_DIR}`);
  console.log(`[sidecar] explorer: http://127.0.0.1:${HTTP_PORT}/`);
});

const shutdown = () => {
  server.close();
  db.close();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
