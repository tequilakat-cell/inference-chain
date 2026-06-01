import express from "express";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
import { Peerbit } from "peerbit";
import { InferenceChainProgram } from "./program.js";
import { buildRouter } from "./routes.js";

const HTTP_PORT = parseInt(process.env.PEERBIT_HTTP_PORT ?? "7731");
const P2P_PORT = parseInt(process.env.PEERBIT_P2P_PORT ?? "9011");
const BOOTSTRAP = (process.env.PEERBIT_BOOTSTRAP ?? "").split(",").filter(Boolean);
const DATA_DIR = path.join(os.homedir(), ".peerbit-sidecar");
const ADDR_FILE = path.join(DATA_DIR, "program.addr");

// Accept --address <addr> to connect to an existing program
const addrArgIdx = process.argv.indexOf("--address");
const addrArg = addrArgIdx !== -1 ? process.argv[addrArgIdx + 1] : undefined;

async function main() {
  fs.mkdirSync(DATA_DIR, { recursive: true });

  const peer = await Peerbit.create({
    directory: path.join(DATA_DIR, "peerbit"),
    listenPort: P2P_PORT,
  });

  for (const addr of BOOTSTRAP) {
    try {
      await peer.dial(addr);
      console.log(`[peerbit] dialed bootstrap: ${addr}`);
    } catch (e) {
      console.warn(`[peerbit] bootstrap dial failed: ${addr} — ${e}`);
    }
  }

  // Determine program address: CLI arg > saved file > create new
  let program: InferenceChainProgram;
  const savedAddr = fs.existsSync(ADDR_FILE)
    ? fs.readFileSync(ADDR_FILE, "utf-8").trim()
    : undefined;
  const openAddr = addrArg ?? savedAddr;

  if (openAddr) {
    console.log(`[peerbit] opening existing program: ${openAddr}`);
    program = await peer.open<InferenceChainProgram>(openAddr as any);
  } else {
    console.log("[peerbit] creating new program...");
    program = await peer.open(new InferenceChainProgram());
    const addr = program.address.toString();
    fs.writeFileSync(ADDR_FILE, addr);
    console.log(`[peerbit] program address: ${addr}`);
    console.log(`[peerbit] address saved to ${ADDR_FILE}`);
    console.log(`[peerbit] distribute this address to all miners via peerbit_program_addr config`);
  }

  const app = express();
  app.use(express.json({ limit: "10mb" }));
  app.use(express.static(path.join(__dirname, "../public")));
  app.use(buildRouter(program));

  const server = app.listen(HTTP_PORT, "127.0.0.1", () => {
    console.log(`[peerbit-sidecar] HTTP API listening on http://127.0.0.1:${HTTP_PORT}`);
  });

  const shutdown = async () => {
    console.log("[peerbit-sidecar] shutting down...");
    server.close();
    await peer.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((e) => {
  console.error("[peerbit-sidecar] fatal:", e);
  process.exit(1);
});
