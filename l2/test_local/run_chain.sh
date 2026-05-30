#!/bin/bash
# Run the local InferenceChain L2 node for two-miner split testing
set -e
cd "$(dirname "$0")/.."

export SEQUENCER_PRIVATE_KEY="4da245a36de729dcbe5263060b146e570674a384a047394fe0491015cf72095f"
export PYTHONPATH="$(pwd)"

echo "[chain] Starting local L2 node on RPC :18545 P2P :19000 health :19095"
exec /home/khadas/INFT/.venv/bin/python3 -m chain \
  --genesis "$(pwd)/test_local/genesis.json" \
  --rpc-port 18545 \
  --p2p-port 19000 \
  --health-port 19095 \
  --db "$(pwd)/test_local/data" \
  --log-level INFO
