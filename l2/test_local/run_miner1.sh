#!/bin/bash
# Run miner1 (address 0x8fd3...) — works on Khadas, Mac, and Linux
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)"
export PRIVATE_KEY="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

# Auto-detect Python: prefer the project venv, fall back to system python3
if [ -f "$HOME/INFT/.venv/bin/python3" ]; then
  PYTHON="$HOME/INFT/.venv/bin/python3"
elif [ -f "$(pwd)/.venv/bin/python3" ]; then
  PYTHON="$(pwd)/.venv/bin/python3"
else
  PYTHON="python3"
fi

echo "[miner1] Starting miner on P2P :19001 health :19091 (python: $PYTHON)"
exec "$PYTHON" -m miner \
  --config "$(pwd)/test_local/config_miner1.json"
