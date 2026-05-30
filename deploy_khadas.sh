#!/bin/bash
# Deploy latest miner code to Khadas Edge 2 (shard 1 worker)
# Run with:  bash deploy_khadas.sh   (from inference-chain/ root)
# NOTE: Khadas machine must also have the repo at ~/inference-chain/
set -e

KHADAS_HOST="khadas"
KHADAS_USER="khadas"
REMOTE="${KHADAS_USER}@${KHADAS_HOST}"
REMOTE_L1="~/inference-chain/l1"
REMOTE_L2="~/inference-chain/l2"
REMOTE_VENV="~/inference-chain/.venv"

echo "=== Deploying l1/miner backend to Khadas ==="
scp "$(dirname "$0")/l1/miner/backends/llama_cpp_backend.py" \
    "${REMOTE}:${REMOTE_L1}/miner/backends/llama_cpp_backend.py"

echo "=== Restarting Khadas miner ==="
ssh "${REMOTE}" "
    pkill -9 -f 'miner --config' 2>/dev/null || true
    pkill -9 -f 'rpc-server' 2>/dev/null || true
    sleep 2
    cd ${REMOTE_L2}
    nohup ${REMOTE_VENV}/bin/python3 -m miner \
        --config ${REMOTE_L2}/test_local/config_khadas.json \
        >> ~/miner.log 2>&1 &
    sleep 2
    pgrep -a python3 && echo 'Khadas miner started' || echo 'WARNING: miner did not start'
    tail -5 ~/miner.log
"

echo "=== Done ==="
