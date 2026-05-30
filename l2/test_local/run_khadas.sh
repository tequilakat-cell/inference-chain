#!/bin/bash
# Run the Khadas Edge 2 miner (address 0x80fc28...)
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)"
export PRIVATE_KEY="0e78d827a32f4546c52c9c7b02354adcf9b3c78045a76b2ad57f85e342f700bf"
# Vulkan ICD: prefer Panfrost (Mali GPU) over software renderer
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/panfrost_icd.json:/usr/share/vulkan/icd.d/lvp_icd.json"

echo "[khadas] Starting Khadas Edge 2 miner on P2P :19002 health :19092 backend=vulkan"
exec /home/khadas/INFT/.venv/bin/python3 -m miner \
  --config "$(pwd)/test_local/config_khadas.json"
