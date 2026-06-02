#!/usr/bin/env bash
# Build the llama.cpp submodule and produce llama-jacobi + libllama.
# Run this once after cloning inference-chain (or after updating the submodule).
#
#   git clone git@github.com:tequilakat-cell/inference-chain.git
#   cd inference-chain
#   git submodule update --init --recursive
#   bash build-llama.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

echo "[build-llama] submodule: $LLAMA_DIR"

if [ ! -f "$LLAMA_DIR/CMakeLists.txt" ]; then
    echo "[build-llama] submodule not initialised — running: git submodule update --init"
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

echo "[build-llama] configuring..."
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_METAL=ON \
    -DGGML_METAL=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    "$@"                          # pass extra -D flags through

JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
echo "[build-llama] building with $JOBS jobs..."
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

# Symlink llama-jacobi to the repo root for easy discovery
BIN="$BUILD_DIR/bin/llama-jacobi"
if [ -f "$BIN" ]; then
    ln -sf "$BIN" "$LLAMA_DIR/llama-jacobi"
    echo "[build-llama] llama-jacobi → $BIN"
fi

echo "[build-llama] done. Binaries in $BUILD_DIR/bin/"
