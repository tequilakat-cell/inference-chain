#!/usr/bin/env bash
set -euo pipefail

# build.sh — build the forked llama.cpp with native Jacobi parallel decoding.
#
# Prerequisites:
#   cmake >= 3.16, a C++17 compiler (gcc >= 9, clang >= 10).
#   The forked llama.cpp lives at l2/jacobi/llama.cpp/.
#
# Usage:
#   bash build.sh                    # default: Release, -j$(nproc)
#   bash build.sh Debug              # debug build
#   bash build.sh Release -j4        # explicit parallel build

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FORK_DIR="$SCRIPT_DIR/llama.cpp"
BUILD_TYPE="${1:-Release}"
BUILD_JOBS="${2:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

echo "=== jacobi fork build ==="
echo "  fork dir   : $FORK_DIR"
echo "  build type : $BUILD_TYPE"
echo "  jobs       : $BUILD_JOBS"

if [ ! -f "$FORK_DIR/CMakeLists.txt" ]; then
    echo "ERROR: forked llama.cpp not found at $FORK_DIR"
    echo "  Run: git clone https://github.com/ggerganov/llama.cpp.git $FORK_DIR"
    exit 1
fi

BUILD_DIR="$FORK_DIR/build"

cmake -S "$FORK_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

# Build only the llama library + jacobi-server example
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" \
    --target llama jacobi-server -j "$BUILD_JOBS"

echo ""
echo "=== done ==="
echo "  lib:     $(find "$BUILD_DIR" -name 'libllama.*' -type f | head -1)"
echo "  binary:  $BUILD_DIR/bin/jacobi-server"
ls -lh "$BUILD_DIR/bin/jacobi-server" 2>/dev/null || echo "  (binary in examples/jacobi-server/)"
