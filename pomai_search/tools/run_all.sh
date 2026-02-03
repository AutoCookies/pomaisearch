#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/../build"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j
ctest --test-dir "$BUILD_DIR" --output-on-failure

"$BUILD_DIR/pomai_search_bench_flat" --n 1000 --dim 32 --queries 50 --topk 5
"$BUILD_DIR/pomai_search_bench_hnsw" --n 1000 --dim 32 --queries 50 --topk 5
"$BUILD_DIR/pomai_search_bench_recall" --n 1000 --dim 32 --queries 50 --topk 5
