# Pomai Search Engine

Pomai Search is an in-memory vector search engine designed for fast similarity search and hybrid vector + keyword queries. It is **not** a source-of-truth database.

## Why Pomai Search is not FAISS

Pomai Search is a full **search engine** with a deterministic contract, replayable snapshots, and policy-first planning. FAISS is a similarity library; Pomai adds:

- Deterministic, contract-driven ranking (stable ordering + NaN-safe scores).
- Explain plans with per-stage budgets and per-result score breakdowns.
- Policy-first planners that enforce latency and candidate budgets.
- Replayable snapshot + contract tests for regression gating.

In short: Pomai returns a contract, not a surprise.

## Determinism contract

Pomai guarantees:

- Stable hash routing via `StableHash64` (no `std::hash`).
- Deterministic tie-break: score desc, key asc, internal_id asc.
- NaN-safe scoring (NaN treated as `-inf`).
- Deterministic JSON serialization order for explain/contract/metrics.

Every ranking change must bump `contract_version` and regenerate contracts.

## Quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run the server

```bash
./build/pomai_search/pomai-searchd --dim 3 --shards 2 --port 8080 --index flat
```

### Upsert a document

```bash
./build/pomai_search/pomai-search --command upsert --key doc-1 --vector 1,0,0 --metadata tag=demo --text "hello world"
```

### Vector search

```bash
./build/pomai_search/pomai-search --command search --vector 1,0,0 --topk 5
```

### Hybrid search

```bash
./build/pomai_search/pomai-search --command search_hybrid --vector 1,0,0 --text "hello" --alpha 0.7 --topk 5
```

### Explain search

```bash
curl -s -X POST http://localhost:8080/v1/search_explain \
  -H 'Content-Type: application/json' \
  -d '{"vector":[1,0,0],"topk":5,"policy":{"max_latency_ms":50,"max_candidates":50}}'
```

### Policy usage

```bash
curl -s -X POST http://localhost:8080/v1/search_hybrid_explain \
  -H 'Content-Type: application/json' \
  -d '{
    "vector":[1,0,0],
    "text_query":"hello",
    "alpha":0.6,
    "topk":5,
    "policy":{"max_latency_ms":25,"max_candidates":100,"recall_bias":0.7,"fusion_method":"rrf"}
  }'
```

### Contract record/verify workflow

```bash
./build/pomai_search/pomai-search --command contract-record \
  --dataset tests/contracts/dataset.json \
  --queries tests/contracts/queries.json \
  --out tests/contracts

./build/pomai_search/pomai-search --command contract-verify --dir tests/contracts
```

### Snapshot usage

Snapshots are lightweight engine saves for replay:

```bash
# in code: SnapshotWriter::Write(engine, "snapshot.bin")
# in code: auto engine = SnapshotReader::Read("snapshot.bin")
```

## Build options

- `POMAI_SEARCH_ENABLE_AVX2` (default ON)
- `POMAI_SEARCH_ENABLE_ASAN` (default OFF)
- `POMAI_SEARCH_ENABLE_TSAN` (default OFF)
- `POMAI_SEARCH_ENABLE_UBSAN` (default OFF)

## Benchmark examples

```bash
./build/pomai_search/pomai_search_bench_flat --n 10000 --dim 64 --queries 1000 --topk 10 --shards 4
./build/pomai_search/pomai_search_bench_hnsw --n 10000 --dim 64 --queries 1000 --topk 10 --shards 4
./build/pomai_search/pomai_search_bench_recall --n 20000 --dim 64 --queries 500 --topk 10
./build/pomai_search/pomai_search_bench_quality
./build/pomai_search/pomai_search_bench_simd
```

## Testing

```bash
ctest --test-dir build/pomai_search --output-on-failure
```
