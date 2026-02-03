# Pomai Search Engine

Pomai Search is an in-memory vector search engine designed for fast similarity search and hybrid vector + keyword queries. It is **not** a source-of-truth database.

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
```

## Testing

```bash
ctest --test-dir build/pomai_search --output-on-failure
```
