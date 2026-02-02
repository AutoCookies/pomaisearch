# Pomai Search Engine (Phase 1)

Pomai Search Engine is a standalone, in-memory vector search engine focused on retrieval and ranking. It provides:

- Flat scan vector search with dot or cosine similarity.
- Metadata filters (`tag`, `source`, `lang`).
- Multi-shard, multi-threaded query execution.
- AVX2-optimized kernels with scalar fallback.
- HTTP server (`pomai-searchd`) and CLI client (`pomai-search`).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

### Optional Sanitizers

```bash
cmake -S . -B build-asan -DPOMAI_SEARCH_ENABLE_ASAN=ON
cmake --build build-asan -j
```

## Run Server

```bash
./build/pomai_search/pomai-searchd --dim 128 --port 8080 --similarity cosine --shards 4 --threads 4
```

## CLI Examples

```bash
./build/pomai_search/pomai-search upsert --key doc1 --vec "0.1,0.2,0.3,0.4" --tag work
./build/pomai_search/pomai-search search --vec "0.1,0.2,0.3,0.4" --topk 5
./build/pomai_search/pomai-search stats
```

## HTTP API

- `POST /v1/upsert`
- `POST /v1/delete`
- `POST /v1/search`
- `POST /v1/search_by_key`
- `GET /v1/stats`

Example:

```bash
curl -X POST http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1,0.2,0.3,0.4], "topk": 3}'
```

## Tests

```bash
./build/pomai_search/pomai-search-tests
```

## Benchmarks

```bash
./build/pomai_search/pomai-search-bench
```
