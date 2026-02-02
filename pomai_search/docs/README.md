# Pomai Search Engine (V1 Hardening)

Pomai Search Engine is an in-memory **vector search engine**, not a database. It focuses on retrieval and ranking of dense embeddings with deterministic results, predictable latency, and production-grade safety checks.

Key traits:
- Flat-scan vector search with dot or cosine similarity.
- Deterministic ranking (score desc, key asc).
- Multi-shard design with configurable concurrency.
- AVX2+FMA kernels with runtime dispatch and scalar fallback.
- HTTP server (`pomai-searchd`) and CLI client (`pomai-search`).

## Quickstart

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

### Run server

```bash
./build/pomai_search/pomai-searchd --dim 128 --port 8080 --similarity cosine --shards 4 --threads 4
```

### CLI examples

```bash
./build/pomai_search/pomai-search --host 127.0.0.1 --port 8080 --dim 4 \
  upsert --key doc1 --vec "0.1,0.2,0.3,0.4" --tag work

./build/pomai_search/pomai-search --host 127.0.0.1 --port 8080 --dim 4 \
  search --vec "0.1,0.2,0.3,0.4" --topk 5

./build/pomai_search/pomai-search --host 127.0.0.1 --port 8080 stats
```

## HTTP API (JSON)

### `POST /v1/upsert`

```json
{
  "key": "doc1",
  "vector": [0.1, 0.2, 0.3, 0.4],
  "metadata": { "tag": "work", "source": "docs" },
  "ttl_ms": 60000
}
```

### `POST /v1/search`

```json
{
  "vector": [0.1, 0.2, 0.3, 0.4],
  "topk": 3,
  "scope": "global",
  "filter": { "tag": "work" }
}
```

### `POST /v1/search_by_key`

```json
{
  "key": "doc1",
  "topk": 3,
  "scope": "local"
}
```

**Scope behavior**

- `local` for vector search targets shard 0.
- `local` for search-by-key targets the key owner shard.

### `GET /v1/stats`

```json
{
  "num_points": 1000,
  "num_deleted": 3,
  "last_query_ms_p50": 1.23
}
```

## Performance notes

- The scalar and AVX2 kernels are compiled in separate translation units.
- AVX2 is **enabled at runtime only if** the CPU and OS support it.
- For cosine similarity, per-vector norms are cached to avoid repeated work.
- Query results are deterministic: primary sort by score descending, tie-break by key ascending.

## Limitations (V1)

- Flat scan only (no ANN index).
- In-memory only (no persistence or snapshots).
- HTTP parser is intentionally lightweight and supports one request per connection.
- No distributed clustering or replication.

## Benchmarks

```bash
./build/pomai_search/pomai_search_bench --n 20000 --dim 128 --queries 2000 --topk 10 \
  --shards 4 --threads 4 --similarity cosine --avx2 both --seed 42
```

## Tests

```bash
./build/pomai_search/pomai_search_tests
```
