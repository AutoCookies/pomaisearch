# Roadmap

## Phase B (current)

- Pluggable index interface with FlatIndex and HNSW implementations.
- Hybrid search: vector + keyword scoring with TF-IDF fusion.
- Deterministic search results across shards and threads.
- Recall benchmark and latency reporting.
- Observability: metrics counters and histogram buckets.

## Phase C (future)

- HNSW compaction/rebuild for deleted vectors.
- Persistent storage with snapshotting.
- Quantization and compression.
- Distributed query routing and replication.
- AuthN/AuthZ, rate limits, and production hardening.
