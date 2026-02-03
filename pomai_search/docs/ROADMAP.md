# Roadmap

## Phase B (completed)

- Pluggable index interface with FlatIndex and HNSW implementations.
- Hybrid search: vector + keyword scoring with TF-IDF fusion.
- Deterministic search results across shards and threads.
- Recall benchmark and latency reporting.
- Observability: metrics counters and histogram buckets.

## Phase C (completed)

- Deterministic search contract + explain-first APIs.
- Policy-first planner and staged retrieval.
- Replayable snapshots and contract tests.
- Quality-aware benchmarks.

## Next

- IVF-PQ for FAISS-scale recall/latency tradeoffs.
- PQ rerank + hybrid reranking hooks.
- gRPC transport and better HTTP pipeline.
