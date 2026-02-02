# Pomai Search Engine Roadmap (Phase 2)

Phase 2 expands Pomai from flat-scan V1 into a more capable retrieval system while preserving the core API and determinism.

## 1) Persistent snapshots (optional)

Goal: warm-start from a snapshot without requiring DB-grade persistence.

- Serialize shard data (keys, vectors, metadata, norms) to a snapshot file.
- Snapshots are **best-effort** and do not imply full durability guarantees.
- Optional background snapshotting with configurable cadence.

## 2) HNSW index per shard (optional module)

- Introduce an HNSW-backed `VectorIndex` implementation.
- Allow hybrid mode: HNSW for recall + flat-scan rerank.
- Configurable `ef_search` and `M` parameters per shard.

## 3) IVF-PQ (optional module)

- IVF coarse quantizer with PQ codebooks per shard.
- Allow lower memory footprint and faster search for large collections.
- Optional re-ranking against original vectors.

## 4) Hybrid search (BM25 + vector)

- Add a text index and BM25 scoring.
- Combine BM25 and vector scores via a `ScoreFusion` interface.
- Configuration supports linear fusion or learned weights.

## 5) Improved HTTP server or gRPC

- Replace minimal parser with buffered, keep-alive capable server.
- Consider gRPC for typed clients and streaming results.

## 6) Query policy engine

- Route queries through cache tiers and reranking hooks.
- Support per-tenant query policies and guardrails.

## 7) Observability

- Metrics counters (ingest/search/delete).
- Latency histograms for ingest/search.
- Export to Prometheus or OpenTelemetry.

---

## Stub interfaces (V1-compatible)

The following interfaces are defined in `include/pomai_search/index.h` to enable Phase 2 work without rewriting V1 core logic:

- `VectorIndex`: drop-in interface for flat scan, HNSW, IVF-PQ.
- `SnapshotStore`: optional persistence for shard state.
- `ScoreFusion`: combine BM25 + vector scores.
- `QueryPolicy`: select cache/db/rerank behavior.
- `MetricsSink`: pluggable metrics collection.
