# Pomai Search

Pomai Search is a deterministic, explainable, safe-by-design vector search engine for single-node embedding workloads.

## Positioning
- ✅ Canonical-path ANN search engine.
- ❌ Not a database.
- ❌ No WAL / crash-recovery promises.
- ❌ No distributed consensus or replication.

## Canonical pipeline
`Query -> Coarse Routing -> Bounded Candidate Gather -> Exact Rerank -> Top-K`

Index families (Flat/HNSW/IVF) are routing implementations only; returned ranking is exact over gathered candidates.

## Contract-first docs
- `docs/ENGINE_CONTRACT.md`
- `docs/CANONICAL_SEARCH_PATH.md`
- `docs/SEARCH_BEHAVIOR_CONTRACT.md`
- `docs/CHECKPOINT_MODEL.md`
- `docs/SEARCH_SAFETY_RAILS.md`
- `docs/EXPLAIN_TRACE.md`
- `docs/CONCURRENCY_MODEL.md`


## Phase 2 Stability Closure
- IVF-Flat and IVF-SQ8 upserts are generation-versioned; stale postings are filtered at gather-time and compacted list-locally.
- Compaction triggers: stale ratio threshold, max list entries, or max list bytes.
- Deterministic degradation: bounded max candidates + bounded rerank factor + bounded IVF probes.
- Search rerank now snapshots candidate metadata under shard lock, scores outside the lock, and validates generation before return.
