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
