# Pomai Search Engine Contract

## What Pomai Search IS
Pomai Search is a deterministic, embeddable, single-node vector search engine with a canonical search pipeline and optional system explain traces.

## What Pomai Search IS NOT
- Not a database.
- Not a WAL/recovery/durability system.
- Not a distributed service.
- Not a multi-path ANN experimentation framework.

## Guarantees
- **Correctness**: final ranking is computed by exact scoring over gathered candidates.
- **Recall**: bounded by configured routing/candidate limits; approximate routing may miss true global nearest neighbors.
- **Determinism**: same checkpoint + same query + same config yields stable ordering (score, key, id tie-break).
- **Durability**: no durability guarantee; checkpoint files are for warm start/reproducibility only.
- **Concurrency**: shard-local synchronization protects concurrent upsert/delete/search operations.
- **Checkpoint**: corrupted checkpoint is rejected; caller must rebuild from source data.

## Non-guarantees
- Cross-process transactional consistency.
- Crash recovery without external source of truth.
- Infinite-memory or unbounded-latency behavior.


## Phase 2 Stability Closure
- IVF-Flat and IVF-SQ8 upserts are generation-versioned; stale postings are filtered at gather-time and compacted list-locally.
- Compaction triggers: stale ratio threshold, max list entries, or max list bytes.
- Deterministic degradation: bounded max candidates + bounded rerank factor + bounded IVF probes.
- Search rerank now snapshots candidate metadata under shard lock, scores outside the lock, and validates generation before return.
