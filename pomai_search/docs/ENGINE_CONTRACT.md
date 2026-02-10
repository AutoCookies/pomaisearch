# Pomai Search Engine Contract

This document is the **single source of truth** for behavior and guarantees.

## Core Guarantees

### Data Model

- Keys are unique within a shard (hash partitioning by key).
- A key maps to a single vector + metadata + optional text.
- Upsert is idempotent: the latest write replaces previous state for that key.

### Consistency & Concurrency

- **Search vs. Upsert/Delete**: a search observes either the state **before** or **after** a concurrent write (no torn reads).
- Indexes are protected by per-index reader/writer locks.
- VectorStore reads are protected by shared locks during scoring.
- No shard-wide lock is held during scoring.

### Similarity

- **Dot**: raw vectors are stored and scored directly.
- **Cosine**: vectors are normalized **on ingest**, queries are normalized **on search**.
- All index types operate on the same normalized representation for cosine.

### TTL Semantics

- Expired documents are **always filtered** from results.
- Expired documents are **reclaimed** on subsequent writes via incremental sweeps.

### Memory Lifecycle

- VectorStore reuses slots for upserts and deletes (free list).
- Memory usage grows to a high-water mark and then stabilizes under steady-state updates.
- Deleting a document or expiring its TTL releases its slot for reuse.

### Snapshot Durability

- Snapshots are atomic and crash-safe:
  - Written to `*.tmp`, `fsync`'d, renamed, directory `fsync`'d.
- Snapshots include payload CRC32 and size.
- Corrupt or partial snapshots fail fast on load.

## Non-Guarantees

- No write-ahead log (WAL). Data since the last snapshot may be lost after a crash.
- No strict real-time TTL cleanup without writes (reclaim happens on subsequent writes).
- No distributed replication or consensus.
- Recall/latency guarantees depend on index parameters and dataset characteristics.

## Lock Ordering

To avoid deadlocks, **all writes** acquire locks in this order:

```
Shard (doc table) → Index → VectorStore
```

