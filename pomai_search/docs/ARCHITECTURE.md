# Architecture Guide

This document describes the **current** Pomai Search Engine architecture and its operational invariants.

## Overview

Pomai Search is an embeddable, sharded vector search engine with optional keyword search. Each shard owns its data and index, allowing parallel search and ingest.

```
SearchEngine
├── Shard[0..N)
│   ├── VectorStore (block-allocated, reusable slots)
│   ├── Index (Flat / HNSW / IVF-Flat / IVF-SQ8)
│   ├── KeywordIndex (token→posting list)
│   └── Doc table (key→id, metadata, TTL)
└── Thread pools (query, ingest)
```

## Sharding Model

Shard assignment is deterministic:

```
shard_id = StableHash64(key) % num_shards
```

Shards are **independent**: there is no cross-shard lock or shared mutable state.

## Vector Storage

### Block-Allocated VectorStore

VectorStore stores vectors in fixed-size blocks, each block containing `block_vectors` vectors. Each vector is addressed by a **slot id** (not a byte offset). The store maintains:

- **Live slots** (current vectors)
- **Free list** (slots eligible for reuse)

This prevents unbounded memory growth during upserts and deletes. See `docs/vector_lifecycle.md` for details.

### Similarity Normalization

When similarity is **Cosine**:

- Vectors are normalized on **ingest**.
- Queries are normalized on **search**.

All indices operate on normalized vectors for cosine similarity, ensuring consistent scoring across index types.

## Index Layer

Each shard owns exactly one index implementation:

- **FlatIndex**: brute-force scan.
- **HnswIndex**: graph-based ANN.
- **IvfFlatIndex**: coarse quantization + exact scan.
- **IvfSq8Index**: coarse quantization + scalar-quantized scan + refinement.

Index implementations are thread-safe via internal `shared_mutex` guards.

## Locking Model

### Lock Order

All write paths use the same lock order to avoid deadlocks:

```
Shard (doc table) lock → Index lock → VectorStore lock
```

### Search Path

Search does **not** take a shard-global lock while scoring. It:

1. Acquires the index shared lock.
2. Acquires a VectorStore shared lock (to protect vector reads).
3. Scores candidates.
4. Acquires the shard shared lock only for doc materialization.

This ensures:

- No long shard-wide lock during scoring.
- Readers do not block each other.

## TTL Handling

TTL is enforced in two ways:

- **Filtering**: expired docs are skipped at query time.
- **Lifecycle**: expired docs are reclaimed on subsequent writes via incremental GC sweeps.

See `docs/vector_lifecycle.md`.

## Snapshot Serialization

Snapshots are **versioned**, **checksummed**, and **atomic**:

- Header includes a CRC32 checksum and payload size.
- Writes are `fsync`'d and atomically renamed.
- Directory is `fsync`'d to ensure durability of the rename.

## Observability

Metrics JSON includes:

- Query/upsert/delete counters
- Latency histogram
- VectorStore live/free/allocated counts
- Index size/deletion stats

Logging is structured (JSON) for snapshot and lifecycle events.

