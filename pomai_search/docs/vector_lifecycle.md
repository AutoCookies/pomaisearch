# Vector Lifecycle

This document defines the vector lifecycle model and memory behavior.

## Goals

- Deterministic reclamation of replaced and expired vectors.
- No use-after-free for concurrent readers.
- Stable memory usage under repeated upserts and TTL expiry.

## Storage Model

VectorStore uses **block allocation** with fixed-size blocks of vectors. Each vector is stored in a **slot**.

- **Live slots**: active vectors.
- **Free list**: released slots available for reuse.

## Lifecycle Events

### Upsert

- **Existing key**: vector is overwritten **in-place** in the same slot.
- **New key**: vector is stored in a free slot if available, otherwise a new slot is allocated.

### Delete

- Document is marked deleted.
- Index entry is removed.
- Slot is released back to the free list.

### TTL Expiry

- Expired documents are filtered out at search time.
- Reclamation happens during subsequent writes via incremental sweeps:
  - A bounded number of docs are checked per write.
  - Expired docs are marked deleted and their slots released.

## Memory Behavior

Memory usage grows to the high-water mark of slots allocated, then stabilizes as slots are reused.

## Concurrency Safety

- Index operations hold shared locks for reads and exclusive locks for writes.
- VectorStore reads use shared locks, preventing concurrent writes during scoring.
- Slot release happens only after index deletion to avoid use-after-free.

