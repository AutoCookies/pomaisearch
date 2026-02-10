# Production Readiness

This document defines release gates and operational expectations for Pomai Search.

## Release Gates

### Correctness

- Cross-index parity (Flat vs IVF) for cosine when IVF is configured with `nlist=1`.
- Snapshot corruption detection tests must pass.
- TTL reclamation tests must pass.

### Performance

- Flat index baseline documented.
- IVF/HNSW speedup measured relative to Flat (same dataset).
- No >2× latency regression under concurrent ingest workloads.

### Stability

- Long-run upsert loop stabilizes memory (slot reuse).
- Concurrency test: simultaneous ingest + search for multiple seconds.
- Snapshot load/save must be deterministic and fail on corruption.

## Operational Invariants

- Memory usage stabilizes after reaching the high-water mark of slots.
- Snapshots are atomic, versioned, and validated with CRC32.
- TTL expiry is enforced on query; reclamation occurs during writes.

## How to Validate

- Unit tests: `ctest --output-on-failure`
- Benchmarks: see `bench/` for `bench_flat`, `bench_ivf`, `bench_hnsw`, `bench_sq8`.

