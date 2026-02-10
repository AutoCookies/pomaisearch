# Search Safety Rails

Pomai Search applies bounded execution controls:
- Max candidate gather per shard (`2000`).
- Positive `topk` required.
- Optional shard capacity (`max_points_per_shard`) to cap memory growth.

On bound pressure, behavior degrades deterministically by clamping candidate volume and returning stable top-k ordering.


## Phase 2 Stability Closure
- IVF-Flat and IVF-SQ8 upserts are generation-versioned; stale postings are filtered at gather-time and compacted list-locally.
- Compaction triggers: stale ratio threshold, max list entries, or max list bytes.
- Deterministic degradation: bounded max candidates + bounded rerank factor + bounded IVF probes.
- Search rerank now snapshots candidate metadata under shard lock, scores outside the lock, and validates generation before return.
