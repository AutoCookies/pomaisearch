# Known Risks

This document lists risks that were addressed and those intentionally accepted.

## Eliminated Risks

- **Unbounded vector memory growth** due to append-only storage.
- **Stale vectors on upsert** (overwrites now reuse slots).
- **TTL as filter only** (expired vectors are reclaimed on subsequent writes).
- **Silent snapshot corruption** (CRC32 + size validation enforced).

## Accepted Risks (with reasoning)

- **No WAL / incremental durability**: snapshots are the only durability mechanism. This is acceptable for embedded deployments where periodic snapshots are sufficient.
- **TTL reclamation requires writes**: without new writes, expired vectors are filtered but not reclaimed. A background sweeper is intentionally omitted to keep the runtime simple and deterministic.
- **Approximate index recall**: IVF and HNSW recall depends on tuning parameters. Users must choose appropriate settings for their workloads.
- **Blocking writes during index updates**: index updates acquire writer locks and can briefly block concurrent searches. This is minimized by avoiding shard-wide locks during scoring.

