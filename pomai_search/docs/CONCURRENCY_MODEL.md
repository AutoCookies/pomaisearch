# Concurrency Model

- Writes (upsert/delete) use shard-local exclusive locking for metadata updates.
- Searches use shared locking and read guards for vector storage.
- Query fanout across shards is optionally parallelized via thread pool.

Determinism is maintained by explicit tie-break ordering after shard merge.

See tests:
- `tests/test_concurrency.cc`
- `tests/test_determinism.cc`
