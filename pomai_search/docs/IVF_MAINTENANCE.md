# IVF Maintenance

## Invariants

- Posting lists store `(doc_id, generation)` pairs.
- A posting is live iff doc exists, `doc.deleted == false`, and `doc.gen == posting.gen`.
- Upserts increment generation and append exactly one new posting.
- Deletes/TTL invalidation increment generation, making old postings stale.

## Compaction

Compaction is incremental and list-local.

A list is compacted when any condition is true:
- `stale_ratio > stale_ratio_threshold`
- `list_size > max_list_size`
- `list_bytes > max_list_bytes`

Compaction rebuilds the list by keeping only live postings and (for SQ8) matching codes.

## Deterministic Degradation

- Candidate collection is bounded by `max_candidates`.
- Rerank fanout is bounded by `rerank_factor`.
- IVF probing remains bounded by `ivf_nprobe`.
- On overload, the engine returns fewer candidates; it does not allocate unbounded structures.
