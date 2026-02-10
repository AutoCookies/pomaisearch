# Canonical Search Path

Pomai Search enforces a single vector search flow:

`Query -> Coarse Routing -> Bounded Candidate Gather -> Exact Rerank -> Top-K`

## Enforcement
- Index implementations (Flat/HNSW/IVF) provide **candidate routing only**.
- Final score and ordering are recomputed exactly from stored vectors before returning.
- Candidate gathering is hard-bounded (`<= 2000` per shard in current implementation).
- Sorting uses deterministic tie-breakers: score desc, key asc, id asc.

Any pipeline that returns approximate index scores directly is considered invalid.
