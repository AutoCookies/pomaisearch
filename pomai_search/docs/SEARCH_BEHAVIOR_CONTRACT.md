# Search Behavior Contract

## Invariants
1. Final ranking is exact over gathered candidates.
2. Approximation is allowed only in candidate gather.
3. Candidate count is hard bounded.
4. Metric semantics are globally consistent:
   - Dot: raw dot product.
   - Cosine: vectors normalized on write and query.
5. Same checkpoint + query + config => deterministic ordering.

## Validation
- Invalid `topk <= 0` is rejected.
- Candidate limit is clamped.
- Debug builds assert sorted output invariants.
