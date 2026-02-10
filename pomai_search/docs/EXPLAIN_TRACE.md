# Explain Trace

`SearchWithExplain` returns a deterministic, serializable explain artifact.

Trace includes:
- Canonical stage metadata (`canonical_search`, index family, candidate bound, elapsed time).
- Per-result confirmation that final score comes from exact rerank (`fusion_method = exact_rerank`).

Explain trace is an API output, not a log side-channel.
