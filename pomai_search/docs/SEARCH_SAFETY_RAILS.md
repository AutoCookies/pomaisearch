# Search Safety Rails

Pomai Search applies bounded execution controls:
- Max candidate gather per shard (`2000`).
- Positive `topk` required.
- Optional shard capacity (`max_points_per_shard`) to cap memory growth.

On bound pressure, behavior degrades deterministically by clamping candidate volume and returning stable top-k ordering.
