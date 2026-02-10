# Checkpoint Model

Pomai Search snapshots are treated as **checkpoints**:
- Purpose: warm start and reproducibility.
- Not a durability boundary.
- Not a correctness recovery protocol.

If a checkpoint is missing/corrupt, Pomai Search must fail fast and require rebuild from the caller's source data.

No WAL and no log replay are provided.
