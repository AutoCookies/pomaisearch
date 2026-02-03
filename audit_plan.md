# Audit Report & Refactoring Plan

## Current State Analysis

### Directory Structure mismatch
**Current**:
- `pomai_search/src/search_engine.cc` (Monolithic controller)
- `pomai_search/src/index/` (Flat/HNSW)
- `pomai_search/src/simd/` (Kernels)

**Target**:
- `core/vectorstore/`
- `core/kernels/`
- `core/index/`
- `core/query/`
- `core/serialize/`
- `app/`

### Feature Gaps (Phase 1 - HNSW)
- **Neighbors**: Currently `std::vector<uint32_t>` (Good, flattened).
- **Pruning**: Heuristic implemented, but `extend_candidates` (neighbors of neighbors) is MISSING.
- **Hot Path**: `Visited` set optimization (Generation Counters) is MISSING (likely using std::vector or set reset).
- **Benchmarks**: `bench_recall` exists but output format needs standardization.

### DB Drift Check
- **TTL**: Implemented using `system_clock`. This is acceptable as a "filtering" feature, provided it's serialized in snapshots and not via WAL. *Verdict: Safe, provided no WAL is added.*
- **Snapshots**: Currently `snapshot.cc`. Needs moving to `core/serialize`.
- **Sharding**: Currently `SearchEngineImpl::shards`. This logic belongs in `core/index` (CompositeIndex or ShardedIndex) or `core/query` layer, not mixed in a monolithic engine class.

## Design Plan: Modularization

### 1. VectorStore (`core/vectorstore`)
- **Move**: `src/vector_store.cc` -> `core/vectorstore/vector_store.cc`
- **Scope**: Owns `VectorArena`. Append-only + Deletion bitmap (optional).

### 2. Kernels (`core/kernels`)
- **Move**: `src/simd/*` -> `core/kernels/`
- **Refine**: Ensure dispatch table is clean (Function Pointers).

### 3. Index (`core/index`)
- **Move**: `src/index/*` -> `core/index/`
- **Refactor**:
    - `Index` interface should be pure.
    - `HNSWIndex`: Add `extend_candidates` and `visited_stamp`.
    - `FlatIndex`: Keep as baseline.
    - **Future**: `IVFIndex`.

### 4. Query (`core/query`)
- **Move**: Logic from `search_engine.cc` (Search, Filter, TopK) -> `core/query/`
- **Components**: `FilterEvaluator`, `TopKHeap`.

### 5. Serialize (`core/serialize`)
- **Move**: `src/snapshot.cc` -> `core/serialize/snapshot.cc`
- **Format**: Ensure Magic/Version/Checksum.

### 6. App (`app/`)
- **Move**: `src/cli/`, `src/server/` -> `app/`

## Execution Steps

1.  **Scaffold**: Create new directory structure.
2.  **Move & Fix**: Move files and fix include paths (CMake updates).
3.  **Phase 1 Execution (HNSW)**:
    - Implement `extend_candidates`.
    - Implement `visited_stamp`.
    - Tune defaults.
    - Add Guardrail Tests.
