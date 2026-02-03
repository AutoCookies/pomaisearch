# Pomai Search Engine

**Tagline**: CPU-first, single-node, minimal deps, FAISS-like vector search engine.

## Scope & Philosophy

Pomai Search Engine is designed to be a lightweight, high-performance embedded library for approximate nearest neighbor (ANN) search. It focuses on modular indexing, tuned CPU kernels (AVX2), and correctness.

It is **NOT** a database. It strictly adheres to the "Exactly One Copy" principle for vector data and provides snapshotting for fast loading, not for durability guarantees.

## Non-Goals (Anti-PomaiDB Policy)

To insure lightweight design and focus, this project strictly **BANS** the following features:

*   **No WAL (Write-Ahead-Log)**: Durability is the responsibility of the caller/upstream system.
*   **No Crash Recovery**: We load valid snapshots. We do not replay logs.
*   **No DB Lifecycle**: Concepts like memtables, freezing, flushing, or compaction strategies designed for LSM-trees are forbidden. VectorStore compaction is strictly for memory reclamation, not data visibility.
*   **No Transactions (ACID)**: There is no isolation or multi-statement transaction support.
*   **No Distributed Systems**: No replication, raft, consensus, or sharding across network nodes. This is a single-node library.
*   **No Query Planner**: No SQL-like parsing or query optimization engines.

## Architecture

The system is divided into strict modules:

*   `core/vectorstore`: Owns the raw vector data (Source of Truth).
*   `core/kernels`: SIMD-optimized distance functions (AVX2/Scalar).
*   `core/index`: ANN implementations (Flat, HNSW, IVF-Flat).
*   `core/query`: Execution logic (filtering, top-k collection).
*   `core/serialize`: Binary format for save/load.

## Usage

```cpp
#include "core/index/index_factory.h"

// 1. Create Config
IndexConfig config;
config.dim = 128;
config.type = IndexType::HNSW;

// 2. Open Engine
auto index = IndexFactory::Create(config);

// 3. Add Vectors
index->Add(id, vector_data);

// 4. Search
auto results = index->Search(query_vector, k);
```
