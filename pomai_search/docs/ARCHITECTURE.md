# Architecture Guide

Deep dive into Pomai Search Engine internals.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Index Implementations](#index-implementations)
- [Serialization](#serialization)
- [Threading Model](#threading-model)
- [Memory Management](#memory-management)

---

## Overview

Pomai Search is designed as an **embeddable** vector search engine with the following design principles:

1. **Zero external dependencies** (except standard library)
2. **Predictable performance** (no GC pauses, bounded latency)
3. **Production-ready** (comprehensive error handling, monitoring)
4. **Extensible** (pluggable index types, similarity metrics)

---

## System Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────┐
│                    SearchEngine                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │                SearchEngine::Impl                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │ │
│  │  │  Shard 1 │  │  Shard 2 │  │  Shard N │        │ │
│  │  └──────────┘  └──────────┘  └──────────┘        │ │
│  │  ┌──────────────────────────────────────┐         │ │
│  │  │         Thread Pools                 │         │ │
│  │  │  - Query Pool (parallel search)      │         │ │
│  │  │  - Ingest Pool (parallel upsert)     │         │ │
│  │  └──────────────────────────────────────┘         │ │
│  │  ┌──────────────────────────────────────┐         │ │
│  │  │         Metrics                      │         │ │
│  │  └──────────────────────────────────────┘         │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Shard Architecture

Each shard is an independent search unit:

```
┌─────────────────────────────────────────────┐
│                  Shard                      │
│  ┌────────────────────────────────────────┐ │
│  │         VectorStore                    │ │
│  │  - Contiguous float buffer             │ │
│  │  - Aligned for SIMD                    │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │         Index (one of:)                │ │
│  │  - FlatIndex                           │ │
│  │  - HnswIndex                           │ │
│  │  - IvfFlatIndex                        │ │
│  │  - IvfSq8Index                         │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │         KeywordIndex (BM25)            │ │
│  │  - Inverted index                      │ │
│  │  - TF-IDF scoring                      │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │         Document Metadata              │ │
│  │  - Key → ID mapping                    │ │
│  │  - Metadata storage                    │ │
│  │  - TTL tracking                        │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Sharding Strategy

Documents are assigned to shards using **consistent hashing**:

```cpp
shard_id = StableHash64(key) % num_shards
```

**Benefits:**
- Deterministic placement
- Even distribution
- Parallel query execution

---

## Index Implementations

### FlatIndex

**Structure:**
```cpp
class FlatIndex {
    std::vector<uint32_t> ids_;           // Document IDs
    std::vector<Item> items_;             // {offset, norm, deleted}
    std::unordered_map<uint32_t, size_t> id_to_index_;
};
```

**Search Algorithm:**
```
1. For each vector in index:
   a. Compute similarity with query
   b. Maintain top-k heap
2. Return sorted results
```

**Complexity:** O(N × D) where N = vectors, D = dimensions

---

### HnswIndex

**Structure:**
```cpp
class HnswIndex {
    std::vector<int> levels_;                    // Node levels
    std::vector<std::vector<uint32_t>> graph_;   // Adjacency lists (layer 0)
    std::unordered_map<uint32_t, std::vector<std::vector<uint32_t>>> upper_links_;
    uint32_t entry_point_;
    int ep_level_;
};
```

**Graph Construction:**
```
For each new vector v:
1. Determine level l = RandomLevel()
2. Find entry point at top layer
3. For each layer from top to l:
   a. Greedy search to find nearest neighbors
   b. Connect v to M nearest neighbors
   c. Prune neighbors using heuristic
```

**Search Algorithm:**
```
1. Start at entry point, top layer
2. For each layer (top to bottom):
   a. Greedy search to find ef nearest neighbors
   b. Move to next layer with best candidate
3. At layer 0, expand to ef_search candidates
4. Return top-k results
```

**Key Parameters:**
- `M`: Graph connectivity (affects recall and memory)
- `ef_construction`: Build-time search width (affects build quality)
- `ef_search`: Query-time search width (affects recall/speed tradeoff)

**Complexity:** O(log N × M × D) average case

---

### IvfFlatIndex

**Structure:**
```cpp
class IvfFlatIndex {
    KMeans kmeans_;                              // Centroid clustering
    std::vector<std::vector<uint32_t>> lists_;   // Inverted lists
    std::unordered_map<uint32_t, size_t> id_to_offset_;
};
```

**Training:**
```
1. Sample training vectors
2. Run k-means to find nlist centroids
3. Assign each vector to nearest centroid
```

**Search Algorithm:**
```
1. Find nprobe nearest centroids to query
2. For each selected centroid:
   a. Scan its inverted list
   b. Compute exact similarities
3. Merge and return top-k
```

**Complexity:** O(nprobe × N/nlist × D)

---

### IvfSq8Index

**Structure:**
```cpp
class IvfSq8Index {
    KMeans kmeans_;
    ScalarQuantizer quantizer_;                  // Per-dimension quantization
    std::vector<std::vector<uint32_t>> lists_;
    std::vector<std::vector<uint8_t>> codes_;    // Compressed vectors
};
```

**Scalar Quantization:**
```
For each dimension d:
1. Compute min_d, max_d across training set
2. Quantize: code[d] = round((val[d] - min_d) / step_d * 255)
3. Dequantize: val[d] = min_d + code[d] * step_d
```

**Search with Refinement:**
```
1. Find nprobe nearest centroids
2. Approximate search on quantized codes
3. Select top (refine_factor × k) candidates
4. Re-rank using full-precision vectors
5. Return top-k
```

**Memory Savings:** 4x reduction (float32 → uint8)

---

## Serialization

### Snapshot Format V2

Binary format for fast save/load:

```
┌──────────────────────────────────────┐
│ Magic: "POMAI_V2" (8 bytes)          │
├──────────────────────────────────────┤
│ Version: 2 (uint32_t)                │
├──────────────────────────────────────┤
│ Config (binary SearchEngineConfig)   │
│  - dim, index_type, similarity, etc. │
├──────────────────────────────────────┤
│ Num Shards (uint64_t)                │
├──────────────────────────────────────┤
│ For each shard:                      │
│  ┌────────────────────────────────┐  │
│  │ Shard Data                     │  │
│  │  - Documents                   │  │
│  │  - Key→ID map                  │  │
│  │  - VectorStore                 │  │
│  │  - Index state                 │  │
│  │  - KeywordIndex                │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### Index-Specific Serialization

**HnswIndex:**
- Graph structure (adjacency lists)
- Node levels
- Entry point
- Visited stamps

**IvfSq8Index:**
- KMeans centroids
- ScalarQuantizer parameters (min, step per dimension)
- Inverted lists
- Compressed codes

**Load Time:** O(N) - linear in data size, no re-indexing needed

---

## Threading Model

### Query Parallelism

```cpp
// Parallel shard search
std::vector<std::future<Results>> futures;
for (auto& shard : shards) {
    futures.push_back(query_pool->Submit([&]() {
        return shard->Search(query);
    }));
}

// Merge results
for (auto& future : futures) {
    auto results = future.get();
    merged.insert(merged.end(), results.begin(), results.end());
}
std::nth_element(merged.begin(), merged.begin() + k, merged.end(), better);
```

### Synchronization

- **Shard-level locks:** `std::shared_mutex` for concurrent reads
- **Lock-free reads:** Readers don't block each other
- **Write serialization:** Writers acquire exclusive lock

**Lock Hierarchy:**
```
SearchEngine::Impl
  └─ Shard::mutex_ (shared_mutex)
       ├─ VectorStore (no additional lock)
       ├─ Index::mutex_ (shared_mutex)
       └─ KeywordIndex::mutex_ (shared_mutex)
```

---

## Memory Management

### VectorStore

Contiguous allocation for cache efficiency:

```cpp
class VectorStore {
    std::vector<float> data_;  // Aligned allocation
    int dim_;
    size_t count_;
};
```

**Alignment:** 32-byte aligned for AVX2 SIMD

**Growth Strategy:** Exponential (2x) to amortize allocations

### SIMD Optimization

**Dot Product (AVX2):**
```cpp
__m256 sum = _mm256_setzero_ps();
for (int i = 0; i < dim; i += 8) {
    __m256 a = _mm256_load_ps(&vec1[i]);
    __m256 b = _mm256_load_ps(&vec2[i]);
    sum = _mm256_fmadd_ps(a, b, sum);
}
```

**Speedup:** 4-8x vs scalar code

### Memory Footprint

**Per vector overhead:**

| Index Type | Overhead | Notes |
|-----------|----------|-------|
| Flat | 12 bytes | ID + metadata pointer |
| HNSW | ~200 bytes | Graph edges (M=16) |
| IVF-Flat | 8 bytes | List pointer |
| IVF-SQ8 | 2 bytes | Quantized code pointer |

**Total memory:**
```
Total = N × (D × 4 + overhead) + index_structure
```

---

## Performance Characteristics

### Time Complexity

| Operation | Flat | HNSW | IVF-Flat | IVF-SQ8 |
|-----------|------|------|----------|---------|
| Insert | O(1) | O(M × log N) | O(1)* | O(1)* |
| Search | O(N × D) | O(log N × M × D) | O(nprobe × N/nlist × D) | O(nprobe × N/nlist × D) |
| Delete | O(1) | O(1) | O(1) | O(1) |

*After training

### Space Complexity

| Index Type | Space |
|-----------|-------|
| Flat | O(N × D) |
| HNSW | O(N × (D + M × log N)) |
| IVF-Flat | O(N × D + nlist × D) |
| IVF-SQ8 | O(N × D/4 + nlist × D) |

---

## Design Decisions

### Why Sharding?

1. **Parallelism:** Query multiple shards concurrently
2. **Scalability:** Distribute load across cores
3. **Isolation:** Failures contained to single shard

### Why Consistent Hashing?

1. **Deterministic:** Same key always goes to same shard
2. **Balanced:** Even distribution
3. **Simple:** No coordination needed

### Why Native Serialization?

1. **Fast startup:** O(N) load vs O(N log N) rebuild
2. **Exact restore:** Bit-identical state
3. **Portable:** Works across restarts

### Why C++20?

1. **Performance:** Zero-cost abstractions
2. **Control:** Manual memory management
3. **Portability:** Runs anywhere
4. **Modern features:** Concepts, ranges, coroutines (future)

---

## Future Enhancements

### Planned Features

1. **Compaction:** Remove deleted vectors, rebuild index
2. **Dynamic resizing:** Grow/shrink shards
3. **Distributed mode:** Multi-node deployment
4. **GPU acceleration:** CUDA/ROCm support
5. **Product quantization:** Better compression
6. **Graph compression:** Reduce HNSW memory

### Research Directions

1. **Learned indexes:** ML-based routing
2. **Approximate graph:** Probabilistic edges
3. **Streaming updates:** Incremental index updates
4. **Multi-modal:** Images, text, audio in one index

---

## References

1. **HNSW:** Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
2. **IVF:** Jégou et al., "Product quantization for nearest neighbor search" (2011)
3. **Scalar Quantization:** Johnson et al., "Billion-scale similarity search with GPUs" (2017)
4. **BM25:** Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
