# API Reference

Complete API documentation for Pomai Search Engine.

## Table of Contents

- [SearchEngine](#searchengine)
- [Configuration](#configuration)
- [Data Types](#data-types)
- [Snapshot API](#snapshot-api)
- [Error Handling](#error-handling)

---

## SearchEngine

The main interface for vector search operations.

### Opening an Engine

```cpp
static StatusOr<std::unique_ptr<SearchEngine>> Open(const SearchEngineConfig& cfg);
```

**Parameters:**
- `cfg`: Configuration object specifying dimensions, index type, and parameters

**Returns:**
- `StatusOr` containing the engine instance or error status

**Example:**
```cpp
SearchEngineConfig cfg;
cfg.dim = 128;
cfg.index_type = SearchEngineConfig::IndexType::Hnsw;

auto engine_or = SearchEngine::Open(cfg);
if (!engine_or.ok()) {
    std::cerr << "Error: " << engine_or.status().message() << "\n";
    return;
}
auto engine = std::move(engine_or.value());
```

---

### Upsert

```cpp
Status Upsert(std::string_view key, 
              VectorView vec, 
              Metadata meta = {},
              std::optional<std::chrono::milliseconds> ttl = std::nullopt,
              std::optional<std::string> text = std::nullopt);
```

Inserts or updates a vector with associated metadata.

**Parameters:**
- `key`: Unique identifier for the vector (max 256 chars recommended)
- `vec`: Vector data (`VectorView{data, dim}`)
- `meta`: Optional metadata key-value pairs
- `ttl`: Optional time-to-live duration (milliseconds from now)
- `text`: Optional text content for hybrid search

**Returns:**
- `Status::Ok()` on success
- Error status on failure (e.g., dimension mismatch, capacity exceeded)

**Behavior:**
- If `key` exists, the vector and metadata are **replaced**
- Metadata is **not** merged with existing metadata
- TTL is calculated from insertion time

**Example:**
```cpp
float vec[128] = {...};
Status s = engine->Upsert(
    "doc123",
    VectorView{vec, 128},
    {{"category", "tech"}, {"author", "alice"}},
    std::chrono::milliseconds(3600000),  // 1 hour
    "Introduction to vector databases"
);
if (!s.ok()) {
    std::cerr << "Upsert failed: " << s.message() << "\n";
}
```

---

### Delete

```cpp
Status Delete(std::string_view key);
```

Marks a vector as deleted. The vector is not immediately removed but excluded from search results.

**Parameters:**
- `key`: Key of the vector to delete

**Returns:**
- `Status::Ok()` on success
- `StatusCode::kNotFound` if key doesn't exist

**Note:** Deleted vectors are removed during compaction (not yet implemented).

**Example:**
```cpp
Status s = engine->Delete("doc123");
```

---

### Exists

```cpp
StatusOr<bool> Exists(std::string_view key) const;
```

Checks if a key exists and is not expired or deleted.

**Parameters:**
- `key`: Key to check

**Returns:**
- `StatusOr<bool>` containing `true` if exists, `false` otherwise

**Example:**
```cpp
auto exists = engine->Exists("doc123");
if (exists.ok() && exists.value()) {
    std::cout << "Document exists\n";
}
```

---

### GetVector

```cpp
StatusOr<std::vector<float>> GetVector(std::string_view key) const;
```

Retrieves the vector associated with a key.

**Parameters:**
- `key`: Key of the vector to retrieve

**Returns:**
- `StatusOr<std::vector<float>>` containing the vector
- `StatusCode::kNotFound` if key doesn't exist or is expired

**Example:**
```cpp
auto vec_or = engine->GetVector("doc123");
if (vec_or.ok()) {
    const auto& vec = vec_or.value();
    std::cout << "Vector dim: " << vec.size() << "\n";
}
```

---

### Search

```cpp
StatusOr<std::vector<ResultItem>> Search(VectorView q, QueryOptions opt);
StatusOr<std::vector<ResultItem>> Search(VectorView q);  // Uses default options
```

Searches for similar vectors.

**Parameters:**
- `q`: Query vector
- `opt`: Query options (topk, filters)

**Returns:**
- `StatusOr<std::vector<ResultItem>>` containing ranked results (highest score first)

**QueryOptions:**
```cpp
struct QueryOptions {
    int topk = 0;        // Number of results (0 = use default)
    Filter filter;       // Metadata and expiry filters
};
```

**ResultItem:**
```cpp
struct ResultItem {
    std::string key;
    float score;
    Metadata meta;
    uint32_t internal_id;
};
```

**Example:**
```cpp
float query[128] = {...};
QueryOptions opts;
opts.topk = 10;
opts.filter.tag = "tech";

auto results = engine->Search(VectorView{query, 128}, opts);
if (results.ok()) {
    for (const auto& item : results.value()) {
        std::cout << item.key << ": " << item.score << "\n";
    }
}
```

---

### SearchByKey

```cpp
StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key, QueryOptions opt);
StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key);
```

Searches using an existing vector as the query.

**Parameters:**
- `key`: Key of the vector to use as query
- `opt`: Query options

**Returns:**
- Same as `Search()`

**Example:**
```cpp
auto results = engine->SearchByKey("doc123");
```

---

### SearchHybrid

```cpp
StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query, QueryOptions opt);
StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query);
```

Performs hybrid vector + keyword search.

**HybridQuery:**
```cpp
struct HybridQuery {
    std::optional<std::vector<float>> vector_query;
    std::optional<std::string> text_query;
    float alpha = 0.5f;  // Weight: 0=text only, 1=vector only
};
```

**Scoring:**
```
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```

**Example:**
```cpp
HybridQuery query;
query.vector_query = std::vector<float>{...};
query.text_query = "machine learning tutorial";
query.alpha = 0.7;  // 70% vector, 30% text

auto results = engine->SearchHybrid(query);
```

---

### SearchWithExplain

```cpp
StatusOr<SearchResponse> SearchWithExplain(VectorView q, 
                                           QueryOptions opt, 
                                           QueryPolicy policy);
```

Searches with detailed query explanation for debugging.

**SearchResponse:**
```cpp
struct SearchResponse {
    std::vector<ResultItem> results;
    QueryExplain explain;
};

struct QueryExplain {
    std::string snapshot_id;
    double query_time_ms;
    // Additional debug info
};
```

**Example:**
```cpp
QueryPolicy policy;
auto response = engine->SearchWithExplain(query, opts, policy);
if (response.ok()) {
    std::cout << "Query time: " << response.value().explain.query_time_ms << "ms\n";
}
```

---

### GetStats

```cpp
Stats GetStats() const;
```

Returns current engine statistics.

**Stats:**
```cpp
struct Stats {
    uint64_t num_points;    // Total indexed vectors
    uint64_t num_deleted;   // Deleted vectors
};
```

**Example:**
```cpp
auto stats = engine->GetStats();
std::cout << "Points: " << stats.num_points << "\n";
std::cout << "Deleted: " << stats.num_deleted << "\n";
```

---

### MetricsJson

```cpp
std::string MetricsJson() const;
```

Returns metrics in JSON format for monitoring.

**Returns:**
- JSON string with query latencies, throughput, etc.

**Example:**
```cpp
std::cout << engine->MetricsJson() << "\n";
```

---

### Close

```cpp
Status Close();
```

Closes the engine and releases resources.

**Note:** The engine cannot be used after calling `Close()`. The destructor automatically calls `Close()`.

**Example:**
```cpp
engine->Close();
```

---

## Configuration

### SearchEngineConfig

Complete configuration reference:

```cpp
struct SearchEngineConfig {
    // Required
    int dim = 0;                    // Vector dimensions (must be > 0)
    
    // Index type
    enum class IndexType { 
        Flat,      // Exact search, O(N)
        Hnsw,      // Graph-based, O(log N)
        IvfFlat,   // Inverted file, O(N/nlist)
        IvfSq8     // Quantized IVF, O(N/nlist)
    } index_type = IndexType::Flat;
    
    // Similarity metric
    enum class Similarity { 
        Dot,       // Dot product (higher = more similar)
        Cosine     // Cosine similarity (normalized dot product)
    } similarity = Similarity::Dot;
    
    // Performance
    int num_shards = 1;             // Parallelism (1-16 recommended)
    int topk_default = 10;          // Default results
    size_t max_points_per_shard = 0; // Capacity limit (0 = unlimited)
    bool enable_avx2 = true;        // SIMD acceleration
    int query_threads = 0;          // Query pool size (0 = auto)
    int ingest_threads = 0;         // Ingest pool size (0 = auto)
    
    // HNSW parameters
    int hnsw_m = 16;                // Graph connectivity (8-64)
    int hnsw_ef_construction = 200; // Build quality (100-500)
    int hnsw_ef_search = 50;        // Search quality (10-500)
    uint32_t hnsw_seed = 42;        // Random seed
    
    // IVF parameters
    int ivf_nlist = 100;            // Number of clusters (10-10000)
    int ivf_nprobe = 10;            // Clusters to search (1-nlist)
    
    // System
    uint64_t global_seed = 0;       // Global random seed
    int contract_version = 3;       // Serialization version
    size_t memory_alignment = 32;   // Memory alignment (bytes)
};
```

### Tuning Guidelines

**For high recall (>99%):**
```cpp
cfg.index_type = IndexType::Hnsw;
cfg.hnsw_m = 32;
cfg.hnsw_ef_construction = 400;
cfg.hnsw_ef_search = 100;
```

**For low memory:**
```cpp
cfg.index_type = IndexType::IvfSq8;
cfg.ivf_nlist = 1000;
cfg.ivf_nprobe = 20;
```

**For exact search:**
```cpp
cfg.index_type = IndexType::Flat;
```

---

## Data Types

### VectorView

Non-owning view of a vector.

```cpp
struct VectorView {
    const float* data;
    int dim;
};
```

**Example:**
```cpp
std::vector<float> vec = {...};
VectorView view{vec.data(), static_cast<int>(vec.size())};
```

---

### Metadata

Key-value string map.

```cpp
using Metadata = std::unordered_map<std::string, std::string>;
```

**Example:**
```cpp
Metadata meta = {
    {"category", "tech"},
    {"author", "alice"},
    {"date", "2026-02-03"}
};
```

---

### Filter

Search filters.

```cpp
struct Filter {
    std::optional<std::string> tag;
    std::optional<std::string> source;
    std::optional<std::string> lang;
};
```

**Note:** Filters use exact string matching.

---

## Snapshot API

### SnapshotWriter

```cpp
class SnapshotWriter {
public:
    static Status Write(const SearchEngine& engine, std::string_view path);
};
```

Saves engine state to disk in native binary format.

**Parameters:**
- `engine`: Engine to save
- `path`: File path (`.pomai` extension recommended)

**Returns:**
- `Status::Ok()` on success

**Format:** Binary format with magic bytes, version, and compressed data.

**Example:**
```cpp
Status s = SnapshotWriter::Write(*engine, "index.pomai");
if (!s.ok()) {
    std::cerr << "Save failed: " << s.message() << "\n";
}
```

---

### SnapshotReader

```cpp
class SnapshotReader {
public:
    static StatusOr<std::unique_ptr<SearchEngine>> Read(std::string_view path);
};
```

Loads engine from snapshot.

**Parameters:**
- `path`: Snapshot file path

**Returns:**
- `StatusOr` containing loaded engine or error

**Load Time:** O(N) - proportional to data size, typically <1s for 1M vectors.

**Example:**
```cpp
auto engine_or = SnapshotReader::Read("index.pomai");
if (!engine_or.ok()) {
    std::cerr << "Load failed: " << engine_or.status().message() << "\n";
    return;
}
auto engine = std::move(engine_or.value());
```

---

## Error Handling

### Status

```cpp
class Status {
public:
    bool ok() const;
    StatusCode code() const;
    std::string message() const;
    
    static Status Ok();
};
```

**StatusCode enum:**
- `kOk` - Success
- `kInvalidArgument` - Invalid parameters
- `kNotFound` - Key not found
- `kResourceExhausted` - Capacity exceeded
- `kInternal` - Internal error

**Example:**
```cpp
Status s = engine->Delete("key");
if (!s.ok()) {
    if (s.code() == StatusCode::kNotFound) {
        std::cout << "Key not found\n";
    } else {
        std::cerr << "Error: " << s.message() << "\n";
    }
}
```

---

### StatusOr<T>

```cpp
template<typename T>
class StatusOr {
public:
    bool ok() const;
    Status status() const;
    const T& value() const;
    T& value();
};
```

**Example:**
```cpp
auto results = engine->Search(query);
if (results.ok()) {
    for (const auto& item : results.value()) {
        // Process results
    }
} else {
    std::cerr << "Search failed: " << results.status().message() << "\n";
}
```

---

## Thread Safety

- **Concurrent reads**: Safe
- **Concurrent writes**: Safe (internally synchronized)
- **Read during write**: Safe (readers see consistent state)

**Note:** Individual operations are atomic, but sequences of operations are not.

---

## Best Practices

1. **Reuse engines**: Opening an engine is expensive. Reuse instances.
2. **Batch upserts**: Group multiple upserts for better throughput.
3. **Use snapshots**: Save/load for fast restarts.
4. **Monitor metrics**: Use `MetricsJson()` for performance tracking.
5. **Choose right index**: Match index type to your use case.
6. **Normalize vectors**: For cosine similarity, pre-normalize if possible.
7. **Set capacity limits**: Use `max_points_per_shard` to prevent OOM.

---

## Limits

- **Max dimensions**: 65,536
- **Max key length**: 256 characters (recommended)
- **Max vectors**: Limited by available memory
- **Max shards**: 256
- **Max metadata size**: 64KB per document (recommended)
