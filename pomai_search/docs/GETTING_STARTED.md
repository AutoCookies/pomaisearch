# Getting Started with Pomai Search

Complete tutorial for new users.

## Prerequisites

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.15+
- Basic understanding of vector embeddings

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/pomaisearch.git
cd pomaisearch/pomai_search

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure
# Run tests
./pomai_search_tests
```

### Using CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    pomaisearch
    GIT_REPOSITORY https://github.com/yourusername/pomaisearch.git
    GIT_TAG main
)
FetchContent_MakeAvailable(pomaisearch)

target_link_libraries(your_app PRIVATE pomai_search_core)
```

---

## Your First Search Engine

### Step 1: Include Headers

```cpp
#include "pomai_search/search_engine.h"
#include <iostream>
#include <vector>

using namespace pomai_search;
```

### Step 2: Configure Engine

```cpp
SearchEngineConfig cfg;
cfg.dim = 128;                                    // Vector dimensions
cfg.index_type = SearchEngineConfig::IndexType::Hnsw;  // Index type
cfg.similarity = SearchEngineConfig::Similarity::Cosine;
cfg.num_shards = 4;                               // Parallelism
```

### Step 3: Open Engine

```cpp
auto engine_or = SearchEngine::Open(cfg);
if (!engine_or.ok()) {
    std::cerr << "Failed to open: " << engine_or.status().message() << "\n";
    return 1;
}
auto engine = std::move(engine_or.value());
```

### Step 4: Insert Vectors

```cpp
// Create sample vectors
std::vector<float> vec1(128);
vec1[0] = 1.0f;  // Simple example

std::vector<float> vec2(128);
vec2[1] = 1.0f;

// Insert with metadata
engine->Upsert("doc1", VectorView{vec1.data(), 128}, 
               {{"category", "tech"}});
engine->Upsert("doc2", VectorView{vec2.data(), 128},
               {{"category", "science"}});
```

### Step 5: Search

```cpp
// Create query
std::vector<float> query(128);
query[0] = 1.0f;

// Search
auto results = engine->Search(VectorView{query.data(), 128});
if (results.ok()) {
    for (const auto& item : results.value()) {
        std::cout << item.key << ": " << item.score << "\n";
    }
}
```

### Complete Example

```cpp
#include "pomai_search/search_engine.h"
#include <iostream>
#include <vector>

using namespace pomai_search;

int main() {
    // Configure
    SearchEngineConfig cfg;
    cfg.dim = 128;
    cfg.index_type = SearchEngineConfig::IndexType::Hnsw;
    
    // Open
    auto engine = SearchEngine::Open(cfg).value();
    
    // Insert
    std::vector<float> vec(128, 0.0f);
    vec[0] = 1.0f;
    engine->Upsert("doc1", VectorView{vec.data(), 128});
    
    // Search
    auto results = engine->Search(VectorView{vec.data(), 128});
    std::cout << "Found: " << results.value()[0].key << "\n";
    
    return 0;
}
```

---

## Common Use Cases

### 1. Semantic Search

```cpp
// Assuming you have a text embedding model
std::vector<float> embed_text(const std::string& text);

// Index documents
engine->Upsert("doc1", VectorView{embed_text("AI tutorial").data(), 128});
engine->Upsert("doc2", VectorView{embed_text("Cooking recipe").data(), 128});

// Search
auto query_vec = embed_text("machine learning");
auto results = engine->Search(VectorView{query_vec.data(), 128});
```

### 2. Image Search

```cpp
// Assuming you have an image embedding model
std::vector<float> embed_image(const std::string& image_path);

// Index images
engine->Upsert("img1.jpg", VectorView{embed_image("cat.jpg").data(), 512});
engine->Upsert("img2.jpg", VectorView{embed_image("dog.jpg").data(), 512});

// Search by image
auto query_vec = embed_image("kitten.jpg");
auto results = engine->Search(VectorView{query_vec.data(), 512});
```

### 3. Recommendation System

```cpp
// User embeddings
std::vector<float> user_embedding(128);
// ... compute from user history

// Item embeddings
engine->Upsert("item1", VectorView{item1_embedding.data(), 128},
               {{"price", "19.99"}, {"category", "electronics"}});

// Recommend
auto results = engine->Search(VectorView{user_embedding.data(), 128});
// results[0] = best recommendation
```

### 4. Duplicate Detection

```cpp
// Insert documents
for (const auto& doc : documents) {
    auto vec = embed_text(doc.text);
    engine->Upsert(doc.id, VectorView{vec.data(), 128});
}

// Find duplicates (score > 0.95)
auto query_vec = embed_text(new_document);
auto results = engine->Search(VectorView{query_vec.data(), 128});
if (!results.value().empty() && results.value()[0].score > 0.95) {
    std::cout << "Duplicate found: " << results.value()[0].key << "\n";
}
```

---

## Advanced Features

### Metadata Filtering

```cpp
SearchEngine::QueryOptions opts;
opts.topk = 10;
opts.filter.tag = "tech";  // Only return documents with tag="tech"

auto results = engine->Search(query, opts);
```

### TTL (Time-To-Live)

```cpp
using namespace std::chrono;

// Document expires in 1 hour
engine->Upsert("temp_doc", vec, meta,
               milliseconds(3600000),  // TTL
               std::nullopt);

// After 1 hour, document is automatically excluded from search
```

### Hybrid Search

```cpp
SearchEngine::HybridQuery query;
query.vector_query = vec;
query.text_query = "machine learning tutorial";
query.alpha = 0.7;  // 70% vector, 30% text

auto results = engine->SearchHybrid(query);
```

### Persistence

```cpp
// Save to disk
SnapshotWriter::Write(*engine, "my_index.pomai");

// Load from disk (instant startup)
auto loaded_engine = SnapshotReader::Read("my_index.pomai").value();
```

---

## Performance Tips

### 1. Choose the Right Index

```cpp
// Small dataset (<10K): Use Flat
cfg.index_type = SearchEngineConfig::IndexType::Flat;

// Medium dataset (10K-1M): Use HNSW
cfg.index_type = SearchEngineConfig::IndexType::Hnsw;

// Large dataset (>1M): Use IVF-SQ8
cfg.index_type = SearchEngineConfig::IndexType::IvfSq8;
```

### 2. Tune Parameters

```cpp
// For high recall (>99%)
cfg.hnsw_m = 32;
cfg.hnsw_ef_construction = 400;
cfg.hnsw_ef_search = 100;

// For fast queries
cfg.hnsw_m = 16;
cfg.hnsw_ef_search = 30;
```

### 3. Use Multiple Shards

```cpp
cfg.num_shards = std::thread::hardware_concurrency();
```

### 4. Enable SIMD

```cpp
cfg.enable_avx2 = true;  // 4-8x speedup
```

---

## Error Handling

### Checking Status

```cpp
Status s = engine->Upsert("doc1", vec);
if (!s.ok()) {
    std::cerr << "Error: " << s.message() << "\n";
    if (s.code() == StatusCode::kResourceExhausted) {
        std::cerr << "Index is full!\n";
    }
}
```

### Checking StatusOr

```cpp
auto results = engine->Search(query);
if (!results.ok()) {
    std::cerr << "Search failed: " << results.status().message() << "\n";
    return;
}

// Safe to access value
for (const auto& item : results.value()) {
    std::cout << item.key << "\n";
}
```

---

## Best Practices

### 1. Reuse Engine Instances

```cpp
// ❌ Bad: Creating new engine for each query
for (const auto& query : queries) {
    auto engine = SearchEngine::Open(cfg).value();
    engine->Search(query);
}

// ✅ Good: Reuse engine
auto engine = SearchEngine::Open(cfg).value();
for (const auto& query : queries) {
    engine->Search(query);
}
```

### 2. Batch Operations

```cpp
// ✅ Good: Batch upserts
for (const auto& doc : documents) {
    engine->Upsert(doc.id, doc.vec, doc.meta);
}
// Internally optimized for batching
```

### 3. Use Snapshots

```cpp
// On startup
if (std::filesystem::exists("index.pomai")) {
    engine = SnapshotReader::Read("index.pomai").value();
} else {
    engine = SearchEngine::Open(cfg).value();
    // ... build index
    SnapshotWriter::Write(*engine, "index.pomai");
}
```

### 4. Monitor Performance

```cpp
// Periodic monitoring
auto stats = engine->GetStats();
std::cout << "Points: " << stats.num_points << "\n";
std::cout << "Deleted: " << stats.num_deleted << "\n";

auto metrics = engine->MetricsJson();
// Log to monitoring system
```

---

## Troubleshooting

### "Dimension mismatch" Error

```cpp
// ❌ Wrong dimension
cfg.dim = 128;
auto engine = SearchEngine::Open(cfg).value();
float vec[256] = {...};  // Wrong!
engine->Upsert("doc", VectorView{vec, 256});  // Error

// ✅ Correct
float vec[128] = {...};
engine->Upsert("doc", VectorView{vec, 128});  // OK
```

### Low Recall

```cpp
// Increase search quality
cfg.hnsw_ef_search = 100;  // Higher = better recall
cfg.ivf_nprobe = 20;       // For IVF indexes
```

### High Memory Usage

```cpp
// Use quantized index
cfg.index_type = SearchEngineConfig::IndexType::IvfSq8;  // 4x less memory
```

### Slow Queries

```cpp
// Reduce search quality
cfg.hnsw_ef_search = 30;  // Lower = faster
cfg.ivf_nprobe = 5;       // For IVF indexes
```

---

## Next Steps

- Read [API Reference](API.md) for complete API documentation
- Read [Architecture Guide](ARCHITECTURE.md) to understand internals
- Read [Performance Tuning](PERFORMANCE.md) for optimization tips
- Check [examples/](../examples/) for more code samples

---

## Getting Help

- 📖 Documentation: [docs/](.)
- 💬 Discord: [Join community](https://discord.gg/pomaisearch)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/pomaisearch/issues)
- 📧 Email: support@pomaisearch.com

---

## Quick Reference

### Essential Methods

```cpp
// Open engine
auto engine = SearchEngine::Open(cfg).value();

// Insert
engine->Upsert(key, vec, meta, ttl, text);

// Search
auto results = engine->Search(query, options);

// Delete
engine->Delete(key);

// Save/Load
SnapshotWriter::Write(*engine, path);
auto engine = SnapshotReader::Read(path).value();

// Stats
auto stats = engine->GetStats();
```

### Configuration Presets

```cpp
// High recall
cfg.index_type = IndexType::Hnsw;
cfg.hnsw_m = 32;
cfg.hnsw_ef_search = 100;

// Fast queries
cfg.index_type = IndexType::Hnsw;
cfg.hnsw_m = 16;
cfg.hnsw_ef_search = 30;

// Low memory
cfg.index_type = IndexType::IvfSq8;
cfg.ivf_nlist = 1000;
cfg.ivf_nprobe = 10;

// Exact search
cfg.index_type = IndexType::Flat;
```
