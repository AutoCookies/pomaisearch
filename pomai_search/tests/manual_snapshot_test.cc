#include "pomai_search/search_engine.h"
#include "core/serialize/snapshot.h"
#include <iostream>
#include <cstdio>

using namespace pomai_search;

int main() {
    // Create a simple engine
    SearchEngineConfig cfg;
    cfg.dim = 3;
    cfg.num_shards = 1;
    cfg.global_seed = 42;
    cfg.index_type = SearchEngineConfig::IndexType::Flat;
    
    auto engine_or = SearchEngine::Open(cfg);
    if (!engine_or.ok()) {
        std::cerr << "Failed to open engine: " << engine_or.status().message() << std::endl;
        return 1;
    }
    auto engine = std::move(engine_or.value());
    
    // Add some vectors
    float vec1[] = {1.0f, 0.0f, 0.0f};
    float vec2[] = {0.0f, 1.0f, 0.0f};
    float vec3[] = {0.0f, 0.0f, 1.0f};
    
    engine->Upsert("doc1", VectorView{vec1, 3}, {{"type", "test"}}, std::nullopt, "first document");
    engine->Upsert("doc2", VectorView{vec2, 3}, {{"type", "test"}}, std::nullopt, "second document");
    engine->Upsert("doc3", VectorView{vec3, 3}, {{"type", "test"}}, std::nullopt, "third document");
    
    // Search before snapshot
    float query[] = {1.0f, 0.0f, 0.0f};
    auto before_results = engine->Search(VectorView{query, 3});
    if (!before_results.ok()) {
        std::cerr << "Search failed: " << before_results.status().message() << std::endl;
        return 1;
    }
    
    std::cout << "Before snapshot - Top result: " << before_results.value()[0].key 
              << " (score: " << before_results.value()[0].score << ")" << std::endl;
    
    // Save snapshot
    const char* snapshot_path = "/tmp/test_snapshot.pomai";
    Status save_status = SnapshotWriter::Write(*engine, snapshot_path);
    if (!save_status.ok()) {
        std::cerr << "Failed to save snapshot: " << save_status.message() << std::endl;
        return 1;
    }
    std::cout << "✓ Snapshot saved successfully" << std::endl;
    
    // Close original engine
    engine.reset();
    
    // Load snapshot
    auto loaded_or = SnapshotReader::Read(snapshot_path);
    if (!loaded_or.ok()) {
        std::cerr << "Failed to load snapshot: " << loaded_or.status().message() << std::endl;
        return 1;
    }
    auto loaded_engine = std::move(loaded_or.value());
    std::cout << "✓ Snapshot loaded successfully" << std::endl;
    
    // Search after loading
    auto after_results = loaded_engine->Search(VectorView{query, 3});
    if (!after_results.ok()) {
        std::cerr << "Search after load failed: " << after_results.status().message() << std::endl;
        return 1;
    }
    
    std::cout << "After snapshot - Top result: " << after_results.value()[0].key 
              << " (score: " << after_results.value()[0].score << ")" << std::endl;
    
    // Verify results match
    if (before_results.value()[0].key == after_results.value()[0].key &&
        before_results.value()[0].score == after_results.value()[0].score) {
        std::cout << "\n✅ SUCCESS: Snapshot round-trip works correctly!" << std::endl;
        std::remove(snapshot_path);
        return 0;
    } else {
        std::cerr << "\n❌ FAILURE: Results don't match after snapshot!" << std::endl;
        return 1;
    }
}
