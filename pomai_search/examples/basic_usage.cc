#include "pomai_search/search_engine.h"
#include "core/serialize/snapshot.h"
#include <iostream>
#include <vector>

using namespace pomai_search;

int main() {
    // 1. Configure and open the search engine
    SearchEngineConfig cfg;
    cfg.dim = 128;                                    // Vector dimensions
    cfg.num_shards = 4;                               // Number of shards for parallelism
    cfg.index_type = SearchEngineConfig::IndexType::Hnsw;  // Use HNSW for high recall
    cfg.similarity = SearchEngineConfig::Similarity::Cosine;
    
    auto engine_or = SearchEngine::Open(cfg);
    if (!engine_or.ok()) {
        std::cerr << "Failed to open engine: " << engine_or.status().message() << std::endl;
        return 1;
    }
    auto engine = std::move(engine_or.value());
    std::cout << "✓ Engine opened successfully\n";

    // 2. Insert vectors with metadata
    std::vector<float> vec1(128, 0.0f);
    vec1[0] = 1.0f;  // Simple example vector
    
    std::vector<float> vec2(128, 0.0f);
    vec2[1] = 1.0f;
    
    std::vector<float> vec3(128, 0.0f);
    vec3[2] = 1.0f;
    
    // Upsert with metadata
    engine->Upsert("doc1", VectorView{vec1.data(), 128}, 
                   {{"category", "tech"}, {"source", "blog"}},
                   std::nullopt,  // No TTL
                   "Introduction to vector search");
    
    engine->Upsert("doc2", VectorView{vec2.data(), 128},
                   {{"category", "science"}, {"source", "paper"}},
                   std::nullopt,
                   "Machine learning fundamentals");
    
    engine->Upsert("doc3", VectorView{vec3.data(), 128},
                   {{"category", "tech"}, {"source", "docs"}},
                   std::nullopt,
                   "API documentation guide");
    
    std::cout << "✓ Inserted 3 documents\n";

    // 3. Vector search
    std::vector<float> query(128, 0.0f);
    query[0] = 1.0f;  // Query similar to vec1
    
    SearchEngine::QueryOptions opts;
    opts.topk = 2;
    
    auto results = engine->Search(VectorView{query.data(), 128}, opts);
    if (results.ok()) {
        std::cout << "\nVector search results:\n";
        for (const auto& item : results.value()) {
            std::cout << "  - " << item.key << " (score: " << item.score << ")\n";
        }
    }

    // 4. Hybrid search (vector + text)
    SearchEngine::HybridQuery hybrid;
    hybrid.vector_query = query;
    hybrid.text_query = "vector search";
    hybrid.alpha = 0.7f;  // 70% vector, 30% text
    
    auto hybrid_results = engine->SearchHybrid(hybrid, opts);
    if (hybrid_results.ok()) {
        std::cout << "\nHybrid search results:\n";
        for (const auto& item : hybrid_results.value()) {
            std::cout << "  - " << item.key << " (score: " << item.score << ")\n";
        }
    }

    // 5. Get statistics
    auto stats = engine->GetStats();
    std::cout << "\nEngine stats:\n";
    std::cout << "  Total points: " << stats.num_points << "\n";
    std::cout << "  Deleted: " << stats.num_deleted << "\n";

    // 6. Save snapshot for fast restart
    Status save_status = SnapshotWriter::Write(*engine, "my_index.pomai");
    if (save_status.ok()) {
        std::cout << "\n✓ Snapshot saved to my_index.pomai\n";
    }

    // 7. Load from snapshot (in a new session)
    auto loaded_engine_or = SnapshotReader::Read("my_index.pomai");
    if (loaded_engine_or.ok()) {
        std::cout << "✓ Snapshot loaded successfully\n";
        auto loaded_engine = std::move(loaded_engine_or.value());
        
        // Verify data persisted
        auto verify_results = loaded_engine->Search(VectorView{query.data(), 128});
        if (verify_results.ok() && !verify_results.value().empty()) {
            std::cout << "✓ Data verified after load: " << verify_results.value()[0].key << "\n";
        }
    }

    // 8. Clean up
    engine->Close();
    std::cout << "\n✓ Engine closed\n";
    
    return 0;
}
