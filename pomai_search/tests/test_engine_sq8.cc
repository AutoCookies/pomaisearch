#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <iostream>

namespace pomai_search::test {

POMAI_TEST(EngineSq8Recall) {
    // Verify that SQ8 with refinement has high recall.
    SearchEngineConfig cfg;
    cfg.dim = 32;
    cfg.index_type = SearchEngineConfig::IndexType::IvfSq8;
    cfg.ivf_nlist = 10;
    cfg.ivf_nprobe = 3; 
    
    // Create random data (normalized for cosine/dot consistency)
    std::mt19937 rng(42);
    // uniform is fine, but maybe normal distribution is more realistic?
    // Using uniform for now.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int n = 2000; 

    // We open engine here
    auto engine = std::move(SearchEngine::Open(cfg).value());

    std::vector<std::vector<float>> data(n, std::vector<float>(cfg.dim));
    for(int i=0; i<n; ++i) {
        float norm_sq = 0.0f;
        for(int d=0; d<cfg.dim; ++d) {
            data[i][d] = dist(rng);
            norm_sq += data[i][d] * data[i][d];
        }
        float inv = 1.0f / std::sqrt(norm_sq);
        for(int d=0; d<cfg.dim; ++d) data[i][d] *= inv;

        engine->Upsert(std::to_string(i), VectorView{data[i].data(), cfg.dim});
    }

    // Check stats (ensure training happened)
    auto stats = engine->GetStats();
    EXPECT_EQ(stats.num_points, static_cast<uint64_t>(n));

    // Check Recall on known points
    int queries = 100;
    int found_top1 = 0;
    int found_top10 = 0;
    
    for(int i=0; i<queries; ++i) {
        auto res = engine->Search(VectorView{data[i].data(), cfg.dim});
        EXPECT_TRUE(res.ok());
        if (res.value().empty()) continue;
        
        // Top 1 check
        if (res.value().front().key == std::to_string(i)) {
            found_top1++;
        }
        
        // Top 10 check
        bool present = false;
        for(const auto& item : res.value()) {
            if (item.key == std::to_string(i)) present = true;
        }
        if (present) found_top10++;
    }

    std::cout << "SQ8 Recall@1: " << found_top1 << "/" << queries << "\n";
    std::cout << "SQ8 Recall@10: " << found_top10 << "/" << queries << "\n";

    // Requirement: > 94%
    EXPECT_TRUE(found_top1 > 94);
    EXPECT_TRUE(found_top10 > 94);
    
    return true;
}

}  // namespace pomai_search::test
