#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <iostream>

namespace pomai_search::test {

POMAI_TEST(EngineIvfAutoTrain) {
  SearchEngineConfig cfg;
  cfg.dim = 16;
  cfg.index_type = SearchEngineConfig::IndexType::IvfFlat;
  cfg.ivf_nlist = 5; 
  // Min train size default = 39 * 5 = 195.
  // We will insert 300 items.
  
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  
  // 1. Ingest Data (Random)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  int n = 300;
  
  for(int i=0; i<n; ++i) {
      std::string key = std::to_string(i);
      std::vector<float> data(cfg.dim);
      for(int d=0; d<cfg.dim; ++d) data[d] = dist(rng);
      engine->Upsert(key, VectorView{data.data(), cfg.dim});
  }
  
  // At this point, it should have auto-trained.
  // How to verify? Search should work.
  // We perform a search for a known vector (item 0).
  
  auto vec0 = engine->GetVector("0");
  EXPECT_TRUE(vec0.ok());
  
  auto res = engine->Search(VectorView{vec0.value().data(), cfg.dim});
  EXPECT_TRUE(res.ok());
  // Item 0 should be top result with score ~similarity (Dot product of self).
  // Check if "0" is in top results.
  bool found = false;
  for(const auto& item : res.value()) {
      if (item.key == "0") found = true;
  }
  EXPECT_TRUE(found);
  
  return true;
}

POMAI_TEST(EngineIvfRecall) {
    // Verify that IVF works reasonably well (better than random).
    SearchEngineConfig cfg;
    cfg.dim = 16;
    cfg.index_type = SearchEngineConfig::IndexType::IvfFlat;
    cfg.ivf_nlist = 10;
    cfg.ivf_nprobe = 3; 
    
    auto engine = std::move(SearchEngine::Open(cfg).value());
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int n = 1000; 
    
    std::vector<std::vector<float>> data(n, std::vector<float>(cfg.dim));
    for(int i=0; i<n; ++i) {
        float norm_sq = 0.0f;
        for(int d=0; d<cfg.dim; ++d) {
            data[i][d] = dist(rng);
            norm_sq += data[i][d] * data[i][d];
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for(int d=0; d<cfg.dim; ++d) data[i][d] *= inv_norm;

        auto s = engine->Upsert(std::to_string(i), VectorView{data[i].data(), cfg.dim});
        if (!s.ok()) {
            std::cerr << "Upsert failed at i=" << i << ": " << s.message() << "\n";
        }
    }
    
    auto stats = engine->GetStats();
    EXPECT_EQ(stats.num_points, static_cast<uint64_t>(n));
    
    // Check coverage/recall for random queries.
    int queries = 10;
    int found_self = 0;
    // We search for vectors ALREADY in the set. Recall@TopK should be very high (1.0).
    for(int i=0; i<queries; ++i) {
        auto res = engine->Search(VectorView{data[i].data(), cfg.dim});
        EXPECT_TRUE(res.ok());
        if (!res.value().empty() && res.value().front().key == std::to_string(i)) {
            found_self++;
        }
    }
    // With nprobe=3/10, we expect decent recall.
    // If quantization is working, we should find nearest centroid and find self.
    EXPECT_TRUE(found_self > 5); // At least 50% strict self-match.
    
    return true;
}


}  // namespace pomai_search::test
