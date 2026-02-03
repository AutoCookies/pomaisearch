#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"
#include <set>

namespace pomai_search::test {

POMAI_TEST(EngineShardingScatterGather) {
  SearchEngineConfig cfg;
  cfg.dim = 4;
  cfg.num_shards = 2; // Force 2 shards
  cfg.max_points_per_shard = 100;
  cfg.query_threads = 2; // Parallel execution
  
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());

  float vec[4] = {0.1f, 0.2f, 0.3f, 0.4f};
  VectorView view{vec, 4};
  
  // Insert 20 docs. With 2 shards, statistically we should have docs on both.
  for(int i=0; i<20; ++i) {
      EXPECT_TRUE(engine->Upsert("doc" + std::to_string(i), view).ok());
  }

  // Search
  SearchEngine::QueryOptions opt;
  opt.topk = 50;
  auto res_or = engine->Search(view, opt);
  EXPECT_TRUE(res_or.ok());
  auto results = res_or.value();
  
  // Verify we got all 20 results (since vector is identical)
  EXPECT_EQ(results.size(), 20);
  
  // Also verify internal IDs or just that we have 20 unique keys
  std::set<std::string> keys;
  for(const auto& r : results) keys.insert(r.key);
  EXPECT_EQ(keys.size(), 20);
  
  engine->Close();
  return true;
}

POMAI_TEST(EngineHybridSharding) {
    SearchEngineConfig cfg;
    cfg.dim = 4;
    cfg.num_shards = 4; 
    cfg.query_threads = 4; 
    
    auto engine_or = SearchEngine::Open(cfg);
    EXPECT_TRUE(engine_or.ok());
    auto engine = std::move(engine_or.value());
  
    float vec[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    VectorView view{vec, 4};
    
    for(int i=0; i<40; ++i) {
        EXPECT_TRUE(engine->Upsert("hdoc" + std::to_string(i), view, {}, std::nullopt, "banana " + std::to_string(i)).ok());
    }
    
    SearchEngine::HybridQuery hq;
    hq.text_query = "banana";
    hq.vector_query = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
    
    SearchEngine::QueryOptions opt;
    opt.topk = 100;
    
    auto res_or = engine->SearchHybrid(hq, opt);
    EXPECT_TRUE(res_or.ok());
    EXPECT_EQ(res_or.value().size(), 40);
    
    engine->Close();
    return true;
}

} // namespace pomai_search::test
