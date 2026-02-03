#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <random>

namespace pomai_search::test {

POMAI_TEST(DeterministicAcrossRuns) {
  SearchEngineConfig cfg;
  cfg.dim = 3;
  cfg.num_shards = 4;
  cfg.query_threads = 2;
  cfg.index_type = SearchEngineConfig::IndexType::Flat;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < 50; ++i) {
    std::vector<float> vec(cfg.dim);
    for (int d = 0; d < cfg.dim; ++d) {
      vec[d] = dist(rng);
    }
    engine->Upsert("key" + std::to_string(i), VectorView{vec.data(), cfg.dim});
  }
  std::vector<float> query(cfg.dim, 0.5f);
  auto first = engine->Search(VectorView{query.data(), cfg.dim});
  EXPECT_TRUE(first.ok());
  std::string snapshot;
  for (const auto& item : first.value()) {
    snapshot += item.key + ",";
  }
  for (int i = 0; i < 100; ++i) {
    auto result = engine->Search(VectorView{query.data(), cfg.dim});
    EXPECT_TRUE(result.ok());
    std::string current;
    for (const auto& item : result.value()) {
      current += item.key + ",";
    }
    EXPECT_EQ(snapshot, current);
  }
  return true;
}

POMAI_TEST(NoRegressionSnapshot) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  float c[2] = {0.5f, 0.5f};
  engine->Upsert("doc-a", VectorView{a, 2});
  engine->Upsert("doc-b", VectorView{b, 2});
  engine->Upsert("doc-c", VectorView{c, 2});
  float q[2] = {1.0f, 0.2f};
  auto result = engine->Search(VectorView{q, 2});
  EXPECT_TRUE(result.ok());
  std::string snapshot;
  for (const auto& item : result.value()) {
    snapshot += item.key + ",";
  }
  EXPECT_EQ(snapshot, "doc-a,doc-c,doc-b,");
  return true;
}

}  // namespace pomai_search::test
