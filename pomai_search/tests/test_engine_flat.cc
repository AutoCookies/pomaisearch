#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <limits>

namespace pomai_search::test {

POMAI_TEST(EngineFlatSearch) {
  SearchEngineConfig cfg;
  cfg.dim = 3;
  cfg.num_shards = 2;
  cfg.index_type = SearchEngineConfig::IndexType::Flat;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[3] = {1.0f, 0.0f, 0.0f};
  float b[3] = {0.0f, 1.0f, 0.0f};
  engine->Upsert("a", VectorView{a, 3});
  engine->Upsert("b", VectorView{b, 3});
  float qv[3] = {1.0f, 0.1f, 0.0f};
  auto results = engine->Search(VectorView{qv, 3});
  EXPECT_TRUE(results.ok());
  EXPECT_TRUE(!results.value().empty());
  EXPECT_EQ(results.value().front().key, "a");
  return true;
}

POMAI_TEST(EngineFlatNanScoresLast) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.index_type = SearchEngineConfig::IndexType::Flat;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float good[2] = {1.0f, 0.0f};
  float nan_vec[2] = {std::numeric_limits<float>::quiet_NaN(), 0.0f};
  engine->Upsert("good", VectorView{good, 2});
  engine->Upsert("nan", VectorView{nan_vec, 2});
  float query[2] = {1.0f, 0.0f};
  auto results = engine->Search(VectorView{query, 2});
  EXPECT_TRUE(results.ok());
  EXPECT_TRUE(results.value().size() >= 2);
  EXPECT_EQ(results.value().front().key, "good");
  return true;
}

}  // namespace pomai_search::test
