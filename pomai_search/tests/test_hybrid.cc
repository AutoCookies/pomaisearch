#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

namespace pomai_search::test {

POMAI_TEST(HybridKeywordOnly) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  engine->Upsert("a", VectorView{a, 2}, {}, std::nullopt, std::string("red apple"));
  engine->Upsert("b", VectorView{b, 2}, {}, std::nullopt, std::string("blue berry"));
  SearchEngine::HybridQuery query;
  query.text_query = "apple";
  auto results = engine->SearchHybrid(query);
  EXPECT_TRUE(results.ok());
  EXPECT_EQ(results.value().front().key, "a");
  return true;
}

POMAI_TEST(HybridFusion) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.9f, 0.1f};
  engine->Upsert("a", VectorView{a, 2}, {}, std::nullopt, std::string("orange"));
  engine->Upsert("b", VectorView{b, 2}, {}, std::nullopt, std::string("orange juice"));
  SearchEngine::HybridQuery query;
  query.text_query = "orange";
  query.vector_query = std::vector<float>{1.0f, 0.0f};
  query.alpha = 0.7f;
  auto results = engine->SearchHybrid(query);
  EXPECT_TRUE(results.ok());
  EXPECT_EQ(results.value().front().key, "a");
  return true;
}

}  // namespace pomai_search::test
