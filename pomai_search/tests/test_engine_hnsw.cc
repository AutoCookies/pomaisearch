#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

namespace pomai_search::test {

POMAI_TEST(EngineHnswSearch) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.index_type = SearchEngineConfig::IndexType::Hnsw;
  cfg.hnsw_m = 8;
  cfg.hnsw_ef_construction = 50;
  cfg.hnsw_ef_search = 20;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  float c[2] = {0.9f, 0.1f};
  engine->Upsert("a", VectorView{a, 2});
  engine->Upsert("b", VectorView{b, 2});
  engine->Upsert("c", VectorView{c, 2});
  float qv[2] = {1.0f, 0.0f};
  auto results = engine->Search(VectorView{qv, 2});
  EXPECT_TRUE(results.ok());
  EXPECT_EQ(results.value().front().key, "a");
  return true;
}

}  // namespace pomai_search::test
