#include <functional>
#include <string>
#include <vector>

#include "pomai_search/search_engine.h"
#include "test_framework.h"

using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;
using pomai_search::VectorView;

namespace {

int ShardForKey(std::string_view key, int num_shards) {
  return static_cast<int>(std::hash<std::string_view>{}(key) % static_cast<size_t>(num_shards));
}

std::string FindKeyForShard(int target_shard, int num_shards) {
  for (int i = 0; i < 10000; ++i) {
    std::string key = "key" + std::to_string(i);
    if (ShardForKey(key, num_shards) == target_shard) {
      return key;
    }
  }
  return "fallback";
}

SearchEngineConfig DefaultConfig() {
  SearchEngineConfig cfg;
  cfg.dim = 4;
  cfg.num_shards = 2;
  cfg.similarity = SearchEngineConfig::Similarity::Dot;
  cfg.enable_avx2 = false;
  return cfg;
}

}  // namespace

POMAI_TEST(TestScopeLocalVector) {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{1.0f, 0.0f, 0.0f, 0.0f};
  VectorView view{vec.data(), 4};
  std::string shard1_key = FindKeyForShard(1, 2);
  EXPECT_TRUE(engine->Upsert(shard1_key, view).ok());
  SearchEngine::QueryOptions options;
  options.scope = SearchEngine::QueryOptions::Scope::Local;
  auto local = engine->Search(view, options);
  EXPECT_TRUE(local.ok());
  EXPECT_TRUE(local.value().empty());
  return true;
}

POMAI_TEST(TestScopeLocalByKey) {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{0.1f, 0.2f, 0.3f, 0.4f};
  VectorView view{vec.data(), 4};
  std::string key = FindKeyForShard(1, 2);
  EXPECT_TRUE(engine->Upsert(key, view).ok());
  SearchEngine::QueryOptions options;
  options.scope = SearchEngine::QueryOptions::Scope::Local;
  options.topk = 1;
  auto local = engine->SearchByKey(key, options);
  EXPECT_TRUE(local.ok());
  EXPECT_EQ(local.value().size(), 1u);
  EXPECT_EQ(local.value()[0].key, key);
  return true;
}

POMAI_TEST(TestDeterministicResults) {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{1.0f, 0.0f, 0.0f, 0.0f};
  VectorView view{vec.data(), 4};
  EXPECT_TRUE(engine->Upsert("alpha", view).ok());
  EXPECT_TRUE(engine->Upsert("beta", view).ok());
  auto result1 = engine->Search(view);
  auto result2 = engine->Search(view);
  EXPECT_TRUE(result1.ok());
  EXPECT_TRUE(result2.ok());
  EXPECT_EQ(result1.value().size(), result2.value().size());
  EXPECT_EQ(result1.value()[0].key, result2.value()[0].key);
  EXPECT_EQ(result1.value()[1].key, result2.value()[1].key);
  return true;
}
