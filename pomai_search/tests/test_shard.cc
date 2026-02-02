#include <optional>
#include <vector>

#include "shard.h"
#include "test_framework.h"

using pomai_search::Metadata;
using pomai_search::SearchEngineConfig;
using pomai_search::Shard;
using pomai_search::VectorView;

POMAI_TEST(TestShardTopKDeterminism) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.similarity = SearchEngineConfig::Similarity::Dot;
  Shard shard(cfg.dim, 32, 0, cfg.similarity, pomai_search::DotScalar, 0);

  std::vector<float> v1{1.0f, 0.0f};
  std::vector<float> v2{1.0f, 0.0f};
  std::vector<float> v3{0.5f, 0.0f};
  shard.Upsert("b", {v1.data(), 2}, Metadata{}, std::nullopt);
  shard.Upsert("a", {v2.data(), 2}, Metadata{}, std::nullopt);
  shard.Upsert("c", {v3.data(), 2}, Metadata{}, std::nullopt);

  auto result = shard.Search({v1.data(), 2}, 2, Metadata{}, 1.0f);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().size(), 2u);
  EXPECT_EQ(result.value()[0].key, "a");
  EXPECT_EQ(result.value()[1].key, "b");
  return true;
}

POMAI_TEST(TestVectorArenaAlignment) {
  pomai_search::VectorArena arena(4, 32, 1);
  std::vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
  arena.Append(vec.data());
  auto ptr = reinterpret_cast<uintptr_t>(arena.data());
  EXPECT_EQ(ptr % 32u, 0u);
  return true;
}
