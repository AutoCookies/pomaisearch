#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <string>
#include <vector>

namespace pomai_search::test {

POMAI_TEST(EngineIvfUpsertDriftBounded) {
  SearchEngineConfig cfg;
  cfg.dim = 8;
  cfg.index_type = SearchEngineConfig::IndexType::IvfFlat;
  cfg.ivf_nlist = 4;
  cfg.ivf_nprobe = 2;

  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());

  std::vector<float> base(cfg.dim, 0.1f);
  for (int i = 0; i < 500; ++i) {
    std::string k = "seed_" + std::to_string(i);
    base[0] = static_cast<float>(i) * 0.001f;
    EXPECT_TRUE(engine->Upsert(k, VectorView{base.data(), cfg.dim}).ok());
  }

  std::vector<float> vec(cfg.dim, 0.2f);
  for (int i = 0; i < 20000; ++i) {
    vec[0] = static_cast<float>(i % 17) * 0.01f;
    EXPECT_TRUE(engine->Upsert("hot", VectorView{vec.data(), cfg.dim}).ok());
  }

  auto stats = engine->GetStats();
  EXPECT_TRUE(stats.num_points <= 600);

  auto res = engine->Search(VectorView{vec.data(), cfg.dim});
  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(!res.value().empty());
  EXPECT_EQ(res.value().front().key, "hot");
  return true;
}

}  // namespace pomai_search::test
