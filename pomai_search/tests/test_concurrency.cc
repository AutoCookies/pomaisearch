#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <thread>
#include <vector>

namespace pomai_search::test {

POMAI_TEST(ConcurrentUpsertSearch) {
  SearchEngineConfig cfg;
  cfg.dim = 3;
  cfg.num_shards = 2;
  cfg.query_threads = 2;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  std::vector<std::thread> writers;
  for (int i = 0; i < 10; ++i) {
    writers.emplace_back([&, i]() {
      float v[3] = {static_cast<float>(i), 0.0f, 1.0f};
      engine->Upsert("k" + std::to_string(i), VectorView{v, 3});
    });
  }
  for (auto& t : writers) {
    t.join();
  }
  float q[3] = {1.0f, 0.0f, 1.0f};
  std::atomic<bool> ok{true};
  std::vector<std::thread> readers;
  for (int i = 0; i < 5; ++i) {
    readers.emplace_back([&]() {
      auto result = engine->Search(VectorView{q, 3});
      if (!result.ok()) {
        ok.store(false);
      }
    });
  }
  for (auto& t : readers) {
    t.join();
  }
  EXPECT_TRUE(ok.load());
  return true;
}

}  // namespace pomai_search::test
