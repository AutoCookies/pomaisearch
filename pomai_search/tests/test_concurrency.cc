#include <atomic>
#include <thread>
#include <vector>

#include "pomai_search/search_engine.h"
#include "test_framework.h"
#include "thread_pool.h"

using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;
using pomai_search::ThreadPool;
using pomai_search::VectorView;

namespace {

SearchEngineConfig DefaultConfig() {
  SearchEngineConfig cfg;
  cfg.dim = 4;
  cfg.num_shards = 4;
  cfg.query_threads = 4;
  cfg.similarity = SearchEngineConfig::Similarity::Dot;
  cfg.enable_avx2 = false;
  return cfg;
}

}  // namespace

POMAI_TEST(TestThreadPoolShutdown) {
  ThreadPool pool(1);
  auto first = pool.Submit([]() { return 42; });
  EXPECT_TRUE(first.ok());
  EXPECT_EQ(first.value().get(), 42);
  EXPECT_TRUE(pool.Shutdown().ok());
  auto second = pool.Submit([]() { return 7; });
  EXPECT_TRUE(!second.ok());
  return true;
}

POMAI_TEST(TestConcurrentUpsertSearch) {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::atomic<bool> ok{true};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < 50; ++i) {
        std::vector<float> vec{static_cast<float>(t), 0.0f, 0.0f, 0.0f};
        VectorView view{vec.data(), 4};
        if (!engine->Upsert("key" + std::to_string(t * 100 + i), view).ok()) {
          ok.store(false);
        }
        auto result = engine->Search(view);
        if (!result.ok()) {
          ok.store(false);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_TRUE(ok.load());
  auto stats = engine->GetStats();
  EXPECT_TRUE(stats.num_points >= 200);
  return true;
}
