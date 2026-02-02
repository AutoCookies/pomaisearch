#include <chrono>
#include <atomic>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "pomai_search/search_engine.h"
#include "pomai_search/status.h"
#include "pomai_search/types.h"
#include "simd/kernels.h"

using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;
using pomai_search::Status;
using pomai_search::StatusCode;
using pomai_search::StatusOr;
using pomai_search::VectorView;

#define EXPECT_TRUE(cond)                                  \
  do {                                                     \
    if (!(cond)) {                                         \
      std::cerr << "Expectation failed: " #cond "\n";    \
      return false;                                        \
    }                                                      \
  } while (0)

#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))

static bool TestStatus() {
  Status ok;
  EXPECT_TRUE(ok.ok());
  Status err(StatusCode::kInvalidArgument, "bad");
  EXPECT_TRUE(!err.ok());
  EXPECT_TRUE(err.ToString().find("InvalidArgument") != std::string::npos);
  return true;
}

static SearchEngineConfig DefaultConfig() {
  SearchEngineConfig cfg;
  cfg.dim = 4;
  cfg.num_shards = 2;
  cfg.similarity = SearchEngineConfig::Similarity::Dot;
  cfg.enable_avx2 = false;
  return cfg;
}

static bool TestUpsertSearch() {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{1.0f, 0.0f, 0.0f, 0.0f};
  VectorView view{vec.data(), 4};
  EXPECT_TRUE(engine->Upsert("alpha", view).ok());
  auto search = engine->Search(view);
  EXPECT_TRUE(search.ok());
  EXPECT_TRUE(!search.value().empty());
  EXPECT_EQ(search.value()[0].key, "alpha");
  return true;
}

static bool TestDeleteAndExists() {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{1.0f, 1.0f, 1.0f, 1.0f};
  VectorView view{vec.data(), 4};
  EXPECT_TRUE(engine->Upsert("beta", view).ok());
  auto exists = engine->Exists("beta");
  EXPECT_TRUE(exists.ok());
  EXPECT_TRUE(exists.value());
  EXPECT_TRUE(engine->Delete("beta").ok());
  auto exists_after = engine->Exists("beta");
  EXPECT_TRUE(exists_after.ok());
  EXPECT_TRUE(!exists_after.value());
  return true;
}

static bool TestTtlExpiry() {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{0.5f, 0.5f, 0.5f, 0.5f};
  VectorView view{vec.data(), 4};
  EXPECT_TRUE(engine->Upsert("ttl", view, {}, std::chrono::milliseconds(1)).ok());
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto exists = engine->Exists("ttl");
  EXPECT_TRUE(exists.ok());
  EXPECT_TRUE(!exists.value());
  return true;
}

static bool TestTieBreak() {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  std::vector<float> vec{1.0f, 0.0f, 0.0f, 0.0f};
  VectorView view{vec.data(), 4};
  EXPECT_TRUE(engine->Upsert("alpha", view).ok());
  EXPECT_TRUE(engine->Upsert("beta", view).ok());
  SearchEngine::QueryOptions options;
  options.topk = 2;
  auto search = engine->Search(view, options);
  EXPECT_TRUE(search.ok());
  EXPECT_EQ(search.value().size(), 2u);
  EXPECT_EQ(search.value()[0].key, "alpha");
  EXPECT_EQ(search.value()[1].key, "beta");
  return true;
}

static bool TestSimdScalarClose() {
  std::vector<float> a(32);
  std::vector<float> b(32);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i) * 0.1f;
    b[i] = static_cast<float>(i) * 0.05f;
  }
  float scalar = pomai_search::DotScalar(a.data(), b.data(), static_cast<int>(a.size()));
  float avx = pomai_search::DotAvx2(a.data(), b.data(), static_cast<int>(a.size()));
  EXPECT_TRUE(std::fabs(scalar - avx) < 1e-4f);
  return true;
}

static bool TestConcurrency() {
  auto engine_result = SearchEngine::Open(DefaultConfig());
  EXPECT_TRUE(engine_result.ok());
  auto engine = std::move(engine_result.value());
  const int num_threads = 4;
  std::atomic<bool> ok{true};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < 50; ++i) {
        std::vector<float> vec{static_cast<float>(t), 0.0f, 0.0f, 0.0f};
        VectorView view{vec.data(), 4};
        engine->Upsert("key" + std::to_string(t * 100 + i), view);
        auto search = engine->Search(view);
        if (!search.ok()) {
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

int main() {
  struct TestCase {
    const char* name;
    bool (*func)();
  };
  std::vector<TestCase> tests = {
      {"Status", TestStatus},
      {"UpsertSearch", TestUpsertSearch},
      {"DeleteExists", TestDeleteAndExists},
      {"TtlExpiry", TestTtlExpiry},
      {"TieBreak", TestTieBreak},
      {"SimdScalar", TestSimdScalarClose},
      {"Concurrency", TestConcurrency},
  };

  for (const auto& test : tests) {
    if (!test.func()) {
      std::cerr << "Test failed: " << test.name << "\n";
      return 1;
    }
  }
  std::cout << "All tests passed\n";
  return 0;
}
