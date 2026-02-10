#include "pomai_search/search_engine.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace pomai_search;

int main(int argc, char** argv) {
  int seconds = 10;
  if (argc > 1) seconds = std::max(1, std::atoi(argv[1]));

  SearchEngineConfig cfg;
  cfg.dim = 16;
  cfg.num_shards = 4;
  cfg.index_type = SearchEngineConfig::IndexType::IvfSq8;
  cfg.ivf_nlist = 32;
  cfg.ivf_nprobe = 8;

  auto eng_or = SearchEngine::Open(cfg);
  if (!eng_or.ok()) {
    std::cerr << eng_or.status().message() << "\n";
    return 1;
  }
  auto engine = std::move(eng_or.value());

  std::atomic<bool> stop{false};
  std::atomic<int> failures{0};

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::vector<std::thread> threads;

  for (int w = 0; w < 4; ++w) {
    threads.emplace_back([&, w]() {
      std::vector<float> v(cfg.dim, 0.0f);
      int i = 0;
      while (!stop.load()) {
        for (int d = 0; d < cfg.dim; ++d) v[d] = static_cast<float>((i + d + w) % 100) / 100.0f;
        auto s = engine->Upsert("k" + std::to_string((i + w * 997) % 100000), VectorView{v.data(), cfg.dim});
        if (!s.ok()) ++failures;
        ++i;
      }
    });
  }

  for (int r = 0; r < 6; ++r) {
    threads.emplace_back([&, r]() {
      std::vector<float> q(cfg.dim, 0.0f);
      int i = 0;
      while (!stop.load()) {
        for (int d = 0; d < cfg.dim; ++d) q[d] = static_cast<float>((i + d + r) % 100) / 100.0f;
        auto res = engine->Search(VectorView{q.data(), cfg.dim});
        if (!res.ok() || res.value().size() > static_cast<size_t>(cfg.topk_default)) {
          ++failures;
        }
        ++i;
      }
    });
  }

  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  stop.store(true);
  for (auto& t : threads) t.detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::cout << "failures=" << failures.load() << "\n";
  return failures.load() == 0 ? 0 : 2;
}
