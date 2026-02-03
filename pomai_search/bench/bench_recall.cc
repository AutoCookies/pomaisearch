#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "pomai_search/index/flat_index.h"
#include "pomai_search/index/hnsw_index.h"
#include "pomai_search/simd/kernels.h"
#include "pomai_search/vector_store.h"

namespace pomai_search {

namespace {

struct BenchConfig {
  int n = 1000;
  int dim = 32;
  int queries = 100;
  int topk = 10;
  bool avx2 = true;
  uint32_t seed = 42;
  int hnsw_m = 16;
  int hnsw_ef_construction = 200;
  int hnsw_ef_search = 50;
};

BenchConfig ParseArgs(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() {
      if (i + 1 >= argc) {
        return std::string();
      }
      return std::string(argv[++i]);
    };
    if (arg == "--n") {
      cfg.n = std::stoi(next());
    } else if (arg == "--dim") {
      cfg.dim = std::stoi(next());
    } else if (arg == "--queries") {
      cfg.queries = std::stoi(next());
    } else if (arg == "--topk") {
      cfg.topk = std::stoi(next());
    } else if (arg == "--avx2") {
      cfg.avx2 = next() == "on";
    } else if (arg == "--seed") {
      cfg.seed = static_cast<uint32_t>(std::stoul(next()));
    } else if (arg == "--hnsw_m") {
      cfg.hnsw_m = std::stoi(next());
    } else if (arg == "--hnsw_ef_construction") {
      cfg.hnsw_ef_construction = std::stoi(next());
    } else if (arg == "--hnsw_ef_search") {
      cfg.hnsw_ef_search = std::stoi(next());
    }
  }
  return cfg;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>(p * (values.size() - 1));
  return values[idx];
}

}  // namespace

}  // namespace pomai_search

int main(int argc, char** argv) {
  auto cfg = pomai_search::ParseArgs(argc, argv);
  pomai_search::DotFunc dot_func = pomai_search::GetDotFunc(cfg.avx2);
  pomai_search::VectorStore store(cfg.dim, 32, cfg.n);
  pomai_search::FlatIndex flat(&store, cfg.dim, pomai_search::SearchEngineConfig::Similarity::Dot, dot_func,
                               32);
  pomai_search::HnswIndex hnsw(&store, cfg.dim, pomai_search::SearchEngineConfig::Similarity::Dot, dot_func,
                               cfg.n, cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef_search,
                               cfg.seed);

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data(cfg.n, std::vector<float>(cfg.dim));
  for (int i = 0; i < cfg.n; ++i) {
    float norm_sq = 0.0f;
    for (int d = 0; d < cfg.dim; ++d) {
      data[i][d] = dist(rng);
      norm_sq += data[i][d] * data[i][d];
    }
    float norm = std::sqrt(norm_sq);
    size_t offset = store.Append(data[i].data());
    flat.Upsert(i, offset, norm);
    hnsw.Upsert(i, offset, norm);
  }

  std::vector<std::vector<float>> queries(cfg.queries, std::vector<float>(cfg.dim));
  for (int i = 0; i < cfg.queries; ++i) {
    for (int d = 0; d < cfg.dim; ++d) {
      queries[i][d] = dist(rng);
    }
  }

  std::vector<double> flat_latencies;
  std::vector<double> hnsw_latencies;
  double recall_hits = 0.0;
  for (int i = 0; i < cfg.queries; ++i) {
    pomai_search::VectorView view{queries[i].data(), cfg.dim};
    auto start = std::chrono::steady_clock::now();
    auto flat_results = flat.Search(view, cfg.topk, pomai_search::Filter{});
    auto mid = std::chrono::steady_clock::now();
    auto hnsw_results = hnsw.Search(view, cfg.topk, pomai_search::Filter{});
    auto end = std::chrono::steady_clock::now();
    flat_latencies.push_back(
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(mid - start).count());
    hnsw_latencies.push_back(
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - mid).count());
    if (!flat_results.ok() || !hnsw_results.ok()) {
      continue;
    }
    std::unordered_set<uint32_t> truth;
    for (const auto& cand : flat_results.value()) {
      truth.insert(cand.id);
    }
    int hits = 0;
    for (const auto& cand : hnsw_results.value()) {
      if (truth.find(cand.id) != truth.end()) {
        ++hits;
      }
    }
    recall_hits += static_cast<double>(hits) / static_cast<double>(cfg.topk);
  }
  double recall = recall_hits / static_cast<double>(cfg.queries);

  double flat_avg = std::accumulate(flat_latencies.begin(), flat_latencies.end(), 0.0) /
                    flat_latencies.size();
  double flat_p50 = pomai_search::Percentile(flat_latencies, 0.50);
  double flat_p95 = pomai_search::Percentile(flat_latencies, 0.95);
  double hnsw_avg = std::accumulate(hnsw_latencies.begin(), hnsw_latencies.end(), 0.0) /
                    hnsw_latencies.size();
  double hnsw_p50 = pomai_search::Percentile(hnsw_latencies, 0.50);
  double hnsw_p95 = pomai_search::Percentile(hnsw_latencies, 0.95);

  std::cout << "Recall benchmark\n";
  std::cout << "recall@" << cfg.topk << ": " << recall << "\n";
  std::cout << "flat avg ms: " << flat_avg << " p50: " << flat_p50 << " p95: " << flat_p95 << "\n";
  std::cout << "hnsw avg ms: " << hnsw_avg << " p50: " << hnsw_p50 << " p95: " << hnsw_p95 << "\n";
  return 0;
}
