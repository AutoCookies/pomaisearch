#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "pomai_search/search_engine.h"

namespace pomai_search {

namespace {

struct BenchConfig {
  int n = 10000;
  int dim = 32;
  int queries = 100;
  int topk = 10;
  int ivf_nlist = 100;
  int ivf_nprobe = 100;
  uint32_t seed = 42;
};

BenchConfig ParseArgs(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (arg == "--n") cfg.n = std::stoi(next());
    else if (arg == "--dim") cfg.dim = std::stoi(next());
    else if (arg == "--queries") cfg.queries = std::stoi(next());
    else if (arg == "--ivf_nlist") cfg.ivf_nlist = std::stoi(next());
    else if (arg == "--ivf_nprobe") cfg.ivf_nprobe = std::stoi(next());
  }
  return cfg;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>(p * (values.size() - 1));
  return values[idx];
}

}  // namespace

}  // namespace pomai_search

int main(int argc, char** argv) {
  auto cfg = pomai_search::ParseArgs(argc, argv);
  pomai_search::SearchEngineConfig engine_cfg;
  engine_cfg.dim = cfg.dim;
  engine_cfg.index_type = pomai_search::SearchEngineConfig::IndexType::IvfSq8;
  engine_cfg.ivf_nlist = cfg.ivf_nlist;
  engine_cfg.ivf_nprobe = cfg.ivf_nprobe;
  // Refine factor is hardcoded in implementation (3.0)

  auto engine = std::move(pomai_search::SearchEngine::Open(engine_cfg).value());
  pomai_search::SearchEngineConfig flat_cfg = engine_cfg;
  flat_cfg.index_type = pomai_search::SearchEngineConfig::IndexType::Flat;
  auto flat = std::move(pomai_search::SearchEngine::Open(flat_cfg).value());

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data(cfg.n, std::vector<float>(cfg.dim));
  
  // Normalize data for consistent testing
  for (int i = 0; i < cfg.n; ++i) {
    float norm_sq = 0.0f;
    for (int d = 0; d < cfg.dim; ++d) {
      data[i][d] = dist(rng);
      norm_sq += data[i][d] * data[i][d];
    }
    float inv = 1.0f / std::sqrt(norm_sq);
    for (int d = 0; d < cfg.dim; ++d) data[i][d] *= inv;
  }

  auto ingest_start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.n; ++i) {
    auto view = pomai_search::VectorView{data[i].data(), cfg.dim};
    engine->Upsert(std::to_string(i), view);
    flat->Upsert(std::to_string(i), view);
  }
  auto ingest_end = std::chrono::steady_clock::now();
  double ingest_s = std::chrono::duration_cast<std::chrono::duration<double>>(ingest_end - ingest_start).count();
  
  // Queries (Self Search)
  int recall_1 = 0;
  double recall_sum = 0.0;
  std::vector<double> latencies;
  latencies.reserve(cfg.queries);

  for (int i = 0; i < cfg.queries; ++i) {
    pomai_search::SearchEngine::QueryOptions opts;
    opts.topk = cfg.topk;
    
    auto start = std::chrono::steady_clock::now();
    auto res = engine->Search(pomai_search::VectorView{data[i].data(), cfg.dim}, opts);
    auto end = std::chrono::steady_clock::now();
    
    latencies.push_back(std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count());
    
    if (res.ok() && !res.value().empty() && res.value().front().key == std::to_string(i)) {
      recall_1++;
    }
    auto truth = flat->Search(pomai_search::VectorView{data[i].data(), cfg.dim}, opts);
    if (res.ok() && truth.ok()) {
      std::unordered_set<std::string> truth_keys;
      for (const auto& item : truth.value()) {
        truth_keys.insert(item.key);
      }
      size_t hits = 0;
      for (const auto& item : res.value()) {
        if (truth_keys.count(item.key) > 0) {
          ++hits;
        }
      }
      recall_sum += static_cast<double>(hits) / static_cast<double>(cfg.topk);
    }
  }

  double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  double p95 = pomai_search::Percentile(latencies, 0.95);

  std::cout << "Benchmark SQ8\n";
  std::cout << "Ingest: " << (cfg.n / ingest_s) << " ops/s\n";
  std::cout << "Latency: avg=" << avg << "ms p95=" << p95 << "ms\n";
  std::cout << "Recall@1: " << recall_1 << "/" << cfg.queries << " (" << (100.0 * recall_1 / cfg.queries) << "%)\n";
  std::cout << "Recall@" << cfg.topk << ": " << (recall_sum / static_cast<double>(cfg.queries)) << "\n";
  
  return 0;
}
