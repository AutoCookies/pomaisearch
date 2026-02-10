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
  int n = 1000;
  int dim = 32;
  int queries = 100;
  int topk = 10;
  int shards = 1;
  int threads = 0;
  SearchEngineConfig::Similarity similarity = SearchEngineConfig::Similarity::Dot;
  bool avx2 = true;
  uint32_t seed = 42;
  int ivf_nlist = 100;
  int ivf_nprobe = 10;
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
    } else if (arg == "--shards") {
      cfg.shards = std::stoi(next());
    } else if (arg == "--threads") {
      cfg.threads = std::stoi(next());
    } else if (arg == "--similarity") {
      std::string value = next();
      cfg.similarity = value == "cosine" ? SearchEngineConfig::Similarity::Cosine
                                          : SearchEngineConfig::Similarity::Dot;
    } else if (arg == "--avx2") {
      cfg.avx2 = next() == "on";
    } else if (arg == "--seed") {
      cfg.seed = static_cast<uint32_t>(std::stoul(next()));
    } else if (arg == "--ivf_nlist") {
      cfg.ivf_nlist = std::stoi(next());
    } else if (arg == "--ivf_nprobe") {
      cfg.ivf_nprobe = std::stoi(next());
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
  pomai_search::SearchEngineConfig engine_cfg;
  engine_cfg.dim = cfg.dim;
  engine_cfg.num_shards = cfg.shards;
  engine_cfg.index_type = pomai_search::SearchEngineConfig::IndexType::IvfFlat;
  engine_cfg.similarity = cfg.similarity;
  engine_cfg.enable_avx2 = cfg.avx2;
  engine_cfg.query_threads = cfg.threads;
  engine_cfg.ivf_nlist = cfg.ivf_nlist;
  engine_cfg.ivf_nprobe = cfg.ivf_nprobe;

  auto engine_or = pomai_search::SearchEngine::Open(engine_cfg);
  if (!engine_or.ok()) {
    std::cerr << "Failed to open engine: " << engine_or.status().ToString() << "\n";
    return 1;
  }
  auto engine = std::move(engine_or.value());
  pomai_search::SearchEngineConfig flat_cfg = engine_cfg;
  flat_cfg.index_type = pomai_search::SearchEngineConfig::IndexType::Flat;
  auto flat_or = pomai_search::SearchEngine::Open(flat_cfg);
  if (!flat_or.ok()) {
    std::cerr << "Failed to open flat engine: " << flat_or.status().ToString() << "\n";
    return 1;
  }
  auto flat = std::move(flat_or.value());

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data(cfg.n, std::vector<float>(cfg.dim));
  for (int i = 0; i < cfg.n; ++i) {
    for (int d = 0; d < cfg.dim; ++d) {
      data[i][d] = dist(rng);
    }
  }

  auto ingest_start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.n; ++i) {
    auto view = pomai_search::VectorView{data[i].data(), cfg.dim};
    engine->Upsert("doc" + std::to_string(i), view);
    flat->Upsert("doc" + std::to_string(i), view);
  }
  auto ingest_end = std::chrono::steady_clock::now();
  double ingest_s = std::chrono::duration_cast<std::chrono::duration<double>>(ingest_end - ingest_start).count();
  double ingest_rate = cfg.n / ingest_s;

  std::vector<std::vector<float>> queries(cfg.queries, std::vector<float>(cfg.dim));
  for (int i = 0; i < cfg.queries; ++i) {
    for (int d = 0; d < cfg.dim; ++d) {
      queries[i][d] = dist(rng);
    }
  }
  // Warmup
  for (int i = 0; i < std::min(cfg.queries, 10); ++i) {
    pomai_search::SearchEngine::QueryOptions opts;
    opts.topk = cfg.topk;
    engine->Search(pomai_search::VectorView{queries[i].data(), cfg.dim}, opts);
  }

  std::vector<double> latencies;
  latencies.reserve(cfg.queries);
  double recall_sum = 0.0;
  for (int i = 0; i < cfg.queries; ++i) {
    pomai_search::SearchEngine::QueryOptions opts;
    opts.topk = cfg.topk;
    auto start = std::chrono::steady_clock::now();
    auto approx = engine->Search(pomai_search::VectorView{queries[i].data(), cfg.dim}, opts);
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    latencies.push_back(ms);
    auto truth = flat->Search(pomai_search::VectorView{queries[i].data(), cfg.dim}, opts);
    if (approx.ok() && truth.ok()) {
      std::unordered_set<std::string> truth_keys;
      for (const auto& item : truth.value()) {
        truth_keys.insert(item.key);
      }
      size_t hits = 0;
      for (const auto& item : approx.value()) {
        if (truth_keys.count(item.key) > 0) {
          ++hits;
        }
      }
      recall_sum += static_cast<double>(hits) / static_cast<double>(cfg.topk);
    }
  }
  double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  double p50 = pomai_search::Percentile(latencies, 0.50);
  double p95 = pomai_search::Percentile(latencies, 0.95);
  double recall = latencies.empty() ? 0.0 : recall_sum / static_cast<double>(cfg.queries);

  std::cout << "Benchmark ivf\n";
  std::cout << "ingest ops/s: " << ingest_rate << "\n";
  std::cout << "query avg ms: " << avg << " p50: " << p50 << " p95: " << p95 << "\n";
  std::cout << "recall@" << cfg.topk << ": " << recall << "\n";
  return 0;
}
