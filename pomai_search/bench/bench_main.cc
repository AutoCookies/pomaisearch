#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "pomai_search/search_engine.h"
#include "pomai_search/types.h"
#include "simd/kernels.h"

using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;
using pomai_search::VectorView;

namespace {

struct BenchConfig {
  size_t n = 10000;
  int dim = 128;
  size_t queries = 1000;
  int topk = 10;
  int shards = 4;
  int threads = 4;
  std::string similarity = "dot";
  std::string avx2 = "on";
  int seed = 42;
};

BenchConfig ParseArgs(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      cfg.n = static_cast<size_t>(std::stoull(argv[++i]));
    } else if (arg == "--dim" && i + 1 < argc) {
      cfg.dim = std::stoi(argv[++i]);
    } else if (arg == "--queries" && i + 1 < argc) {
      cfg.queries = static_cast<size_t>(std::stoull(argv[++i]));
    } else if (arg == "--topk" && i + 1 < argc) {
      cfg.topk = std::stoi(argv[++i]);
    } else if (arg == "--shards" && i + 1 < argc) {
      cfg.shards = std::stoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      cfg.threads = std::stoi(argv[++i]);
    } else if (arg == "--similarity" && i + 1 < argc) {
      cfg.similarity = argv[++i];
    } else if (arg == "--avx2" && i + 1 < argc) {
      cfg.avx2 = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      cfg.seed = std::stoi(argv[++i]);
    }
  }
  return cfg;
}

std::vector<std::vector<float>> GenerateVectors(size_t count, int dim, int seed) {
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& vec : vectors) {
    for (int i = 0; i < dim; ++i) {
      vec[i] = dist(rng);
    }
  }
  return vectors;
}

struct BenchResult {
  double ingest_ops_per_s = 0.0;
  double query_avg_ms = 0.0;
  double query_p50_ms = 0.0;
  double query_p95_ms = 0.0;
};

BenchResult RunBench(const BenchConfig& cfg, bool enable_avx2) {
  SearchEngineConfig engine_cfg;
  engine_cfg.dim = cfg.dim;
  engine_cfg.num_shards = cfg.shards;
  engine_cfg.query_threads = cfg.threads;
  engine_cfg.ingest_threads = cfg.threads;
  engine_cfg.topk_default = cfg.topk;
  engine_cfg.enable_avx2 = enable_avx2;
  engine_cfg.similarity = (cfg.similarity == "cosine") ? SearchEngineConfig::Similarity::Cosine
                                                         : SearchEngineConfig::Similarity::Dot;

  auto engine_result = SearchEngine::Open(engine_cfg);
  if (!engine_result.ok()) {
    std::cerr << engine_result.status().ToString() << "\n";
    return {};
  }
  auto engine = std::move(engine_result.value());

  auto vectors = GenerateVectors(cfg.n, cfg.dim, cfg.seed);

  for (size_t i = 0; i < std::min<size_t>(cfg.n, 100); ++i) {
    VectorView view{vectors[i].data(), cfg.dim};
    engine->Upsert("warmup" + std::to_string(i), view);
  }

  auto start_ingest = std::chrono::steady_clock::now();
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorView view{vectors[i].data(), cfg.dim};
    engine->Upsert("key" + std::to_string(i), view);
  }
  auto end_ingest = std::chrono::steady_clock::now();
  double ingest_ms = std::chrono::duration<double, std::milli>(end_ingest - start_ingest).count();

  std::vector<double> latencies_ms;
  latencies_ms.reserve(cfg.queries);
  for (size_t i = 0; i < std::min<size_t>(cfg.queries, 100); ++i) {
    VectorView view{vectors[i % vectors.size()].data(), cfg.dim};
    engine->Search(view);
  }

  auto start_query = std::chrono::steady_clock::now();
  for (size_t i = 0; i < cfg.queries; ++i) {
    VectorView view{vectors[i % vectors.size()].data(), cfg.dim};
    auto start = std::chrono::steady_clock::now();
    engine->Search(view);
    auto end = std::chrono::steady_clock::now();
    latencies_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  auto end_query = std::chrono::steady_clock::now();
  double total_query_ms = std::chrono::duration<double, std::milli>(end_query - start_query).count();

  std::sort(latencies_ms.begin(), latencies_ms.end());
  auto percentile = [&](double p) {
    if (latencies_ms.empty()) {
      return 0.0;
    }
    size_t idx = static_cast<size_t>(p * (latencies_ms.size() - 1));
    return latencies_ms[idx];
  };

  BenchResult result;
  result.ingest_ops_per_s = cfg.n / (ingest_ms / 1000.0);
  result.query_avg_ms = total_query_ms / static_cast<double>(cfg.queries);
  result.query_p50_ms = percentile(0.50);
  result.query_p95_ms = percentile(0.95);
  return result;
}

void PrintResult(const std::string& label, const BenchResult& result) {
  std::cout << label << " ingest ops/s: " << result.ingest_ops_per_s << "\n";
  std::cout << label << " query avg ms: " << result.query_avg_ms << "\n";
  std::cout << label << " query p50 ms: " << result.query_p50_ms << "\n";
  std::cout << label << " query p95 ms: " << result.query_p95_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  auto cfg = ParseArgs(argc, argv);
  bool want_scalar = cfg.avx2 == "off" || cfg.avx2 == "both";
  bool want_avx2 = cfg.avx2 == "on" || cfg.avx2 == "both";

  BenchResult scalar_result;
  BenchResult avx2_result;

  if (want_scalar) {
    scalar_result = RunBench(cfg, false);
    PrintResult("Scalar", scalar_result);
  }
  if (want_avx2) {
    if (!pomai_search::CpuSupportsAvx2()) {
      std::cout << "AVX2 not supported on this CPU, skipping" << "\n";
    } else {
      avx2_result = RunBench(cfg, true);
      PrintResult("AVX2", avx2_result);
    }
  }
  if (want_scalar && want_avx2 && avx2_result.query_avg_ms > 0.0) {
    std::cout << "Speedup (avg): " << (scalar_result.query_avg_ms / avx2_result.query_avg_ms)
              << "x\n";
  }
  return 0;
}
