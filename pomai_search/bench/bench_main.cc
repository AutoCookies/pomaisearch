#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "pomai_search/search_engine.h"
#include "pomai_search/types.h"

using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;
using pomai_search::VectorView;

static std::vector<std::vector<float>> GenerateVectors(size_t count, int dim) {
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& vec : vectors) {
    for (int i = 0; i < dim; ++i) {
      vec[i] = dist(rng);
    }
  }
  return vectors;
}

static void RunBench(bool avx2_enabled) {
  SearchEngineConfig cfg;
  cfg.dim = 64;
  cfg.num_shards = 4;
  cfg.query_threads = 4;
  cfg.enable_avx2 = avx2_enabled;

  auto engine_result = SearchEngine::Open(cfg);
  if (!engine_result.ok()) {
    std::cerr << engine_result.status().ToString() << "\n";
    return;
  }
  auto engine = std::move(engine_result.value());

  size_t num_vectors = 2000;
  auto vectors = GenerateVectors(num_vectors, cfg.dim);

  auto start_ingest = std::chrono::steady_clock::now();
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorView view{vectors[i].data(), cfg.dim};
    engine->Upsert("key" + std::to_string(i), view);
  }
  auto end_ingest = std::chrono::steady_clock::now();
  double ingest_ms = std::chrono::duration<double, std::milli>(end_ingest - start_ingest).count();

  size_t queries = 200;
  auto start_query = std::chrono::steady_clock::now();
  for (size_t i = 0; i < queries; ++i) {
    VectorView view{vectors[i].data(), cfg.dim};
    engine->Search(view);
  }
  auto end_query = std::chrono::steady_clock::now();
  double query_ms = std::chrono::duration<double, std::milli>(end_query - start_query).count();

  std::cout << (avx2_enabled ? "AVX2" : "Scalar") << " ingest ops/s: "
            << (num_vectors / (ingest_ms / 1000.0)) << "\n";
  std::cout << (avx2_enabled ? "AVX2" : "Scalar") << " query avg ms: "
            << (query_ms / queries) << "\n";
}

int main() {
  RunBench(false);
  RunBench(true);
  return 0;
}
