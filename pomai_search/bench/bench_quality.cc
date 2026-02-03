#include "pomai_search/search_engine.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_set>

namespace pomai_search {
namespace {

std::vector<float> RandomVector(int dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> vec(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    vec[static_cast<size_t>(i)] = dist(rng);
  }
  return vec;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>(p * (values.size() - 1));
  return values[idx];
}

double RecallAtK(const std::vector<ResultItem>& truth, const std::vector<ResultItem>& approx) {
  if (truth.empty()) {
    return 0.0;
  }
  std::unordered_set<std::string> truth_keys;
  for (const auto& item : truth) {
    truth_keys.insert(item.key);
  }
  size_t hit = 0;
  for (const auto& item : approx) {
    if (truth_keys.count(item.key) > 0) {
      ++hit;
    }
  }
  return static_cast<double>(hit) / truth.size();
}

}  // namespace
}  // namespace pomai_search

int main() {
  using namespace pomai_search;
  const int dim = 64;
  const int points = 1000;
  const int queries = 100;
  const int topk = 10;
  const uint64_t seed = 123;

  SearchEngineConfig flat_cfg;
  flat_cfg.dim = dim;
  flat_cfg.num_shards = 1;
  flat_cfg.index_type = SearchEngineConfig::IndexType::Flat;
  flat_cfg.global_seed = seed;

  SearchEngineConfig hnsw_cfg = flat_cfg;
  hnsw_cfg.index_type = SearchEngineConfig::IndexType::Hnsw;

  auto flat_or = SearchEngine::Open(flat_cfg);
  auto hnsw_or = SearchEngine::Open(hnsw_cfg);
  if (!flat_or.ok() || !hnsw_or.ok()) {
    std::cerr << "Failed to init engines\n";
    return 1;
  }
  auto flat = std::move(flat_or.value());
  auto hnsw = std::move(hnsw_or.value());

  std::mt19937 rng(static_cast<uint32_t>(seed));
  for (int i = 0; i < points; ++i) {
    auto vec = RandomVector(dim, rng);
    flat->Upsert("doc-" + std::to_string(i), VectorView{vec.data(), dim});
    hnsw->Upsert("doc-" + std::to_string(i), VectorView{vec.data(), dim});
  }

  std::vector<std::vector<float>> query_vectors;
  query_vectors.reserve(queries);
  for (int i = 0; i < queries; ++i) {
    query_vectors.push_back(RandomVector(dim, rng));
  }

  std::vector<double> recalls;
  recalls.reserve(queries);
  std::vector<double> latencies;
  size_t budget_ok = 0;
  for (const auto& q : query_vectors) {
    VectorView view{q.data(), dim};
    SearchEngine::QueryOptions options;
    options.topk = topk;
    auto truth = flat->Search(view, options);
    auto start = std::chrono::steady_clock::now();
    auto approx = hnsw->Search(view, options);
    auto end = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    latencies.push_back(ms);
    if (ms <= 10.0) {
      ++budget_ok;
    }
    if (truth.ok() && approx.ok()) {
      recalls.push_back(RecallAtK(truth.value(), approx.value()));
    }
  }

  double recall_avg = 0.0;
  for (double r : recalls) {
    recall_avg += r;
  }
  recall_avg = recalls.empty() ? 0.0 : recall_avg / recalls.size();
  double avg_ms = 0.0;
  for (double ms : latencies) {
    avg_ms += ms;
  }
  avg_ms = latencies.empty() ? 0.0 : avg_ms / latencies.size();
  double p50 = Percentile(latencies, 0.5);
  double p95 = Percentile(latencies, 0.95);
  double budget = latencies.empty() ? 0.0 : static_cast<double>(budget_ok) / latencies.size();

  std::cout << "bench_quality\n";
  std::cout << "recall@k=" << recall_avg << " avg_ms=" << avg_ms << " p50_ms=" << p50
            << " p95_ms=" << p95 << " budget_ok=" << budget << "\n";
  std::cout << "{\"recall_at_k\":" << recall_avg << ",\"avg_ms\":" << avg_ms
            << ",\"p50_ms\":" << p50 << ",\"p95_ms\":" << p95
            << ",\"budget_adherence\":" << budget << "}\n";
  return 0;
}
