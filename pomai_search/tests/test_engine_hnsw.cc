#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace pomai_search::test {

POMAI_TEST(EngineHnswSearch) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.index_type = SearchEngineConfig::IndexType::Hnsw;
  cfg.hnsw_m = 8;
  cfg.hnsw_ef_construction = 50;
  cfg.hnsw_ef_search = 20;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  float c[2] = {0.9f, 0.1f};
  engine->Upsert("a", VectorView{a, 2});
  engine->Upsert("b", VectorView{b, 2});
  engine->Upsert("c", VectorView{c, 2});
  float qv[2] = {1.0f, 0.0f};
  auto results = engine->Search(VectorView{qv, 2});
  EXPECT_TRUE(results.ok());
  EXPECT_EQ(results.value().front().key, "a");
  return true;
}

POMAI_TEST(HnswRecallGuardrail) {
  int n = 2000;
  int dim = 16;
  int topk = 10;
  int queries = 50;
  
  // Ground Truth Engine (Flat)
  SearchEngineConfig flat_cfg;
  flat_cfg.dim = dim;
  flat_cfg.index_type = SearchEngineConfig::IndexType::Flat;
  auto flat_engine = std::move(SearchEngine::Open(flat_cfg).value());
  
  // HNSW Engine
  SearchEngineConfig hnsw_cfg;
  hnsw_cfg.dim = dim;
  hnsw_cfg.index_type = SearchEngineConfig::IndexType::Hnsw;
  hnsw_cfg.hnsw_m = 16;
  hnsw_cfg.hnsw_ef_construction = 200;
  hnsw_cfg.hnsw_ef_search = 50; // High for guardrail
  auto hnsw_engine = std::move(SearchEngine::Open(hnsw_cfg).value());
  
  // Data Generation
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data(n, std::vector<float>(dim));
  
  for(int i=0; i<n; ++i) {
      for(int d=0; d<dim; ++d) data[i][d] = dist(rng);
      std::string key = std::to_string(i);
      VectorView v{data[i].data(), dim};
      flat_engine->Upsert(key, v);
      hnsw_engine->Upsert(key, v);
  }
  
  // Verify Recall
  double total_recall = 0.0;
  std::vector<float> q(dim);
  
  for(int i=0; i<queries; ++i) {
      for(int d=0; d<dim; ++d) q[d] = dist(rng);
      VectorView qv{q.data(), dim};
      
      SearchEngine::QueryOptions opts;
      opts.topk = topk;
      
      auto flat_res = flat_engine->Search(qv, opts).value();
      auto hnsw_res = hnsw_engine->Search(qv, opts).value();
      
      std::set<std::string> gt;
      for(const auto& c : flat_res) gt.insert(c.key);
      
      int hits = 0;
      for(const auto& c : hnsw_res) {
          if (gt.count(c.key)) hits++;
      }
      
      if (!flat_res.empty()) {
          total_recall += static_cast<double>(hits) / flat_res.size();
      }
  }
  
  double avg_recall = total_recall / queries;
  // EXPECT_GT(avg_recall, 0.95);
  // With extend_candidates and ef=50, should be near 1.0 for small N
  // But due to random data, HNSW is very good.
  // 0.95 is safe buffer.
  if (avg_recall <= 0.95) {
      std::cerr << "Recall Guardrail Failed! Avg Recall: " << avg_recall << "\n";
      return false;
  }
  return true;
}

}  // namespace pomai_search::test
