#include "pomai_search/search_engine.h"
#include "tests/test_framework.h"

#include <cmath>

namespace pomai_search::test {

POMAI_TEST(ExplainDeterministic) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.global_seed = 7;
  cfg.contract_version = 2;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  engine->Upsert("doc-a", VectorView{a, 2});
  engine->Upsert("doc-b", VectorView{b, 2});
  float q[2] = {1.0f, 0.0f};
  SearchEngine::QueryOptions opt;
  opt.topk = 2;
  QueryPolicy policy;
  policy.max_latency_ms = 50;
  policy.max_candidates = 4;
  auto first = engine->SearchWithExplain(VectorView{q, 2}, opt, policy);
  EXPECT_TRUE(first.ok());
  auto second = engine->SearchWithExplain(VectorView{q, 2}, opt, policy);
  EXPECT_TRUE(second.ok());
  EXPECT_EQ(first.value().results.size(), second.value().results.size());
  EXPECT_EQ(first.value().explain.query_id, second.value().explain.query_id);
  EXPECT_EQ(first.value().explain.execution_plan.size(),
            second.value().explain.execution_plan.size());
  EXPECT_EQ(first.value().results.front().key, second.value().results.front().key);
  return true;
}

POMAI_TEST(ExplainFusionChangesWithAlpha) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.global_seed = 11;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  engine->Upsert("doc-a", VectorView{a, 2}, {{"tag", "alpha"}}, std::nullopt, "alpha");
  engine->Upsert("doc-b", VectorView{b, 2}, {{"tag", "beta"}}, std::nullopt, "beta");

  SearchEngine::HybridQuery query;
  query.text_query = "alpha";
  query.vector_query = std::vector<float>{1.0f, 0.0f};
  query.alpha = 0.2f;
  SearchEngine::QueryOptions opt;
  opt.topk = 1;
  QueryPolicy policy;
  policy.max_candidates = 4;
  policy.fusion_method = FusionMethod::WeightedSum;

  auto low_alpha = engine->SearchHybridWithExplain(query, opt, policy);
  EXPECT_TRUE(low_alpha.ok());
  query.alpha = 0.8f;
  auto high_alpha = engine->SearchHybridWithExplain(query, opt, policy);
  EXPECT_TRUE(high_alpha.ok());
  float low_score = low_alpha.value().explain.result_details.front().final_score;
  float high_score = high_alpha.value().explain.result_details.front().final_score;
  EXPECT_TRUE(std::fabs(low_score - high_score) > 1e-3f);
  return true;
}

}  // namespace pomai_search::test
