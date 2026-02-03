#include "pomai_search/search_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "pomai_search/hash.h"
#include "core/index/flat_index.h"
#include "core/index/hnsw_index.h"
#include "pomai_search/logging.h"
#include "pomai_search/observability/metrics.h"
#include "pomai_search/scoring.h"
#include "core/serialize/snapshot.h"
#include "core/kernels/kernels.h"
#include "pomai_search/thread_pool.h"
#include "search_engine_impl.h"

namespace pomai_search {

namespace {

uint64_t HashFloatVector(VectorView view) {
  if (view.data == nullptr || view.dim <= 0) {
    return StableHash64("empty_vector");
  }
  return StableHash64Bytes(view.data, static_cast<size_t>(view.dim) * sizeof(float));
}

uint64_t HashQuery(uint64_t seed, std::string_view text, VectorView view, int topk,
                   const Filter& filter) {
  uint64_t hash = StableHash64Combine(seed, text);
  hash = StableHash64Combine(hash, std::to_string(topk));
  hash = StableHash64Combine(hash, filter.tag.value_or(""));
  hash = StableHash64Combine(hash, filter.source.value_or(""));
  hash = StableHash64Combine(hash, filter.lang.value_or(""));
  uint64_t vec_hash = HashFloatVector(view);
  hash ^= vec_hash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

std::string ComputeSnapshotId(const SearchEngineConfig& cfg, const SearchEngine::Stats& stats) {
  uint64_t hash = StableHash64Combine(StableHash64("snapshot"), std::to_string(cfg.contract_version));
  hash = StableHash64Combine(hash, std::to_string(cfg.global_seed));
  hash = StableHash64Combine(hash, std::to_string(stats.num_points));
  hash = StableHash64Combine(hash, std::to_string(stats.num_deleted));
  return std::to_string(hash);
}

}  // namespace

SearchEngine::SearchEngine() = default;

SearchEngine::~SearchEngine() {
  Close();
}

StatusOr<std::unique_ptr<SearchEngine>> SearchEngine::Open(const SearchEngineConfig& cfg) {
  auto engine = std::unique_ptr<SearchEngine>(new SearchEngine());
  Status status = engine->Initialize(cfg);
  if (!status.ok()) {
    return status;
  }
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

Status SearchEngine::Initialize(const SearchEngineConfig& cfg) {
  if (cfg.dim <= 0 || cfg.num_shards <= 0) {
    return Status(StatusCode::kInvalidArgument, "invalid config");
  }
  impl_ = std::make_unique<Impl>();
  impl_->cfg = cfg;
  impl_->shards.reserve(static_cast<size_t>(cfg.num_shards));
  size_t reserve_vectors = cfg.max_points_per_shard > 0 ? cfg.max_points_per_shard : 0;
  for (int i = 0; i < cfg.num_shards; ++i) {
    impl_->shards.emplace_back(
        std::make_unique<Shard>(cfg.dim, cfg.memory_alignment, reserve_vectors, cfg));
  }
  if (cfg.query_threads > 0) {
    impl_->query_pool = std::make_unique<ThreadPool>(static_cast<size_t>(cfg.query_threads));
  }
  if (cfg.ingest_threads > 0) {
    impl_->ingest_pool = std::make_unique<ThreadPool>(static_cast<size_t>(cfg.ingest_threads));
  }
  return Status::Ok();
}

Status SearchEngine::Upsert(std::string_view key, VectorView vec, Metadata meta,
                            std::optional<std::chrono::milliseconds> ttl,
                            std::optional<std::string> text) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  impl_->metrics.IncrementQueries();
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  auto expiry = ComputeExpiry(ttl);
  impl_->metrics.IncrementUpserts();
  return impl_->shards[shard_index]->Upsert(key, vec, std::move(meta), expiry, std::move(text));
}

Status SearchEngine::Delete(std::string_view key) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  impl_->metrics.IncrementQueries();
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  impl_->metrics.IncrementDeletes();
  return impl_->shards[shard_index]->Delete(key);
}

StatusOr<bool> SearchEngine::Exists(std::string_view key) const {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  return impl_->shards[shard_index]->Exists(key);
}

StatusOr<std::vector<float>> SearchEngine::GetVector(std::string_view key) const {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  return impl_->shards[shard_index]->GetVectorCopy(key);
}

StatusOr<std::vector<ResultItem>> SearchEngine::Search(VectorView q, QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  impl_->metrics.IncrementQueries();
  auto start = std::chrono::steady_clock::now();
  std::vector<std::vector<ResultItem>> shard_results(impl_->shards.size());
  std::vector<std::future<StatusOr<std::vector<ResultItem>>>> futures;
  if (impl_->query_pool) {
    futures.reserve(impl_->shards.size());
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto submitted = impl_->query_pool->Submit([
          shard = impl_->shards[i].get(), q, topk, filter = opt.filter]() {
        return shard->Search(q, topk, filter);
      });
      if (!submitted.ok()) {
        return submitted.status();
      }
      futures.push_back(std::move(submitted.value()));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      auto result = futures[i].get();
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  } else {
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto result = impl_->shards[i]->Search(q, topk, opt.filter);
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  }
  std::vector<ResultItem> merged;
  for (const auto& shard_result : shard_results) {
    merged.insert(merged.end(), shard_result.begin(), shard_result.end());
  }
  auto better = [](const ResultItem& a, const ResultItem& b) {
    float score_a = SanitizeScore(a.score);
    float score_b = SanitizeScore(b.score);
    if (score_a != score_b) {
      return score_a > score_b;
    }
    if (a.key != b.key) {
      return a.key < b.key;
    }
    return a.internal_id < b.internal_id;
  };
  if (static_cast<int>(merged.size()) > topk) {
    std::nth_element(merged.begin(), merged.begin() + topk, merged.end(), better);
    merged.resize(static_cast<size_t>(topk));
  }
  std::sort(merged.begin(), merged.end(), better);
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<std::vector<ResultItem>>(std::move(merged));
}

StatusOr<std::vector<ResultItem>> SearchEngine::SearchByKey(std::string_view key, QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  impl_->metrics.IncrementQueries();
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  auto vec_or = impl_->shards[shard_index]->GetVectorCopy(key);
  if (!vec_or.ok()) {
    return vec_or.status();
  }
  VectorView view{vec_or.value().data(), impl_->cfg.dim};
  return Search(view, opt);
}

StatusOr<std::vector<ResultItem>> SearchEngine::SearchHybrid(const HybridQuery& query,
                                                             QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  impl_->metrics.IncrementQueries();
  auto start = std::chrono::steady_clock::now();
  bool has_vector = query.vector_query.has_value();
  VectorView view{};
  if (has_vector) {
    if (static_cast<int>(query.vector_query->size()) != impl_->cfg.dim) {
      return Status(StatusCode::kInvalidArgument, "dimension mismatch");
    }
    view = VectorView{query.vector_query->data(), impl_->cfg.dim};
  }
  std::vector<ResultItem> merged;
  std::vector<std::vector<ResultItem>> shard_results(impl_->shards.size());
  if (impl_->query_pool) {
    std::vector<std::future<StatusOr<std::vector<ResultItem>>>> futures;
    futures.reserve(impl_->shards.size());
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto submitted = impl_->query_pool->Submit([
          shard = impl_->shards[i].get(), text = query.text_query.value_or(""), view, has_vector,
          alpha = query.alpha, topk, filter = opt.filter]() {
        return shard->SearchHybrid(text, view, has_vector, alpha, topk, filter);
      });
      if (!submitted.ok()) {
        return submitted.status();
      }
      futures.push_back(std::move(submitted.value()));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      auto result = futures[i].get();
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  } else {
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto result = impl_->shards[i]->SearchHybrid(query.text_query.value_or(""), view, has_vector,
                                                   query.alpha, topk, opt.filter);
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  }
  for (const auto& shard_result : shard_results) {
    merged.insert(merged.end(), shard_result.begin(), shard_result.end());
  }
  auto better = [](const ResultItem& a, const ResultItem& b) {
    float score_a = SanitizeScore(a.score);
    float score_b = SanitizeScore(b.score);
    if (score_a != score_b) {
      return score_a > score_b;
    }
    if (a.key != b.key) {
      return a.key < b.key;
    }
    return a.internal_id < b.internal_id;
  };
  if (static_cast<int>(merged.size()) > topk) {
    std::nth_element(merged.begin(), merged.begin() + topk, merged.end(), better);
    merged.resize(static_cast<size_t>(topk));
  }
  std::sort(merged.begin(), merged.end(), better);
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<std::vector<ResultItem>>(std::move(merged));
}

StatusOr<SearchResponse> SearchEngine::SearchWithExplain(VectorView q, QueryOptions opt,
                                                         QueryPolicy policy) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  if (topk <= 0) {
    return Status(StatusCode::kInvalidArgument, "topk must be positive");
  }
  QueryExplain explain;
  explain.global_seed = impl_->cfg.global_seed;
  explain.contract_version = impl_->cfg.contract_version;
  explain.snapshot_id = ComputeSnapshotId(impl_->cfg, GetStats());
  explain.query_id = HashQuery(StableHash64("query"), "", q, topk, opt.filter);

  policy.recall_bias = std::clamp(policy.recall_bias, 0.0f, 1.0f);
  int max_candidates = policy.max_candidates > 0 ? policy.max_candidates : topk * 2;
  max_candidates = std::max(max_candidates, topk);
  int stage1_candidates = std::min(max_candidates, topk * 2);
  int stage2_candidates = std::min(max_candidates, topk * 4);

  auto start = std::chrono::steady_clock::now();
  auto stage1_start = std::chrono::steady_clock::now();
  std::vector<std::vector<Shard::CandidateResult>> shard_results(impl_->shards.size());
  if (impl_->query_pool) {
    std::vector<std::future<StatusOr<std::vector<Shard::CandidateResult>>>> futures;
    futures.reserve(impl_->shards.size());
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto submitted = impl_->query_pool->Submit([
          shard = impl_->shards[i].get(), q, stage1_candidates, filter = opt.filter]() {
        return shard->SearchCandidates(q, stage1_candidates, filter);
      });
      if (!submitted.ok()) {
        return submitted.status();
      }
      futures.push_back(std::move(submitted.value()));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      auto result = futures[i].get();
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  } else {
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto result = impl_->shards[i]->SearchCandidates(q, stage1_candidates, opt.filter);
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  }
  std::vector<Shard::CandidateResult> merged_candidates;
  for (const auto& shard_result : shard_results) {
    merged_candidates.insert(merged_candidates.end(), shard_result.begin(), shard_result.end());
  }
  std::sort(merged_candidates.begin(), merged_candidates.end(),
            [](const Shard::CandidateResult& a, const Shard::CandidateResult& b) {
              float score_a = SanitizeScore(a.score);
              float score_b = SanitizeScore(b.score);
              if (score_a != score_b) {
                return score_a > score_b;
              }
              if (a.key != b.key) {
                return a.key < b.key;
              }
              return a.id < b.id;
            });
  if (static_cast<int>(merged_candidates.size()) > stage1_candidates) {
    merged_candidates.resize(static_cast<size_t>(stage1_candidates));
  }
  auto stage1_end = std::chrono::steady_clock::now();
  StageExplain stage1;
  stage1.name = "vector_stage_1";
  stage1.index = impl_->cfg.index_type == SearchEngineConfig::IndexType::Hnsw ? "hnsw" : "flat";
  stage1.ef_search = impl_->cfg.hnsw_ef_search;
  stage1.max_candidates = stage1_candidates;
  stage1.time_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stage1_end - stage1_start)
          .count();
  stage1.candidates_out = static_cast<int>(merged_candidates.size());
  explain.execution_plan.push_back(stage1);

  bool run_stage2 = policy.recall_bias > 0.5f && stage2_candidates > stage1_candidates;
  if (policy.max_latency_ms > 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        stage1_end - start);
    if (elapsed.count() >= static_cast<double>(policy.max_latency_ms)) {
      run_stage2 = false;
    }
  }

  if (run_stage2) {
    auto stage2_start = std::chrono::steady_clock::now();
    std::vector<std::vector<Shard::CandidateResult>> stage2_shards(impl_->shards.size());
    if (impl_->query_pool) {
      std::vector<std::future<StatusOr<std::vector<Shard::CandidateResult>>>> futures;
      futures.reserve(impl_->shards.size());
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto submitted = impl_->query_pool->Submit([
            shard = impl_->shards[i].get(), q, stage2_candidates, filter = opt.filter]() {
          return shard->SearchCandidates(q, stage2_candidates, filter);
        });
        if (!submitted.ok()) {
          return submitted.status();
        }
        futures.push_back(std::move(submitted.value()));
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (!result.ok()) {
          return result.status();
        }
        stage2_shards[i] = std::move(result.value());
      }
    } else {
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto result = impl_->shards[i]->SearchCandidates(q, stage2_candidates, opt.filter);
        if (!result.ok()) {
          return result.status();
        }
        stage2_shards[i] = std::move(result.value());
      }
    }
    merged_candidates.clear();
    for (const auto& shard_result : stage2_shards) {
      merged_candidates.insert(merged_candidates.end(), shard_result.begin(), shard_result.end());
    }
    std::sort(merged_candidates.begin(), merged_candidates.end(),
              [](const Shard::CandidateResult& a, const Shard::CandidateResult& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.id < b.id;
              });
    if (static_cast<int>(merged_candidates.size()) > stage2_candidates) {
      merged_candidates.resize(static_cast<size_t>(stage2_candidates));
    }
    auto stage2_end = std::chrono::steady_clock::now();
    StageExplain stage2;
    stage2.name = "vector_stage_2";
    stage2.index = impl_->cfg.index_type == SearchEngineConfig::IndexType::Hnsw ? "hnsw" : "flat";
    stage2.ef_search = impl_->cfg.hnsw_ef_search + static_cast<int>(policy.recall_bias * 20.0f);
    stage2.max_candidates = stage2_candidates;
    stage2.time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                         stage2_end - stage2_start)
                         .count();
    stage2.candidates_out = static_cast<int>(merged_candidates.size());
    explain.execution_plan.push_back(stage2);
  }

  SearchResponse response;
  response.explain = std::move(explain);
  for (size_t i = 0; i < merged_candidates.size(); ++i) {
    const auto& cand = merged_candidates[i];
    ResultItem item;
    item.key = cand.key;
    item.score = SanitizeScore(cand.score);
    item.meta = cand.meta;
    item.internal_id = cand.id;
    response.results.push_back(std::move(item));
    ResultExplain detail;
    detail.vector_score_raw = SanitizeScore(cand.score);
    detail.vector_score_normed = detail.vector_score_raw;
    detail.keyword_score_raw = 0.0f;
    detail.keyword_score_normed = 0.0f;
    detail.fusion_method = "weighted_sum";
    detail.final_score = detail.vector_score_raw;
    detail.rank_before_fusion = static_cast<int>(i) + 1;
    detail.rank_after_fusion = static_cast<int>(i) + 1;
    response.explain.result_details.push_back(detail);
    if (static_cast<int>(response.results.size()) >= topk) {
      break;
    }
  }
  if (static_cast<int>(response.results.size()) > topk) {
    response.results.resize(static_cast<size_t>(topk));
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<SearchResponse>(std::move(response));
}

StatusOr<SearchResponse> SearchEngine::SearchHybridWithExplain(const HybridQuery& query,
                                                               QueryOptions opt,
                                                               QueryPolicy policy) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  if (topk <= 0) {
    return Status(StatusCode::kInvalidArgument, "topk must be positive");
  }
  bool has_vector = query.vector_query.has_value();
  VectorView view{};
  if (has_vector) {
    if (static_cast<int>(query.vector_query->size()) != impl_->cfg.dim) {
      return Status(StatusCode::kInvalidArgument, "dimension mismatch");
    }
    view = VectorView{query.vector_query->data(), impl_->cfg.dim};
  }

  QueryExplain explain;
  explain.global_seed = impl_->cfg.global_seed;
  explain.contract_version = impl_->cfg.contract_version;
  explain.snapshot_id = ComputeSnapshotId(impl_->cfg, GetStats());
  explain.query_id =
      HashQuery(StableHash64("query"), query.text_query.value_or(""), view, topk, opt.filter);

  int max_candidates = policy.max_candidates > 0 ? policy.max_candidates : topk * 2;
  max_candidates = std::max(max_candidates, topk);
  int stage1_candidates = std::min(max_candidates, topk * 2);

  std::vector<Shard::CandidateResult> vector_candidates;
  std::vector<Shard::CandidateResult> keyword_candidates;
  auto start = std::chrono::steady_clock::now();
  if (has_vector) {
    auto stage_start = std::chrono::steady_clock::now();
    std::vector<std::vector<Shard::CandidateResult>> shard_results(impl_->shards.size());
    if (impl_->query_pool) {
      std::vector<std::future<StatusOr<std::vector<Shard::CandidateResult>>>> futures;
      futures.reserve(impl_->shards.size());
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto submitted = impl_->query_pool->Submit([
            shard = impl_->shards[i].get(), view, stage1_candidates, filter = opt.filter]() {
          return shard->SearchCandidates(view, stage1_candidates, filter);
        });
        if (!submitted.ok()) {
          return submitted.status();
        }
        futures.push_back(std::move(submitted.value()));
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    } else {
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto result = impl_->shards[i]->SearchCandidates(view, stage1_candidates, opt.filter);
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    }
    for (const auto& shard_result : shard_results) {
      vector_candidates.insert(vector_candidates.end(), shard_result.begin(), shard_result.end());
    }
    std::sort(vector_candidates.begin(), vector_candidates.end(),
              [](const Shard::CandidateResult& a, const Shard::CandidateResult& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.id < b.id;
              });
    if (static_cast<int>(vector_candidates.size()) > stage1_candidates) {
      vector_candidates.resize(static_cast<size_t>(stage1_candidates));
    }
    auto stage_end = std::chrono::steady_clock::now();
    StageExplain stage;
    stage.name = "vector_stage_1";
    stage.index = impl_->cfg.index_type == SearchEngineConfig::IndexType::Hnsw ? "hnsw" : "flat";
    stage.ef_search = impl_->cfg.hnsw_ef_search;
    stage.max_candidates = stage1_candidates;
    stage.time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                        stage_end - stage_start)
                        .count();
    stage.candidates_out = static_cast<int>(vector_candidates.size());
    explain.execution_plan.push_back(stage);
  }

  if (query.text_query && !query.text_query->empty()) {
    auto stage_start = std::chrono::steady_clock::now();
    std::vector<std::vector<Shard::CandidateResult>> shard_results(impl_->shards.size());
    if (impl_->query_pool) {
      std::vector<std::future<StatusOr<std::vector<Shard::CandidateResult>>>> futures;
      futures.reserve(impl_->shards.size());
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto submitted = impl_->query_pool->Submit([
            shard = impl_->shards[i].get(), text = query.text_query.value_or(""),
            stage1_candidates, filter = opt.filter]() {
          return shard->KeywordCandidatesWithDocs(text, stage1_candidates, filter);
        });
        if (!submitted.ok()) {
          return submitted.status();
        }
        futures.push_back(std::move(submitted.value()));
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    } else {
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto result =
            impl_->shards[i]->KeywordCandidatesWithDocs(query.text_query.value_or(""),
                                                        stage1_candidates, opt.filter);
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    }
    for (const auto& shard_result : shard_results) {
      keyword_candidates.insert(keyword_candidates.end(), shard_result.begin(), shard_result.end());
    }
    std::sort(keyword_candidates.begin(), keyword_candidates.end(),
              [](const Shard::CandidateResult& a, const Shard::CandidateResult& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.id < b.id;
              });
    if (static_cast<int>(keyword_candidates.size()) > stage1_candidates) {
      keyword_candidates.resize(static_cast<size_t>(stage1_candidates));
    }
    auto stage_end = std::chrono::steady_clock::now();
    StageExplain stage;
    stage.name = "keyword_stage";
    stage.index = "keyword";
    stage.max_candidates = stage1_candidates;
    stage.time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                        stage_end - stage_start)
                        .count();
    stage.candidates_out = static_cast<int>(keyword_candidates.size());
    explain.execution_plan.push_back(stage);
  }

  std::unordered_map<uint32_t, size_t> id_to_index;
  std::vector<Shard::CandidateResult> combined;
  combined.reserve(vector_candidates.size() + keyword_candidates.size());
  auto add_candidate = [&](const Shard::CandidateResult& candidate, bool is_vector) {
    auto it = id_to_index.find(candidate.id);
    if (it == id_to_index.end()) {
      id_to_index.emplace(candidate.id, combined.size());
      combined.push_back(candidate);
    } else if (!is_vector) {
      combined[it->second].meta = candidate.meta;
      combined[it->second].key = candidate.key;
    }
  };
  for (const auto& cand : vector_candidates) {
    add_candidate(cand, true);
  }
  for (const auto& cand : keyword_candidates) {
    add_candidate(cand, false);
  }

  std::unordered_map<uint32_t, int> vector_rank;
  for (size_t i = 0; i < vector_candidates.size(); ++i) {
    vector_rank[vector_candidates[i].id] = static_cast<int>(i) + 1;
  }
  std::unordered_map<uint32_t, int> keyword_rank;
  for (size_t i = 0; i < keyword_candidates.size(); ++i) {
    keyword_rank[keyword_candidates[i].id] = static_cast<int>(i) + 1;
  }

  struct ScoredResult {
    Shard::CandidateResult base;
    float vector_score = 0.0f;
    float keyword_score = 0.0f;
    float final_score = 0.0f;
    int rank_before = 0;
  };

  std::vector<ScoredResult> scored;
  scored.reserve(combined.size());
  for (const auto& cand : combined) {
    ScoredResult item;
    item.base = cand;
    item.vector_score = 0.0f;
    item.keyword_score = 0.0f;
    auto vec_it = vector_rank.find(cand.id);
    if (vec_it != vector_rank.end()) {
      item.vector_score = vector_candidates[static_cast<size_t>(vec_it->second - 1)].score;
      item.rank_before = vec_it->second;
    }
    auto key_it = keyword_rank.find(cand.id);
    if (key_it != keyword_rank.end()) {
      item.keyword_score = keyword_candidates[static_cast<size_t>(key_it->second - 1)].score;
    }
    if (policy.fusion_method == FusionMethod::Rrf) {
      int vec_rank = vec_it != vector_rank.end() ? vec_it->second : static_cast<int>(combined.size()) + 1;
      int key_rank = key_it != keyword_rank.end() ? key_it->second : static_cast<int>(combined.size()) + 1;
      float rrf_k = 60.0f;
      item.final_score = (1.0f / (rrf_k + vec_rank)) + (1.0f / (rrf_k + key_rank));
    } else {
      float alpha = query.alpha;
      item.final_score = alpha * item.vector_score + (1.0f - alpha) * item.keyword_score;
    }
    scored.push_back(std::move(item));
  }

  std::sort(scored.begin(), scored.end(), [](const ScoredResult& a, const ScoredResult& b) {
    float score_a = SanitizeScore(a.final_score);
    float score_b = SanitizeScore(b.final_score);
    if (score_a != score_b) {
      return score_a > score_b;
    }
    if (a.base.key != b.base.key) {
      return a.base.key < b.base.key;
    }
    return a.base.id < b.base.id;
  });

  SearchResponse response;
  response.explain = std::move(explain);
  std::string fusion_method =
      policy.fusion_method == FusionMethod::Rrf ? "rrf" : "weighted_sum";
  for (size_t i = 0; i < scored.size(); ++i) {
    const auto& item = scored[i];
    ResultItem result;
    result.key = item.base.key;
    result.score = SanitizeScore(item.final_score);
    result.meta = item.base.meta;
    result.internal_id = item.base.id;
    response.results.push_back(std::move(result));

    ResultExplain detail;
    detail.vector_score_raw = SanitizeScore(item.vector_score);
    detail.vector_score_normed = detail.vector_score_raw;
    detail.keyword_score_raw = SanitizeScore(item.keyword_score);
    detail.keyword_score_normed = detail.keyword_score_raw;
    detail.fusion_method = fusion_method;
    detail.final_score = SanitizeScore(item.final_score);
    detail.rank_before_fusion = item.rank_before;
    detail.rank_after_fusion = static_cast<int>(i) + 1;
    response.explain.result_details.push_back(std::move(detail));
    if (static_cast<int>(response.results.size()) >= topk) {
      break;
    }
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<SearchResponse>(std::move(response));
}

SearchEngine::Stats SearchEngine::GetStats() const {
  Stats stats;
  if (!impl_) {
    return stats;
  }
  for (const auto& shard : impl_->shards) {
    auto shard_stats = shard->GetStats();
    stats.num_points += shard_stats.num_points;
    stats.num_deleted += shard_stats.num_deleted;
  }
  return stats;
}

std::string SearchEngine::MetricsJson() const {
  if (!impl_) {
    return "{}";
  }
  return impl_->metrics.ToJson();
}

Status SearchEngine::Close() {
  if (!impl_) {
    return Status::Ok();
  }
  impl_->query_pool.reset();
  impl_->ingest_pool.reset();
  impl_.reset();
  return Status::Ok();
}

}  // namespace pomai_search
