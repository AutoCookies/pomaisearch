#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pomai_search {

using Metadata = std::unordered_map<std::string, std::string>;

struct VectorView {
  const float* data = nullptr;
  int dim = 0;
};

struct Filter {
  std::optional<std::string> tag;
  std::optional<std::string> source;
  std::optional<std::string> lang;
};

struct ResultItem {
  std::string key;
  float score = 0.0f;
  Metadata meta;
  uint32_t internal_id = 0;
};

struct SearchEngineConfig {
  int dim = 0;
  int num_shards = 1;
  enum class Similarity { Dot, Cosine } similarity = Similarity::Dot;
  int topk_default = 10;
  size_t max_points_per_shard = 0;
  bool enable_avx2 = true;
  int query_threads = 0;
  int ingest_threads = 0;
  size_t memory_alignment = 32;
  enum class IndexType { Flat, Hnsw, IvfFlat, IvfSq8 } index_type = IndexType::Flat;
  int hnsw_m = 16;
  int hnsw_ef_construction = 200;
  int hnsw_ef_search = 50;
  uint32_t hnsw_seed = 42;
  int ivf_nlist = 100;
  int ivf_nprobe = 10;
  int max_candidates = 2000;
  int rerank_factor = 4;
  uint64_t global_seed = 0;
  uint32_t contract_version = 3;
};

enum class FusionMethod { WeightedSum, Rrf };

struct QueryPolicy {
  int max_latency_ms = 0;
  int max_candidates = 0;
  float recall_bias = 0.5f;
  FusionMethod fusion_method = FusionMethod::WeightedSum;
};

struct StageExplain {
  std::string name;
  std::string index;
  int ef_search = 0;
  int nprobe = 0;
  int max_candidates = 0;
  double time_ms = 0.0;
  int candidates_out = 0;
};

struct ResultExplain {
  float vector_score_raw = 0.0f;
  float vector_score_normed = 0.0f;
  float keyword_score_raw = 0.0f;
  float keyword_score_normed = 0.0f;
  std::string fusion_method;
  float final_score = 0.0f;
  int rank_before_fusion = 0;
  int rank_after_fusion = 0;
};

struct QueryExplain {
  uint64_t query_id = 0;
  uint64_t global_seed = 0;
  uint32_t contract_version = 0;
  std::string snapshot_id;
  std::vector<StageExplain> execution_plan;
  std::vector<ResultExplain> result_details;
};

struct SearchResponse {
  std::vector<ResultItem> results;
  QueryExplain explain;
};

inline bool MetadataFilterMatch(const Filter& filter, const Metadata& candidate) {
  if (!filter.tag && !filter.source && !filter.lang) {
    return true;
  }
  if (filter.tag) {
    auto it = candidate.find("tag");
    if (it == candidate.end() || it->second != *filter.tag) {
      return false;
    }
  }
  if (filter.source) {
    auto it = candidate.find("source");
    if (it == candidate.end() || it->second != *filter.source) {
      return false;
    }
  }
  if (filter.lang) {
    auto it = candidate.find("lang");
    if (it == candidate.end() || it->second != *filter.lang) {
      return false;
    }
  }
  return true;
}

inline std::optional<std::chrono::system_clock::time_point> ComputeExpiry(
    const std::optional<std::chrono::milliseconds>& ttl) {
  if (!ttl.has_value()) {
    return std::nullopt;
  }
  return std::chrono::system_clock::now() + ttl.value();
}

inline bool IsExpired(const std::optional<std::chrono::system_clock::time_point>& expiry) {
  if (!expiry.has_value()) {
    return false;
  }
  return std::chrono::system_clock::now() >= expiry.value();
}

}  // namespace pomai_search
