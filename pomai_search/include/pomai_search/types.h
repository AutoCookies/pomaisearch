#pragma once

#include <chrono>
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
  enum class IndexType { Flat, Hnsw } index_type = IndexType::Flat;
  int hnsw_m = 16;
  int hnsw_ef_construction = 200;
  int hnsw_ef_search = 50;
  uint32_t hnsw_seed = 42;
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

inline std::optional<std::chrono::steady_clock::time_point> ComputeExpiry(
    const std::optional<std::chrono::milliseconds>& ttl) {
  if (!ttl.has_value()) {
    return std::nullopt;
  }
  return std::chrono::steady_clock::now() + ttl.value();
}

inline bool IsExpired(const std::optional<std::chrono::steady_clock::time_point>& expiry) {
  if (!expiry.has_value()) {
    return false;
  }
  return std::chrono::steady_clock::now() >= expiry.value();
}

}  // namespace pomai_search
