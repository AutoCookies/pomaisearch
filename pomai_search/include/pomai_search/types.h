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

struct ResultItem {
  std::string key;
  float score = 0.0f;
  Metadata meta;
};

struct SearchEngineConfig {
  int dim = 0;
  int num_shards = 0;
  enum class Similarity { Dot, Cosine } similarity = Similarity::Dot;
  int topk_default = 10;
  size_t max_points_per_shard = 0;
  bool enable_avx2 = true;
  int query_threads = 0;
  int ingest_threads = 0;
  size_t memory_alignment = 32;
};

inline bool MetadataFilterMatch(const Metadata& filter, const Metadata& candidate) {
  if (filter.empty()) {
    return true;
  }
  for (const auto& pair : filter) {
    if (pair.first != "tag" && pair.first != "source" && pair.first != "lang") {
      continue;
    }
    auto it = candidate.find(pair.first);
    if (it == candidate.end() || it->second != pair.second) {
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
