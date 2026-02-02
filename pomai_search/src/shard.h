#pragma once

#include <chrono>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai_search/status.h"
#include "pomai_search/types.h"
#include "simd/kernels.h"
#include "vector_arena.h"

namespace pomai_search {

struct ShardItem {
  std::string key;
  Metadata meta;
  size_t offset = 0;
  float norm = 0.0f;
  bool deleted = false;
  std::optional<std::chrono::steady_clock::time_point> expiry;
};

class Shard {
 public:
  Shard(int dim, size_t alignment, size_t reserve_vectors, SearchEngineConfig::Similarity similarity,
        DotFunc dot_func, size_t max_points);

  Status Upsert(std::string_view key, VectorView vec, Metadata meta,
                std::optional<std::chrono::steady_clock::time_point> expiry);
  Status Delete(std::string_view key);
  StatusOr<bool> Exists(std::string_view key) const;

  StatusOr<std::vector<ResultItem>> Search(VectorView query, int topk, const Metadata& filter,
                                           float query_norm) const;
  StatusOr<std::vector<float>> GetVectorCopy(std::string_view key) const;

  uint64_t num_points() const;
  uint64_t num_deleted() const;

 private:
  float ComputeNorm(const float* data) const;

  int dim_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, size_t> index_;
  std::vector<ShardItem> items_;
  VectorArena arena_;
};

}  // namespace pomai_search
