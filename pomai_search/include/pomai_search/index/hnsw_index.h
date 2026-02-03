#pragma once

#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "pomai_search/index/index.h"
#include "pomai_search/simd/kernels.h"
#include "pomai_search/vector_arena.h"

namespace pomai_search {

class HnswIndex : public Index {
 public:
  HnswIndex(int dim, SearchEngineConfig::Similarity similarity, DotFunc dot_func, size_t alignment,
            size_t reserve_vectors, size_t max_points, int m, int ef_construction, int ef_search,
            uint32_t seed);

  Status Upsert(uint32_t id, VectorView v) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override {}
  IndexStats GetStats() const override;

 private:
  struct Node {
    size_t offset = 0;
    float norm = 0.0f;
    bool deleted = false;
    int level = 0;
    std::vector<std::vector<uint32_t>> neighbors;
  };

  float ComputeNorm(const float* data) const;
  float Score(VectorView q, const Node& node) const;
  int RandomLevel();
  std::vector<uint32_t> SearchLayer(VectorView q, uint32_t entry, int level, int ef) const;
  void ConnectNewNode(uint32_t node_id, int level, const std::vector<uint32_t>& candidates);

  int dim_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;
  int m_;
  int ef_construction_;
  int ef_search_;

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, size_t> id_to_index_;
  std::vector<uint32_t> ids_;
  std::vector<Node> nodes_;
  VectorArena arena_;
  uint32_t entry_id_ = 0;
  int max_level_ = -1;
  mutable std::mt19937 rng_;
};

}  // namespace pomai_search
