#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "pomai_search/index/index.h"
#include "pomai_search/simd/kernels.h"
#include "pomai_search/vector_arena.h"

namespace pomai_search {

class FlatIndex : public Index {
 public:
  FlatIndex(int dim, SearchEngineConfig::Similarity similarity, DotFunc dot_func, size_t alignment,
            size_t reserve_vectors, size_t max_points);

  Status Upsert(uint32_t id, VectorView v) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override {}
  IndexStats GetStats() const override;

 private:
  float ComputeNorm(const float* data) const;

  int dim_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;

  struct Item {
    size_t offset = 0;
    float norm = 0.0f;
    bool deleted = false;
  };

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, size_t> id_to_index_;
  std::vector<uint32_t> ids_;
  std::vector<Item> items_;
  VectorArena arena_;
};

}  // namespace pomai_search
