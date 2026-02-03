#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "core/index/index.h"
#include "core/kernels/kernels.h"
#include "core/vectorstore/vector_store.h"

namespace pomai_search {

class FlatIndex : public Index {
 public:
  FlatIndex(const VectorStore* store, int dim, SearchEngineConfig::Similarity similarity,
            DotFunc dot_func, size_t max_points);

  Status Upsert(uint32_t id, size_t offset, float norm) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override {}
  IndexStats GetStats() const override;

 private:
  const VectorStore* store_;
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
};

}  // namespace pomai_search
