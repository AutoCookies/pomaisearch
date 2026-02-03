#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

#include "pomai_search/status.h"
#include "pomai_search/types.h"

namespace pomai_search {

struct Candidate {
  uint32_t id = 0;
  float score = 0.0f;
};

struct IndexStats {
  uint64_t num_points = 0;
  uint64_t num_deleted = 0;
};

class Index {
 public:
  virtual ~Index() = default;
  virtual Status Upsert(uint32_t id, size_t offset, float norm) = 0;
  virtual Status Delete(uint32_t id) = 0;
  virtual StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const = 0;
  virtual void Compact() = 0;
  virtual IndexStats GetStats() const = 0;
  virtual Status Save(std::FILE* out) const = 0;
  virtual Status Load(std::FILE* in) = 0;
};

}  // namespace pomai_search
