#pragma once

#include <cstddef>
#include <vector>

#include "aligned_allocator.h"

namespace pomai_search {

class VectorArena {
 public:
  VectorArena(int dim, size_t alignment, size_t reserve_vectors = 0);

  size_t Append(const float* data);
  const float* Get(size_t offset) const;
  size_t size() const { return count_; }

 private:
  int dim_;
  std::vector<float, AlignedAllocator<float>> data_;
  size_t count_ = 0;
};

}  // namespace pomai_search
