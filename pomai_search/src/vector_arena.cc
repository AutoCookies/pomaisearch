#include "vector_arena.h"

#include <cstring>

namespace pomai_search {

VectorArena::VectorArena(int dim, size_t alignment, size_t reserve_vectors)
    : dim_(dim), data_(AlignedAllocator<float>(alignment)) {
  if (reserve_vectors > 0) {
    data_.reserve(reserve_vectors * static_cast<size_t>(dim_));
  }
}

size_t VectorArena::Append(const float* data) {
  size_t offset = count_ * static_cast<size_t>(dim_);
  data_.insert(data_.end(), data, data + dim_);
  ++count_;
  return offset;
}

const float* VectorArena::Get(size_t offset) const {
  return data_.data() + offset;
}

}  // namespace pomai_search
