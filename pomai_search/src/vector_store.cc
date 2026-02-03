#include "pomai_search/vector_store.h"

#include <algorithm>

namespace pomai_search {

VectorStore::VectorStore(int dim, size_t alignment, size_t reserve_vectors)
    : dim_(dim), data_(AlignedAllocator<float>(alignment)) {
  if (reserve_vectors > 0) {
    data_.reserve(reserve_vectors * static_cast<size_t>(dim));
  }
}

size_t VectorStore::Append(const float* data) {
  size_t offset = data_.size();
  data_.insert(data_.end(), data, data + dim_);
  ++count_;
  return offset;
}

const float* VectorStore::Get(size_t offset) const {
  return data_.data() + offset;
}

}  // namespace pomai_search
