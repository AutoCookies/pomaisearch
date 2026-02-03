#pragma once

#include <cstddef>
#include <vector>

#include "pomai_search/aligned_allocator.h"

namespace pomai_search {

// VectorStore is the authoritative storage for vector data.
// It guarantees a single copy of each vector in memory.
// All indexes (Flat, HNSW) must refer to vectors by offset/pointer within this store.
class VectorStore {
 public:
  VectorStore(int dim, size_t alignment, size_t reserve_vectors = 0);

  // Copy semantics: deleted to prevent accidental copies.
  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;

  // Move semantics: allowed.
  VectorStore(VectorStore&&) = default;
  VectorStore& operator=(VectorStore&&) = default;

  // Appends a vector to the store.
  // Returns the offset (in floats) where the vector starts.
  size_t Append(const float* data);

  // Returns a pointer to the vector at the given offset.
  const float* Get(size_t offset) const;

  // Returns the raw data pointer (for batch access/simd).
  const float* data() const { return data_.data(); }

  size_t alignment() const { return data_.get_allocator().alignment(); }
  size_t count() const { return count_; }
  int dim() const { return dim_; }

 private:
  int dim_;
  std::vector<float, AlignedAllocator<float>> data_;
  size_t count_ = 0;
};

}  // namespace pomai_search
