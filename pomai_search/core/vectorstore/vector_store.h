#pragma once

#include <cstddef>
#include <cstdio>
#include <shared_mutex>
#include <vector>

#include "pomai_search/aligned_allocator.h"
#include "pomai_search/status.h"

namespace pomai_search {

// VectorStore is the authoritative storage for vector data.
// It guarantees a single copy of each vector in memory.
// All indexes (Flat, HNSW) must refer to vectors by offset/pointer within this store.
class VectorStore {
 public:
  struct Stats {
    size_t live_vectors = 0;
    size_t total_vectors = 0;
    size_t free_vectors = 0;
    size_t blocks = 0;
    size_t bytes_allocated = 0;
  };

  class ReadGuard {
   public:
    explicit ReadGuard(const VectorStore& store) : lock_(store.mutex_) {}

   private:
    friend class VectorStore;
    std::shared_lock<std::shared_mutex> lock_;
  };

  class WriteGuard {
   public:
    explicit WriteGuard(VectorStore& store) : lock_(store.mutex_) {}

   private:
    friend class VectorStore;
    std::unique_lock<std::shared_mutex> lock_;
  };

  VectorStore(int dim, size_t alignment, size_t reserve_vectors = 0,
              size_t block_vectors = 4096);

  // Copy semantics: deleted to prevent accidental copies.
  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;

  // Move semantics: allowed.
  VectorStore(VectorStore&&) = default;
  VectorStore& operator=(VectorStore&&) = default;

  // Allocates a slot and writes vector data.
  // Returns the slot index (vector id).
  size_t Insert(const float* data, WriteGuard& guard);

  // Overwrites an existing slot.
  void Update(size_t slot, const float* data, WriteGuard& guard);

  // Releases a slot for reuse.
  void Release(size_t slot, WriteGuard& guard);

  ReadGuard AcquireRead() const { return ReadGuard(*this); }
  WriteGuard AcquireWrite() { return WriteGuard(*this); }

  // Returns a pointer to the vector at the given slot.
  const float* Get(size_t slot, const ReadGuard& guard) const;

  // Returns the raw data pointer (for batch access/simd).
  size_t alignment() const { return alignment_; }
  size_t count() const { return live_count_; }
  int dim() const { return dim_; }

  Stats GetStats() const;
  
  Status Save(std::FILE* out) const;
  Status Load(std::FILE* in);

 private:
  void EnsureBlock(size_t block_index);
  const float* GetUnsafe(size_t slot) const;
  float* GetUnsafe(size_t slot);

  int dim_;
  size_t alignment_;
  size_t block_vectors_;
  size_t block_floats_;
  mutable std::shared_mutex mutex_;
  std::vector<std::vector<float, AlignedAllocator<float>>> blocks_;
  std::vector<size_t> free_list_;
  std::vector<uint8_t> live_;
  size_t live_count_ = 0;
};

}  // namespace pomai_search
