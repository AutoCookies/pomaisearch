#include "core/vectorstore/vector_store.h"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace pomai_search {

VectorStore::VectorStore(int dim, size_t alignment, size_t reserve_vectors,
                         size_t block_vectors)
    : dim_(dim),
      alignment_(alignment < 32 ? 32 : alignment),
      block_vectors_(block_vectors),
      block_floats_(static_cast<size_t>(dim) * block_vectors) {
  if (block_vectors_ == 0) {
    block_vectors_ = 4096;
    block_floats_ = static_cast<size_t>(dim_) * block_vectors_;
  }
  if (reserve_vectors > 0) {
    size_t blocks = (reserve_vectors + block_vectors_ - 1) / block_vectors_;
    blocks_.reserve(blocks);
  }
}

void VectorStore::EnsureBlock(size_t block_index) {
  if (block_index < blocks_.size()) {
    return;
  }
  while (blocks_.size() <= block_index) {
    std::vector<float, AlignedAllocator<float>> block(
        block_floats_, 0.0f, AlignedAllocator<float>(alignment_));
    blocks_.push_back(std::move(block));
  }
}

size_t VectorStore::Insert(const float* data, WriteGuard& /*guard*/) {
  size_t slot = 0;
  if (!free_list_.empty()) {
    slot = free_list_.back();
    free_list_.pop_back();
  } else {
    slot = live_.size();
    live_.push_back(0);
  }
  size_t block_index = slot / block_vectors_;
  size_t block_offset = (slot % block_vectors_) * static_cast<size_t>(dim_);
  EnsureBlock(block_index);
  float* dest = blocks_[block_index].data() + block_offset;
  std::memcpy(dest, data, sizeof(float) * static_cast<size_t>(dim_));
  if (slot >= live_.size()) {
    live_.resize(slot + 1, 0);
  }
  if (live_[slot] == 0) {
    live_[slot] = 1;
    ++live_count_;
  }
  return slot;
}

void VectorStore::Update(size_t slot, const float* data, WriteGuard& /*guard*/) {
  size_t block_index = slot / block_vectors_;
  size_t block_offset = (slot % block_vectors_) * static_cast<size_t>(dim_);
  EnsureBlock(block_index);
  float* dest = blocks_[block_index].data() + block_offset;
  std::memcpy(dest, data, sizeof(float) * static_cast<size_t>(dim_));
  if (slot >= live_.size()) {
    live_.resize(slot + 1, 0);
  }
  if (live_[slot] == 0) {
    live_[slot] = 1;
    ++live_count_;
  }
}

void VectorStore::Release(size_t slot, WriteGuard& /*guard*/) {
  if (slot >= live_.size() || live_[slot] == 0) {
    return;
  }
  live_[slot] = 0;
  free_list_.push_back(slot);
  if (live_count_ > 0) {
    --live_count_;
  }
}

const float* VectorStore::Get(size_t slot, const ReadGuard& /*guard*/) const {
  return GetUnsafe(slot);
}

const float* VectorStore::GetUnsafe(size_t slot) const {
  size_t block_index = slot / block_vectors_;
  size_t block_offset = (slot % block_vectors_) * static_cast<size_t>(dim_);
  return blocks_[block_index].data() + block_offset;
}

float* VectorStore::GetUnsafe(size_t slot) {
  size_t block_index = slot / block_vectors_;
  size_t block_offset = (slot % block_vectors_) * static_cast<size_t>(dim_);
  return blocks_[block_index].data() + block_offset;
}

VectorStore::Stats VectorStore::GetStats() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  Stats stats;
  stats.live_vectors = live_count_;
  stats.total_vectors = live_.size();
  stats.free_vectors = free_list_.size();
  stats.blocks = blocks_.size();
  stats.bytes_allocated = blocks_.size() * block_floats_ * sizeof(float);
  return stats;
}

Status VectorStore::Save(std::FILE* out) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint64_t live_count = live_count_;
  if (std::fwrite(&live_count, sizeof(live_count), 1, out) != 1)
    return Status(StatusCode::kInternal, "write failed");
  if (std::fwrite(&dim_, sizeof(dim_), 1, out) != 1)
    return Status(StatusCode::kInternal, "write failed");
  if (std::fwrite(&block_vectors_, sizeof(block_vectors_), 1, out) != 1)
    return Status(StatusCode::kInternal, "write failed");

  uint64_t total_slots = live_.size();
  if (std::fwrite(&total_slots, sizeof(total_slots), 1, out) != 1)
    return Status(StatusCode::kInternal, "write failed");

  if (total_slots > 0) {
    if (std::fwrite(live_.data(), sizeof(uint8_t), total_slots, out) != total_slots) {
      return Status(StatusCode::kInternal, "write failed");
    }
  }

  uint64_t total_floats = total_slots * static_cast<uint64_t>(dim_);
  if (std::fwrite(&total_floats, sizeof(total_floats), 1, out) != 1)
    return Status(StatusCode::kInternal, "write failed");

  if (total_floats > 0) {
    for (size_t slot = 0; slot < live_.size(); ++slot) {
      const float* data = GetUnsafe(slot);
      if (std::fwrite(data, sizeof(float), dim_, out) != static_cast<size_t>(dim_)) {
        return Status(StatusCode::kInternal, "write failed");
      }
    }
  }
  return Status::Ok();
}

Status VectorStore::Load(std::FILE* in) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  uint64_t live_count = 0;
  if (std::fread(&live_count, sizeof(live_count), 1, in) != 1)
    return Status(StatusCode::kInternal, "read failed");
  int dim = 0;
  if (std::fread(&dim, sizeof(dim), 1, in) != 1)
    return Status(StatusCode::kInternal, "read failed");
  if (dim != dim_) return Status(StatusCode::kInternal, "dimension mismatch");

  size_t block_vectors = 0;
  if (std::fread(&block_vectors, sizeof(block_vectors), 1, in) != 1)
    return Status(StatusCode::kInternal, "read failed");
  if (block_vectors == 0) {
    return Status(StatusCode::kInternal, "invalid block size");
  }
  block_vectors_ = block_vectors;
  block_floats_ = static_cast<size_t>(dim_) * block_vectors_;

  uint64_t total_slots = 0;
  if (std::fread(&total_slots, sizeof(total_slots), 1, in) != 1)
    return Status(StatusCode::kInternal, "read failed");

  live_.assign(static_cast<size_t>(total_slots), 0);
  if (total_slots > 0) {
    if (std::fread(live_.data(), sizeof(uint8_t), total_slots, in) != total_slots) {
      return Status(StatusCode::kInternal, "read failed");
    }
  }

  uint64_t total_floats = 0;
  if (std::fread(&total_floats, sizeof(total_floats), 1, in) != 1)
    return Status(StatusCode::kInternal, "read failed");
  if (total_floats != total_slots * static_cast<uint64_t>(dim_)) {
    return Status(StatusCode::kInternal, "corrupt vector store");
  }

  blocks_.clear();
  size_t blocks_needed = (static_cast<size_t>(total_slots) + block_vectors_ - 1) / block_vectors_;
  blocks_.reserve(blocks_needed);
  for (size_t i = 0; i < blocks_needed; ++i) {
    std::vector<float, AlignedAllocator<float>> block(
        block_floats_, 0.0f, AlignedAllocator<float>(alignment_));
    blocks_.push_back(std::move(block));
  }
  for (size_t slot = 0; slot < live_.size(); ++slot) {
    float* dest = GetUnsafe(slot);
    if (std::fread(dest, sizeof(float), dim_, in) != static_cast<size_t>(dim_)) {
      return Status(StatusCode::kInternal, "read failed");
    }
  }

  size_t computed_live = 0;
  for (uint8_t flag : live_) {
    if (flag != 0) {
      ++computed_live;
    }
  }
  if (computed_live != static_cast<size_t>(live_count)) {
    return Status(StatusCode::kInternal, "corrupt vector store");
  }
  live_count_ = computed_live;
  free_list_.clear();
  for (size_t slot = 0; slot < live_.size(); ++slot) {
    if (live_[slot] == 0) {
      free_list_.push_back(slot);
    }
  }
  return Status::Ok();
}

}  // namespace pomai_search
