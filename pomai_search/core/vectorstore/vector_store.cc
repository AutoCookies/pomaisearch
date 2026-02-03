#include "core/vectorstore/vector_store.h"

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

Status VectorStore::Save(std::FILE* out) const {
  uint64_t count = count_;
  if (std::fwrite(&count, sizeof(count), 1, out) != 1) return Status(StatusCode::kInternal, "write failed");
  if (std::fwrite(&dim_, sizeof(dim_), 1, out) != 1) return Status(StatusCode::kInternal, "write failed");
  
  uint64_t data_size = data_.size();
  if (std::fwrite(&data_size, sizeof(data_size), 1, out) != 1) return Status(StatusCode::kInternal, "write failed");
  
  if (data_size > 0) {
      if (std::fwrite(data_.data(), sizeof(float), data_size, out) != data_size) {
          return Status(StatusCode::kInternal, "write failed");
      }
  }
  return Status::Ok();
}

Status VectorStore::Load(std::FILE* in) {
  uint64_t count = 0;
  if (std::fread(&count, sizeof(count), 1, in) != 1) return Status(StatusCode::kInternal, "read failed");
  int dim = 0;
  if (std::fread(&dim, sizeof(dim), 1, in) != 1) return Status(StatusCode::kInternal, "read failed");
  if (dim != dim_) return Status(StatusCode::kInternal, "dimension mismatch");
  
  uint64_t data_size = 0;
  if (std::fread(&data_size, sizeof(data_size), 1, in) != 1) return Status(StatusCode::kInternal, "read failed");
  
  data_.resize(data_size);
  if (data_size > 0) {
      if (std::fread(data_.data(), sizeof(float), data_size, in) != data_size) {
          return Status(StatusCode::kInternal, "read failed");
      }
  }
  count_ = count;
  return Status::Ok();
}

}  // namespace pomai_search
