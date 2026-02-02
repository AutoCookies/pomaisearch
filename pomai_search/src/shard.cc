#include "shard.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>

namespace pomai_search {

Shard::Shard(int dim, size_t alignment, size_t reserve_vectors, SearchEngineConfig::Similarity similarity,
             DotFunc dot_func, size_t max_points)
    : dim_(dim),
      similarity_(similarity),
      dot_func_(dot_func),
      max_points_(max_points),
      arena_(dim, alignment, reserve_vectors) {}

Status Shard::Upsert(std::string_view key, VectorView vec, Metadata meta,
                     std::optional<std::chrono::steady_clock::time_point> expiry) {
  if (vec.dim != dim_ || vec.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = index_.find(std::string(key));
  if (it != index_.end()) {
    auto& item = items_[it->second];
    item.meta = std::move(meta);
    item.deleted = false;
    item.expiry = expiry;
    item.offset = arena_.Append(vec.data);
    item.norm = ComputeNorm(vec.data);
    return Status::Ok();
  }
  if (max_points_ > 0 && items_.size() >= max_points_) {
    return Status(StatusCode::kResourceExhausted, "shard is at capacity");
  }
  size_t offset = arena_.Append(vec.data);
  ShardItem item;
  item.key = std::string(key);
  item.meta = std::move(meta);
  item.offset = offset;
  item.norm = ComputeNorm(vec.data);
  item.deleted = false;
  item.expiry = expiry;
  index_.emplace(item.key, items_.size());
  items_.push_back(std::move(item));
  return Status::Ok();
}

Status Shard::Delete(std::string_view key) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = index_.find(std::string(key));
  if (it == index_.end()) {
    return Status(StatusCode::kNotFound, "key not found");
  }
  items_[it->second].deleted = true;
  return Status::Ok();
}

StatusOr<bool> Shard::Exists(std::string_view key) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = index_.find(std::string(key));
  if (it == index_.end()) {
    return Status(StatusCode::kNotFound, "key not found");
  }
  const auto& item = items_[it->second];
  bool exists = !item.deleted && !IsExpired(item.expiry);
  return StatusOr<bool>(exists);
}

struct Candidate {
  float score;
  const ShardItem* item;
};

static bool IsBetter(const Candidate& a, const Candidate& b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return a.item->key < b.item->key;
}

StatusOr<std::vector<ResultItem>> Shard::Search(VectorView query, int topk, const Metadata& filter,
                                                float query_norm) const {
  if (query.dim != dim_ || query.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  if (topk <= 0) {
    return Status(StatusCode::kInvalidArgument, "topk must be positive");
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<Candidate> best;
  best.reserve(static_cast<size_t>(topk));
  for (const auto& item : items_) {
    if (item.deleted || IsExpired(item.expiry)) {
      continue;
    }
    if (!MetadataFilterMatch(filter, item.meta)) {
      continue;
    }
    const float* data = arena_.Get(item.offset);
    float score = dot_func_(query.data, data, dim_);
    if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
      if (item.norm == 0.0f || query_norm == 0.0f) {
        continue;
      }
      score = score / (item.norm * query_norm);
    }
    Candidate cand{score, &item};
    if (static_cast<int>(best.size()) < topk) {
      best.push_back(cand);
      continue;
    }
    auto worst_it = best.begin();
    for (auto it = best.begin(); it != best.end(); ++it) {
      if (IsBetter(*worst_it, *it)) {
        worst_it = it;
      }
    }
    if (IsBetter(cand, *worst_it)) {
      *worst_it = cand;
    }
  }
  std::vector<ResultItem> results;
  results.reserve(best.size());
  for (const auto& cand : best) {
    ResultItem item;
    item.key = cand.item->key;
    item.score = cand.score;
    item.meta = cand.item->meta;
    results.push_back(std::move(item));
  }
  std::sort(results.begin(), results.end(), [](const ResultItem& a, const ResultItem& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.key < b.key;
  });
  return StatusOr<std::vector<ResultItem>>(std::move(results));
}

StatusOr<std::vector<float>> Shard::GetVectorCopy(std::string_view key) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = index_.find(std::string(key));
  if (it == index_.end()) {
    return Status(StatusCode::kNotFound, "key not found");
  }
  const auto& item = items_[it->second];
  if (item.deleted || IsExpired(item.expiry)) {
    return Status(StatusCode::kNotFound, "key not found");
  }
  std::vector<float> vec(dim_);
  const float* data = arena_.Get(item.offset);
  std::copy(data, data + dim_, vec.begin());
  return StatusOr<std::vector<float>>(std::move(vec));
}

uint64_t Shard::num_points() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return items_.size();
}

uint64_t Shard::num_deleted() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint64_t deleted = 0;
  for (const auto& item : items_) {
    if (item.deleted || IsExpired(item.expiry)) {
      ++deleted;
    }
  }
  return deleted;
}

float Shard::ComputeNorm(const float* data) const {
  float sum = 0.0f;
  for (int i = 0; i < dim_; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

}  // namespace pomai_search
