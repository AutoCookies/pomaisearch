#include "pomai_search/index/flat_index.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace pomai_search {

FlatIndex::FlatIndex(int dim, SearchEngineConfig::Similarity similarity, DotFunc dot_func,
                     size_t alignment, size_t reserve_vectors, size_t max_points)
    : dim_(dim),
      similarity_(similarity),
      dot_func_(dot_func),
      max_points_(max_points),
      arena_(dim, alignment, reserve_vectors) {}

Status FlatIndex::Upsert(uint32_t id, VectorView v) {
  if (v.dim != dim_ || v.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = id_to_index_.find(id);
  if (it != id_to_index_.end()) {
    auto& item = items_[it->second];
    item.deleted = false;
    item.offset = arena_.Append(v.data);
    item.norm = ComputeNorm(v.data);
    return Status::Ok();
  }
  if (max_points_ > 0 && items_.size() >= max_points_) {
    return Status(StatusCode::kResourceExhausted, "index at capacity");
  }
  size_t offset = arena_.Append(v.data);
  Item item;
  item.offset = offset;
  item.norm = ComputeNorm(v.data);
  item.deleted = false;
  id_to_index_.emplace(id, items_.size());
  ids_.push_back(id);
  items_.push_back(item);
  return Status::Ok();
}

Status FlatIndex::Delete(uint32_t id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    return Status(StatusCode::kNotFound, "id not found");
  }
  items_[it->second].deleted = true;
  return Status::Ok();
}

struct FlatCandidate {
  float score;
  uint32_t id;
};

static bool IsBetter(const FlatCandidate& a, const FlatCandidate& b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return a.id < b.id;
}

StatusOr<std::vector<Candidate>> FlatIndex::Search(VectorView q, int topk, const Filter& filter) const {
  (void)filter;
  if (q.dim != dim_ || q.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  if (topk <= 0) {
    return Status(StatusCode::kInvalidArgument, "topk must be positive");
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<FlatCandidate> heap;
  heap.reserve(static_cast<size_t>(topk));
  auto worse_first = [](const FlatCandidate& a, const FlatCandidate& b) { return IsBetter(a, b); };
  float query_norm = 1.0f;
  if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
    query_norm = ComputeNorm(q.data);
  }
  for (size_t i = 0; i < items_.size(); ++i) {
    const auto& item = items_[i];
    if (item.deleted) {
      continue;
    }
    const float* data = arena_.Get(item.offset);
    float score = dot_func_(q.data, data, dim_);
    if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
      if (item.norm == 0.0f || query_norm == 0.0f) {
        score = 0.0f;
      } else {
        score /= (item.norm * query_norm);
      }
    }
    FlatCandidate cand{score, ids_[i]};
    if (static_cast<int>(heap.size()) < topk) {
      heap.push_back(cand);
      std::push_heap(heap.begin(), heap.end(), worse_first);
      continue;
    }
    if (IsBetter(cand, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), worse_first);
      heap.back() = cand;
      std::push_heap(heap.begin(), heap.end(), worse_first);
    }
  }
  std::vector<Candidate> results;
  results.reserve(heap.size());
  for (const auto& cand : heap) {
    results.push_back(Candidate{cand.id, cand.score});
  }
  std::sort(results.begin(), results.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  });
  return StatusOr<std::vector<Candidate>>(std::move(results));
}

IndexStats FlatIndex::GetStats() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  IndexStats stats;
  stats.num_points = items_.size();
  for (const auto& item : items_) {
    if (item.deleted) {
      ++stats.num_deleted;
    }
  }
  return stats;
}

float FlatIndex::ComputeNorm(const float* data) const {
  float sum = 0.0f;
  for (int i = 0; i < dim_; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

}  // namespace pomai_search
