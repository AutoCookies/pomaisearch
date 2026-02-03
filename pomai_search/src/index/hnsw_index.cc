#include "pomai_search/index/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <queue>
#include <unordered_set>

namespace pomai_search {

HnswIndex::HnswIndex(int dim, SearchEngineConfig::Similarity similarity, DotFunc dot_func,
                     size_t alignment, size_t reserve_vectors, size_t max_points, int m,
                     int ef_construction, int ef_search, uint32_t seed)
    : dim_(dim),
      similarity_(similarity),
      dot_func_(dot_func),
      max_points_(max_points),
      m_(m),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      arena_(dim, alignment, reserve_vectors),
      rng_(seed) {}

Status HnswIndex::Upsert(uint32_t id, VectorView v) {
  if (v.dim != dim_ || v.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = id_to_index_.find(id);
  if (it != id_to_index_.end()) {
    Node& node = nodes_[it->second];
    node.deleted = false;
    node.offset = arena_.Append(v.data);
    node.norm = ComputeNorm(v.data);
    return Status::Ok();
  }
  if (max_points_ > 0 && nodes_.size() >= max_points_) {
    return Status(StatusCode::kResourceExhausted, "index at capacity");
  }
  Node node;
  node.offset = arena_.Append(v.data);
  node.norm = ComputeNorm(v.data);
  node.level = RandomLevel();
  node.deleted = false;
  node.neighbors.resize(static_cast<size_t>(node.level) + 1);

  uint32_t node_id = id;
  id_to_index_.emplace(node_id, nodes_.size());
  ids_.push_back(node_id);
  nodes_.push_back(node);

  if (max_level_ < 0) {
    entry_id_ = node_id;
    max_level_ = node.level;
    return Status::Ok();
  }

  uint32_t entry = entry_id_;
  for (int level = max_level_; level > node.level; --level) {
    auto best = SearchLayer(v, entry, level, 1);
    if (!best.empty()) {
      entry = best.front();
    }
  }

  for (int level = std::min(node.level, max_level_); level >= 0; --level) {
    auto candidates = SearchLayer(v, entry, level, ef_construction_);
    ConnectNewNode(node_id, level, candidates);
  }

  if (node.level > max_level_) {
    entry_id_ = node_id;
    max_level_ = node.level;
  }

  return Status::Ok();
}

Status HnswIndex::Delete(uint32_t id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    return Status(StatusCode::kNotFound, "id not found");
  }
  nodes_[it->second].deleted = true;
  return Status::Ok();
}

struct ScoredId {
  float score;
  uint32_t id;
};

struct BetterScore {
  bool operator()(const ScoredId& a, const ScoredId& b) const {
    if (a.score != b.score) {
      return a.score < b.score;
    }
    return a.id > b.id;
  }
};

struct WorseScore {
  bool operator()(const ScoredId& a, const ScoredId& b) const {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  }
};

std::vector<uint32_t> HnswIndex::SearchLayer(VectorView q, uint32_t entry, int level, int ef) const {
  std::priority_queue<ScoredId, std::vector<ScoredId>, BetterScore> candidates;
  std::priority_queue<ScoredId, std::vector<ScoredId>, WorseScore> results;
  std::unordered_set<uint32_t> visited;

  auto entry_it = id_to_index_.find(entry);
  if (entry_it == id_to_index_.end()) {
    return {};
  }
  const Node& entry_node = nodes_[entry_it->second];
  float entry_score = Score(q, entry_node);
  candidates.push({entry_score, entry});
  results.push({entry_score, entry});
  visited.insert(entry);

  while (!candidates.empty()) {
    ScoredId current = candidates.top();
    if (!results.empty() && current.score < results.top().score) {
      break;
    }
    candidates.pop();
    auto current_it = id_to_index_.find(current.id);
    if (current_it == id_to_index_.end()) {
      continue;
    }
    const Node& node = nodes_[current_it->second];
    if (level >= static_cast<int>(node.neighbors.size())) {
      continue;
    }
    for (uint32_t neighbor : node.neighbors[static_cast<size_t>(level)]) {
      if (visited.find(neighbor) != visited.end()) {
        continue;
      }
      visited.insert(neighbor);
      auto neighbor_it = id_to_index_.find(neighbor);
      if (neighbor_it == id_to_index_.end()) {
        continue;
      }
      float score = Score(q, nodes_[neighbor_it->second]);
      if (static_cast<int>(results.size()) < ef || score > results.top().score) {
        candidates.push({score, neighbor});
        results.push({score, neighbor});
        if (static_cast<int>(results.size()) > ef) {
          results.pop();
        }
      }
    }
  }

  std::vector<ScoredId> scored;
  scored.reserve(results.size());
  while (!results.empty()) {
    scored.push_back(results.top());
    results.pop();
  }
  std::sort(scored.begin(), scored.end(), [](const ScoredId& a, const ScoredId& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  });
  std::vector<uint32_t> ids;
  ids.reserve(scored.size());
  for (const auto& item : scored) {
    ids.push_back(item.id);
  }
  return ids;
}

void HnswIndex::ConnectNewNode(uint32_t node_id, int level, const std::vector<uint32_t>& candidates) {
  auto node_it = id_to_index_.find(node_id);
  if (node_it == id_to_index_.end()) {
    return;
  }
  Node& node = nodes_[node_it->second];
  std::vector<ScoredId> scored;
  scored.reserve(candidates.size());
  VectorView view{arena_.Get(node.offset), dim_};
  for (uint32_t candidate : candidates) {
    if (candidate == node_id) {
      continue;
    }
    auto candidate_it = id_to_index_.find(candidate);
    if (candidate_it == id_to_index_.end()) {
      continue;
    }
    float score = Score(view, nodes_[candidate_it->second]);
    scored.push_back({score, candidate});
  }
  std::sort(scored.begin(), scored.end(), [](const ScoredId& a, const ScoredId& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  });
  if (static_cast<int>(scored.size()) > m_) {
    scored.resize(static_cast<size_t>(m_));
  }
  if (level >= static_cast<int>(node.neighbors.size())) {
    return;
  }
  auto& neighbors = node.neighbors[static_cast<size_t>(level)];
  neighbors.clear();
  for (const auto& item : scored) {
    neighbors.push_back(item.id);
    auto other_it = id_to_index_.find(item.id);
    if (other_it == id_to_index_.end()) {
      continue;
    }
    Node& other = nodes_[other_it->second];
    if (level >= static_cast<int>(other.neighbors.size())) {
      continue;
    }
    auto& other_neighbors = other.neighbors[static_cast<size_t>(level)];
    other_neighbors.push_back(node_id);
    std::sort(other_neighbors.begin(), other_neighbors.end());
    other_neighbors.erase(std::unique(other_neighbors.begin(), other_neighbors.end()),
                          other_neighbors.end());
    if (static_cast<int>(other_neighbors.size()) > m_) {
      other_neighbors.resize(static_cast<size_t>(m_));
    }
  }
}

StatusOr<std::vector<Candidate>> HnswIndex::Search(VectorView q, int topk, const Filter& filter) const {
  (void)filter;
  if (q.dim != dim_ || q.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  if (topk <= 0) {
    return Status(StatusCode::kInvalidArgument, "topk must be positive");
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (nodes_.empty()) {
    return StatusOr<std::vector<Candidate>>(std::vector<Candidate>{});
  }
  uint32_t entry = entry_id_;
  for (int level = max_level_; level > 0; --level) {
    auto best = SearchLayer(q, entry, level, 1);
    if (!best.empty()) {
      entry = best.front();
    }
  }
  auto candidates = SearchLayer(q, entry, 0, ef_search_);
  std::vector<Candidate> results;
  results.reserve(candidates.size());
  for (uint32_t id : candidates) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
      continue;
    }
    const Node& node = nodes_[it->second];
    if (node.deleted) {
      continue;
    }
    float score = Score(q, node);
    results.push_back(Candidate{id, score});
  }
  std::sort(results.begin(), results.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  });
  if (static_cast<int>(results.size()) > topk) {
    results.resize(static_cast<size_t>(topk));
  }
  return StatusOr<std::vector<Candidate>>(std::move(results));
}

IndexStats HnswIndex::GetStats() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  IndexStats stats;
  stats.num_points = nodes_.size();
  for (const auto& node : nodes_) {
    if (node.deleted) {
      ++stats.num_deleted;
    }
  }
  return stats;
}

float HnswIndex::ComputeNorm(const float* data) const {
  float sum = 0.0f;
  for (int i = 0; i < dim_; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

float HnswIndex::Score(VectorView q, const Node& node) const {
  const float* data = arena_.Get(node.offset);
  float score = dot_func_(q.data, data, dim_);
  if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
    float query_norm = ComputeNorm(q.data);
    if (node.norm == 0.0f || query_norm == 0.0f) {
      return 0.0f;
    }
    score /= (node.norm * query_norm);
  }
  return score;
}

int HnswIndex::RandomLevel() {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  int level = 0;
  while (dist(rng_) < 1.0f / static_cast<float>(m_)) {
    ++level;
  }
  return level;
}

}  // namespace pomai_search
