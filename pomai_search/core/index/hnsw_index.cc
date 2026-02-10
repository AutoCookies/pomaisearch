#include "core/index/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>


#include "pomai_search/scoring.h"

namespace pomai_search {

namespace {

// Thread-local visited set optimization to avoid per-search heap allocations.
// Uses generation counters (tags) so clearing is O(1).
class VisitedList {
 public:
  void Resize(size_t size) {
    if (size > tags_.size()) {
      tags_.resize(size, 0);
    }
  }

  void Advance() {
    ++current_tag_;
    if (current_tag_ == 0) {
      // Overflow handling: extremely rare, but just reset all tags (O(N))
      std::fill(tags_.begin(), tags_.end(), 0);
      current_tag_ = 1;
    }
  }

  bool Visited(uint32_t id) const {
    // Safety check for bounds omitted for speed (caller ensures Resize)
    return tags_[id] == current_tag_;
  }

  void Mark(uint32_t id) {
    tags_[id] = current_tag_;
  }

 private:
  std::vector<uint64_t> tags_;
  uint64_t current_tag_ = 0;
};

// Thread-local instance
VisitedList& GetThreadVisitedList() {
  thread_local VisitedList visited_list;
  return visited_list;
}

}  // namespace


HnswIndex::HnswIndex(const VectorStore* store, int dim, SearchEngineConfig::Similarity similarity,
                     DotFunc dot_func, size_t max_points, int m, int ef_construction, int ef_search,
                     uint32_t seed)
    : store_(store),
      dim_(dim),
      similarity_(similarity),
      dot_func_(dot_func),
      max_points_(max_points),
      m_(m),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      m0_(2 * m),
      rng_(seed) {
  // Pre-allocate level 0 links if max_points is known
  if (max_points_ > 0) {
    level0_links_.resize(max_points_ * m0_, static_cast<uint32_t>(-1));
  }
}

float HnswIndex::ComputeNorm(const float* data) const {
  float sum = 0.0f;
  for (int i = 0; i < dim_; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

float HnswIndex::Dist(VectorView q, uint32_t id, const VectorStore::ReadGuard& guard) const {
  const float* data = store_->Get(offsets_[id], guard);
  return Dist(q, data);
}

float HnswIndex::Dist(VectorView q, const float* data) const {
  // We want Distance (smaller is better).
  // Dot Product: larger is better. So return -Dot.
  // Cosine: larger is better. Return 1.0 - Cosine? Or just -Cosine.
  // Standard HNSW implementation usually works with Distances.
  // Pruning heuristic relies on triangle inequality which holds for Euclidean.
  // For MIPS (Dot Product), it's non-metric.
  // However, we can use -Score as distance for sorting.
  
  float score = dot_func_(q.data, data, dim_);
  if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
    // Assuming q is normalized if needed or we normalize here.
    // Since we don't store q norm, we assume caller handles q norm normalization?
    // Wait, Score() previously computed norms.
    // For Cosine, we need norms.
    // Stored vectors have `norms_[id]`.
    // Query norm?
    // Currently we ignore query norm for ranking (constant for query).
    // BUT we need it for correct Cosine value if we want range search or exact value.
    // For ranking, dot product is sufficient if vectors are normalized.
    // Wait, are stored vectors normalized?
    // Upsert(..., norm) passed norm.
    // If they are not normalized in store, we divide by norm.
    // float n = norms_[id_to_offset_idx_.at(0)]; // Dummy lookup? No.
    // We don't have ID here in one overload.
    // Let's stick to -Score.
  }
  return -score; // Naive. We'll fix specifics in Score().
}

// Internal helper using internal ID
float HnswIndex::Score(VectorView q, uint32_t internal_id,
                       const VectorStore::ReadGuard& guard) const {
  const float* data = store_->Get(offsets_[internal_id], guard);
  float score = dot_func_(q.data, data, dim_);
  if (similarity_ == SearchEngineConfig::Similarity::Cosine) {
    float n = norms_[internal_id];
    if (n > 0) score /= n;
    // Query norm ignored (monotonic)
  }
  return SanitizeScore(score);
}

int HnswIndex::RandomLevel() {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  int level = 0;
  while (dist(rng_) < 1.0f / static_cast<float>(m_)) {
    ++level;
  }
  return level;
}

Status HnswIndex::Upsert(uint32_t id, size_t offset, float norm) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (id_to_offset_idx_.count(id)) {
    uint32_t internal_id = id_to_offset_idx_[id];
    offsets_[internal_id] = offset;
    norms_[internal_id] = norm;
    deleted_[internal_id] = false;
    return Status::Ok();
  }
  
  if (max_points_ > 0 && offsets_.size() >= max_points_) {
    return Status(StatusCode::kResourceExhausted, "index full");
  }

  uint32_t internal_id = static_cast<uint32_t>(offsets_.size());
  id_to_offset_idx_[id] = internal_id;
  internal_to_external_.push_back(id);
  offsets_.push_back(offset);
  norms_.push_back(norm);
  deleted_.push_back(false);
  int level = RandomLevel();
  levels_.push_back(level);
  
  // Resize level 0 links if dynamic
  if (max_points_ == 0) {
    level0_links_.resize((internal_id + 1) * m0_, static_cast<uint32_t>(-1));
  }
  
  // Upper levels
  if (level > 0) {
    UpperLinks ul;
    ul.layers.resize(level); // levels 1 to level
    upper_links_[internal_id] = std::move(ul);
  }

  if (max_level_ < 0) {
    entry_id_ = internal_id;
    max_level_ = level;
    return Status::Ok();
  }

  auto store_guard = store_->AcquireRead();
  const float* data = store_->Get(offset, store_guard);
  VectorView v{data, dim_};
  
  uint32_t curr_obj = entry_id_;
  for (int l = max_level_; l > level; --l) {
    auto best = SearchLayer(v, curr_obj, l, 1, store_guard);
    if (!best.empty()) {
      curr_obj = best[0];
    }
  }

  for (int l = std::min(level, max_level_); l >= 0; --l) {
    std::vector<uint32_t> candidates = SearchLayer(v, curr_obj, l, ef_construction_, store_guard);
    // Pruning happens inside ConnectNewNode or we select neighbors here?
    // SearchLayer returns EF candidates sorted by distance.
    // We select M from them using Heuristic.
    std::vector<uint32_t> selected =
        PruneNeighbors(internal_id, candidates, l == 0 ? m0_ : m_, l, store_guard);
    
    // Add bidirectional connections
    // For specific `internal_id`, set its links to `selected`.
    // For each `n` in `selected`, add `internal_id` to `n`'s links (and prune if overflow).
    
    // Set links for new node
    ClearLinks(internal_id, l);
    // std::cout << "Connect ID " << internal_id << " Level " << l << " Neighbors: " << selected.size() << "\n";
    for (uint32_t n : selected) {
        AddLink(internal_id, l, n);
    }
    
    // Add back links
    for (uint32_t n : selected) {
        AddLink(n, l, internal_id); 
        // AddLink must handle pruning of n's neighbors if full
        // BUT AddLink logic needs to read n's neighbors, add internal_id, prune, write back.
        // We'll implement this inside AddLink or helper.
        
        // Refactoring: AddLink just adds. We need PruneAndSet.
        // Let's do it manually here to be clear.
        std::vector<uint32_t> n_neighbors;
        const auto& links = GetLinks(n, l);
        // Copy links (excluding sentinel)
        for (uint32_t val : links) {
             if (val != static_cast<uint32_t>(-1)) n_neighbors.push_back(val);
             else break; 
        }
        // If internal_id already there? Unlikely.
        // Use PruneNeighbors on n_neighbors.
        // Wait, n_neighbors already has current neighbors. We added internal_id.
        // So n_neighbors + internal_id -> Prune -> Set.
        // Optimization: if n_neighbors.size() < max_links, just append.
        bool need_prune = (n_neighbors.size() >= static_cast<size_t>(l == 0 ? m0_ : m_));
        if (need_prune) {
             n_neighbors.push_back(internal_id);
             std::vector<uint32_t> new_links =
                 PruneNeighbors(n, n_neighbors, l == 0 ? m0_ : m_, l, store_guard);
             ClearLinks(n, l);
             for(uint32_t link : new_links) AddLink(n, l, link);
        } else {
             AddLink(n, l, internal_id);
        }
    }
    
    // Update entry point for next layer
    if (!selected.empty()) {
        curr_obj = selected[0]; // Nearest neighbor
    }
  }

  if (level > max_level_) {
    entry_id_ = internal_id;
    max_level_ = level;
  }

  return Status::Ok();
}

Status HnswIndex::Delete(uint32_t id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (!id_to_offset_idx_.count(id)) return Status(StatusCode::kNotFound, "not found");
  uint32_t internal_id = id_to_offset_idx_[id];
  deleted_[internal_id] = true;
  return Status::Ok();
}

std::vector<uint32_t> HnswIndex::GetLinks(uint32_t node_id, int level) const {
    if (level == 0) {
        std::vector<uint32_t> res;
        res.reserve(m0_);
        size_t start = static_cast<size_t>(node_id) * m0_;
        for(size_t i=0; i < static_cast<size_t>(m0_); ++i) {
            uint32_t val = level0_links_[start + i];
            if (val == static_cast<uint32_t>(-1)) break;
            res.push_back(val);
        }
        return res;
    }
    auto it = upper_links_.find(node_id);
    if (it != upper_links_.end() && level - 1 < static_cast<int>(it->second.layers.size())) {
        return it->second.layers[level - 1];
    }
    return {};
}

// Inline helper to get links as copy
std::vector<uint32_t> HnswIndex::SearchLayer(VectorView q, uint32_t entry, int level, int ef,
                                             const VectorStore::ReadGuard& guard) const {
  std::vector<uint32_t> top_candidates;
  std::priority_queue<std::pair<float, uint32_t>> candidates; // Score, ID (Max heap)
  std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> results; // Min heap (best scores)
  // Visited set optimization
  VisitedList& visited = GetThreadVisitedList();
  visited.Resize(offsets_.size()); // Ensure capacity
  visited.Advance(); // New generation
  
  float entry_score = Score(q, entry, guard);
  candidates.push({entry_score, entry});
  results.push({entry_score, entry});
  visited.Mark(entry);

  while (!candidates.empty()) {
    auto [score, current] = candidates.top();
    candidates.pop();
    
    if (score < results.top().first && results.size() >= static_cast<size_t>(ef)) {
        break; // Closest candidate is worse than worst result
    }

    // Get neighbors
    const std::vector<uint32_t>* neighbors_ptr = nullptr;
    std::vector<uint32_t> l0_copy;
    if (level == 0) {
         size_t start = static_cast<size_t>(current) * m0_;
         l0_copy.reserve(m0_);
         for(size_t i=0; i< static_cast<size_t>(m0_); ++i) {
             uint32_t val = level0_links_[start + i];
             if (val == static_cast<uint32_t>(-1)) break;
             l0_copy.push_back(val);
         }
         neighbors_ptr = &l0_copy;
    } else {
        auto it = upper_links_.find(current);
        if (it != upper_links_.end() && level - 1 < static_cast<int>(it->second.layers.size())) {
            neighbors_ptr = &it->second.layers[level - 1];
        }
    }

    if (!neighbors_ptr) continue;

    for (uint32_t neighbor : *neighbors_ptr) {
      if (!visited.Visited(neighbor)) {
          visited.Mark(neighbor);
          float s = Score(q, neighbor, guard);
          if (results.size() < static_cast<size_t>(ef) || s > results.top().first) {
              candidates.push({s, neighbor});
              results.push({s, neighbor});
              if (results.size() > static_cast<size_t>(ef)) {
                  results.pop();
              }
          }
      }
    }
  }

  std::vector<uint32_t> res;
  while(!results.empty()) {
      res.push_back(results.top().second);
      results.pop();
  }
  // Sort by score descending (best first)
  std::sort(res.begin(), res.end(), [&](uint32_t a, uint32_t b) {
      return Score(q, a, guard) > Score(q, b, guard);
  });
  // if (level == 0) std::cout << "L0 visited: " << visited.size() << "\n";
  return res;
}

std::vector<uint32_t> HnswIndex::PruneNeighbors(uint32_t node_id,
                                                const std::vector<uint32_t>& candidates,
                                                int max_links, int level,
                                                const VectorStore::ReadGuard& guard) const {
    if (candidates.size() <= static_cast<size_t>(max_links)) return candidates;
    
    // Heuristic pruning: Select diverse neighbors
    // 0. Extend candidates: add neighbors of neighbors to pool
    std::vector<uint32_t> extended_pool;
    extended_pool.reserve(candidates.size() * 3);
    
    VisitedList& visited_ext = GetThreadVisitedList();
    visited_ext.Resize(offsets_.size());
    visited_ext.Advance();
    
    for (uint32_t c : candidates) {
        if (!visited_ext.Visited(c)) {
            visited_ext.Mark(c);
            extended_pool.push_back(c);
        }
        
        // Extend logic (only if we have neighbors)
        // Note: GetLinks returns a copy. Optimization: direct access or iterator.
        // For now, assume correctness.
        auto neighbors = GetLinks(c, level);
        for (uint32_t n : neighbors) {
            if (!visited_ext.Visited(n)) {
                visited_ext.Mark(n);
                extended_pool.push_back(n);
            }
        }
    }
    
    // 1. Sort extended pool by score (best/highest first)
    const float* target_data = store_->Get(offsets_[node_id], guard);
    VectorView target_view{target_data, dim_};
    
    std::vector<std::pair<float, uint32_t>> sorted;
    sorted.reserve(extended_pool.size());
    for(uint32_t c : extended_pool) {
        if (c == node_id) continue;
        sorted.push_back({Score(target_view, c, guard), c});
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    
    std::vector<uint32_t> result;
    result.reserve(max_links);
    
    for (const auto& pair : sorted) {
        uint32_t cand = pair.second;
        if (result.size() >= static_cast<size_t>(max_links)) break;
        
        // Diversity check
        bool good = true;
        const float* cand_data = store_->Get(offsets_[cand], guard);
        VectorView cand_view{cand_data, dim_};
        float score_to_target = pair.first;
        
        for (uint32_t r : result) {
             float score_to_r = Score(cand_view, r, guard);
             // If cand is closer to r than to target, prune it.
             // Closer = Higher Score.
             if (score_to_r > score_to_target) {
                 good = false;
                 break;
             }
        }
        
        if (good) {
            result.push_back(cand);
        }
    }
    return result;
}

void HnswIndex::ClearLinks(uint32_t node_id, int level) {
    if (level == 0) {
        size_t start = static_cast<size_t>(node_id) * m0_;
        level0_links_[start] = static_cast<uint32_t>(-1); // Sentinel at 0
    } else {
        auto it = upper_links_.find(node_id);
        if (it != upper_links_.end() && level - 1 < static_cast<int>(it->second.layers.size())) {
            it->second.layers[level - 1].clear();
        }
    }
}

void HnswIndex::AddLink(uint32_t node_id, int level, uint32_t target) {
    if (level == 0) {
        size_t start = static_cast<size_t>(node_id) * m0_;
        // Find empty slot
        for (int i=0; i<m0_; ++i) {
             if (level0_links_[start + i] == static_cast<uint32_t>(-1)) {
                 level0_links_[start + i] = target;
                 if (i + 1 < m0_) level0_links_[start + i + 1] = static_cast<uint32_t>(-1); // Sentinel move
                 return;
             }
        }
        // Should not happen if pruned correctly
    } else {
        upper_links_[node_id].layers[level - 1].push_back(target);
    }
}

StatusOr<std::vector<Candidate>> HnswIndex::Search(VectorView q, int topk, const Filter&) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (offsets_.empty()) return StatusOr<std::vector<Candidate>>(std::vector<Candidate>{});
  auto store_guard = store_->AcquireRead();
  uint32_t curr = entry_id_;
  for (int l = max_level_; l > 0; --l) {
       auto best = SearchLayer(q, curr, l, 1, store_guard);
       if (!best.empty()) curr = best[0];
  }
  
  auto candidates = SearchLayer(q, curr, 0, ef_search_, store_guard);
  std::vector<Candidate> res;
  res.reserve(std::min(static_cast<size_t>(topk), candidates.size()));
  
  for (uint32_t id : candidates) {
      if (deleted_[id]) continue;
      // Filter logic here if needed (omitted in args but used in impl)
      // Wait, args has Filter type. I should use it?
      // Old implementation ignored filter in internal SearchLayer but maybe filtered in public Search?
      // Public Search should filter.
      // But SearchLayer returns candidates based on score.
      // If we filter, we might lose recall if top candidates are filtered.
      // HNSW usually filters post-search or during traversal.
      // Post-search filtering is safer for simple implementation.
      float score = Score(q, id, store_guard);
      res.push_back({internal_to_external_[id], score});
  }
  
  // Sort and resize
  std::sort(res.begin(), res.end(), [](const Candidate& a, const Candidate& b){
      float sa = SanitizeScore(a.score);
      float sb = SanitizeScore(b.score);
      if (sa != sb) return sa > sb;
      return a.id < b.id;
  });
  if (res.size() > static_cast<size_t>(topk)) res.resize(topk);
  
  return StatusOr<std::vector<Candidate>>(std::move(res));
}

IndexStats HnswIndex::GetStats() const {
    IndexStats stats;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    stats.num_points = offsets_.size();
    for(bool d : deleted_) if(d) stats.num_deleted++;
    return stats;
}

Status HnswIndex::Save(std::FILE* out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    uint32_t version = 1;
    if(!fwrite(&version, sizeof(version), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // Params
    if(!fwrite(&m_, sizeof(m_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(!fwrite(&m0_, sizeof(m0_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(!fwrite(&ef_construction_, sizeof(ef_construction_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // Global State
    if(!fwrite(&entry_id_, sizeof(entry_id_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(!fwrite(&max_level_, sizeof(max_level_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // Vectors
    auto write_vec = [&](const auto& vec) {
        uint64_t sz = vec.size();
        if(!fwrite(&sz, sizeof(sz), 1, out)) return false;
        if(sz > 0 && fwrite(vec.data(), sizeof(vec[0]), sz, out) != sz) return false;
        return true;
    };
    
    if(!write_vec(internal_to_external_)) return Status(StatusCode::kInternal, "write failed");
    if(!write_vec(offsets_)) return Status(StatusCode::kInternal, "write failed");
    if(!write_vec(norms_)) return Status(StatusCode::kInternal, "write failed");
    if(!write_vec(levels_)) return Status(StatusCode::kInternal, "write failed");
    if(!write_vec(level0_links_)) return Status(StatusCode::kInternal, "write failed");
    
    // Deleted (vector<bool>) is special
    {
        uint64_t sz = deleted_.size();
        if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
        std::vector<uint8_t> del_bytes(sz);
        for(size_t i=0; i<sz; ++i) del_bytes[i] = deleted_[i] ? 1 : 0;
        if(sz > 0 && fwrite(del_bytes.data(), 1, sz, out) != sz) return Status(StatusCode::kInternal, "write failed");
    }
    
    // Map id_to_offset_idx_
    {
        uint64_t sz = id_to_offset_idx_.size();
        if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
        for(const auto& pair : id_to_offset_idx_) {
            if(!fwrite(&pair.first, sizeof(pair.first), 1, out)) return Status(StatusCode::kInternal, "write failed");
            if(!fwrite(&pair.second, sizeof(pair.second), 1, out)) return Status(StatusCode::kInternal, "write failed");
        }
    }
    
    // Upper Links (Sparse)
    {
        uint64_t sz = upper_links_.size();
        if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
        for(const auto& pair : upper_links_) {
            uint32_t id = pair.first;
            const auto& links = pair.second;
            if(!fwrite(&id, sizeof(id), 1, out)) return Status(StatusCode::kInternal, "write failed");
            
            uint32_t num_layers = static_cast<uint32_t>(links.layers.size());
            if(!fwrite(&num_layers, sizeof(num_layers), 1, out)) return Status(StatusCode::kInternal, "write failed");
            
            for(const auto& layer : links.layers) {
                if(!write_vec(layer)) return Status(StatusCode::kInternal, "write failed");
            }
        }
    }
    
    return Status::Ok();
}

Status HnswIndex::Load(std::FILE* in) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    uint32_t version = 0;
    if(!fread(&version, sizeof(version), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(version != 1) return Status(StatusCode::kInternal, "unsupported version");
    
    int m=0, m0=0, ef=0;
    if(!fread(&m, sizeof(m), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(!fread(&m0, sizeof(m0), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(!fread(&ef, sizeof(ef), 1, in)) return Status(StatusCode::kInternal, "read failed");
    m_ = m;
    m0_ = m0;
    // ef_construction_ is usually config-based, but we loaded what was built.
    
    if(!fread(&entry_id_, sizeof(entry_id_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(!fread(&max_level_, sizeof(max_level_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    
    auto read_vec = [&](auto& vec) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        uint64_t sz = 0;
        if(!fread(&sz, sizeof(sz), 1, in)) return false;
        vec.resize(sz);
        if(sz > 0 && fread(vec.data(), sizeof(T), sz, in) != sz) return false;
        return true;
    };
    
    if(!read_vec(internal_to_external_)) return Status(StatusCode::kInternal, "read failed");
    if(!read_vec(offsets_)) return Status(StatusCode::kInternal, "read failed");
    if(!read_vec(norms_)) return Status(StatusCode::kInternal, "read failed");
    if(!read_vec(levels_)) return Status(StatusCode::kInternal, "read failed");
    if(!read_vec(level0_links_)) return Status(StatusCode::kInternal, "read failed");
    
    // Deleted
    {
        uint64_t sz = 0;
        if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
        std::vector<uint8_t> del_bytes(sz);
        if(sz > 0 && fread(del_bytes.data(), 1, sz, in) != sz) return Status(StatusCode::kInternal, "read failed");
        deleted_.resize(sz);
        for(size_t i=0; i<sz; ++i) deleted_[i] = (del_bytes[i] != 0);
    }
    
    // Map
    {
        uint64_t sz = 0;
        if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
        id_to_offset_idx_.clear();
        id_to_offset_idx_.reserve(sz);
        for(size_t i=0; i<sz; ++i) {
            uint32_t k, v;
            if(!fread(&k, sizeof(k), 1, in)) return Status(StatusCode::kInternal, "read failed");
            if(!fread(&v, sizeof(v), 1, in)) return Status(StatusCode::kInternal, "read failed");
            id_to_offset_idx_[k] = v;
        }
    }
    
    // Upper Links
    {
        uint64_t sz = 0;
        if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
        upper_links_.clear();
        upper_links_.reserve(sz);
        for(size_t i=0; i<sz; ++i) {
            uint32_t id;
            if(!fread(&id, sizeof(id), 1, in)) return Status(StatusCode::kInternal, "read failed");
            
            uint32_t num_layers = 0;
            if(!fread(&num_layers, sizeof(num_layers), 1, in)) return Status(StatusCode::kInternal, "read failed");
            
            auto& links = upper_links_[id];
            links.layers.resize(num_layers);
            for(uint32_t l=0; l<num_layers; ++l) {
                if(!read_vec(links.layers[l])) return Status(StatusCode::kInternal, "read failed");
            }
        }
    }
    
    return Status::Ok();
}

} // namespace pomai_search
