#pragma once

#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "pomai_search/index/index.h"
#include "pomai_search/simd/kernels.h"
#include "pomai_search/vector_store.h"

namespace pomai_search {

class HnswIndex : public Index {
 public:
  HnswIndex(const VectorStore* store, int dim, SearchEngineConfig::Similarity similarity,
            DotFunc dot_func, size_t max_points, int m, int ef_construction, int ef_search,
            uint32_t seed);

  Status Upsert(uint32_t id, size_t offset, float norm) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override {}
  IndexStats GetStats() const override;

 private:
  float ComputeNorm(const float* data) const;
  float Dist(VectorView q, uint32_t id) const;
  float Dist(VectorView q, const float* data) const;
  float Score(VectorView q, uint32_t internal_id) const;
  int RandomLevel();
  std::vector<uint32_t> SearchLayer(VectorView q, uint32_t entry, int level, int ef) const;
  void ConnectNewNode(uint32_t node_id, int level, const std::vector<uint32_t>& candidates);
  std::vector<uint32_t> PruneNeighbors(uint32_t node_id, const std::vector<uint32_t>& candidates,
                                       int max_links) const;
  
  // Helpers for memory layout
  void SetLink(uint32_t node_id, int level, int idx, uint32_t target);
  uint32_t GetLink(uint32_t node_id, int level, int idx) const;
  void AddLink(uint32_t node_id, int level, uint32_t target);
  void ClearLinks(uint32_t node_id, int level);
  std::vector<uint32_t> GetLinks(uint32_t node_id, int level) const;

  const VectorStore* store_;
  int dim_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;
  int m_;
  int ef_construction_;
  int ef_search_;
  int m0_;  // Max links at level 0 (2 * m_)

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint32_t, uint32_t> id_to_offset_idx_; // Ext ID -> Internal Index
  std::vector<uint32_t> internal_to_external_;             // Internal Index -> Ext ID
  
  // Flattened Node Data
  std::vector<size_t> offsets_;
  std::vector<float> norms_;
  std::vector<int> levels_;
  std::vector<bool> deleted_; // Use vector<char> or bitset for density? vector<bool> is bitset specialized.
  
  // Graph Data
  // Level 0: Flattened array. size = capacity * m0_.
  // Access: level0_links_[internal_id * m0_ + i]
  // We need to count how many links are present? 
  // Or just use sentinel? 
  // Sentinel (uint32_t(-1)) is better for cache than separate count.
  std::vector<uint32_t> level0_links_;
  
  // Upper Levels: Sparse map.
  // Map internal_id -> vector of levels -> vector of links
  struct UpperLinks {
    std::vector<std::vector<uint32_t>> layers;
  };
  std::unordered_map<uint32_t, UpperLinks> upper_links_;

  uint32_t entry_id_ = 0; // Internal ID
  int max_level_ = -1;
  mutable std::mt19937 rng_;
  
  // Visited set optimization
  // We use thread-local visited sets? No, Search is const.
  // We need per-thread scratch space.
  // Or we pass scratch space to Search functions.
  // Mutable member is NOT thread safe if shared.
  // We cannot use a single `visited_tags_` in the class unless we use thread-local or pass it down.
  // Given `Search` is `const`, we should allocate scratch space on stack or use thread_local.
  // `vector<uint32_t> visited_tags` is large (N * 4 bytes). N=1M -> 4MB.
  // Stack allocation is risky.
  // ThreadPool should manage scratch space?
  // For now, we allocate `vector<uint32_t>` in `Search` (no optimization yet) OR use `Bitset` if N is small.
  // Optimization: ThreadLocal scratch buffer.


};

}  // namespace pomai_search
