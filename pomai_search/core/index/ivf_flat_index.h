#pragma once

#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "core/clustering/kmeans.h"
#include "core/index/index.h"
#include "core/vectorstore/vector_store.h"

namespace pomai_search {

class IvfFlatIndex : public Index {
 public:
  struct Config {
    int nlist = 100;                 // Number of centroids
    int nprobe = 10;                 // Search width
    size_t min_train_size = 0;       // If 0, defaults to 40 * nlist
    float stale_ratio_threshold = 0.35f;
    size_t max_list_size = 50000;
    size_t max_list_bytes = 1u << 20;
    SearchEngineConfig::Similarity similarity = SearchEngineConfig::Similarity::Dot;
  };

  IvfFlatIndex(const VectorStore* store, int dim, Config config);

  Status Upsert(uint32_t id, size_t offset, float norm) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override;
  IndexStats GetStats() const override;
  Status Save(std::FILE* out) const override;
  Status Load(std::FILE* in) override;

  bool IsTrained() const { return trained_; }
  Status Train();

 private:
  struct DocState {
    size_t offset = 0;
    uint32_t gen = 0;
    bool deleted = false;
  };

  struct PostingEntry {
    uint32_t id = 0;
    uint32_t gen = 0;
  };

  struct ListMetrics {
    size_t stale_entries = 0;
    size_t stale_checked = 0;
  };

  const VectorStore* store_;
  int dim_;
  Config config_;

  mutable std::shared_mutex mutex_;
  bool trained_ = false;

  KMeans kmeans_;
  std::vector<std::vector<PostingEntry>> lists_;
  std::vector<ListMetrics> list_metrics_;

  std::unordered_map<uint32_t, DocState> docs_;
  std::vector<uint32_t> buffer_ids_;

  uint64_t compaction_runs_ = 0;
  uint64_t compaction_time_ms_ = 0;

  Status TrainImpl(const std::unique_lock<std::shared_mutex>& lock);
  Status AddToInvertedList(uint32_t id, uint32_t gen, const float* vec);
  bool EntryIsLive(const PostingEntry& entry) const;
  bool ListNeedsCompaction(size_t list_idx) const;
  void CompactList(size_t list_idx);
};

}  // namespace pomai_search
