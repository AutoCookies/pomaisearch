#pragma once

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "core/index/index.h"
#include "core/clustering/kmeans.h"
#include "core/vectorstore/vector_store.h"

namespace pomai_search {

class IvfFlatIndex : public Index {
 public:
  struct Config {
      int nlist = 100;        // Number of centroids
      int nprobe = 10;        // Search width
      size_t min_train_size = 0; // If 0, defaults to 40 * nlist
      SearchEngineConfig::Similarity similarity = SearchEngineConfig::Similarity::Dot;
  };

  IvfFlatIndex(const VectorStore* store, int dim, Config config);

  Status Upsert(uint32_t id, size_t offset, float norm) override;
  Status Delete(uint32_t id) override;
  StatusOr<std::vector<Candidate>> Search(VectorView q, int topk, const Filter& f) const override;
  void Compact() override {}
  IndexStats GetStats() const override;

  bool IsTrained() const { return trained_; }
  Status Train();

 private:
  const VectorStore* store_;
  int dim_;
  Config config_;
  
  mutable std::shared_mutex mutex_;
  bool trained_ = false;
  
  // Storage for trained state
  KMeans kmeans_; // Used for quantization (centroids)
  std::vector<std::vector<uint32_t>> lists_; // Inverted lists [centroid_idx] -> [doc_id, doc_id...]
  // Wait, we need doc_ids. 'id' passed to Upsert is external ID (uint32_t).
  // HNSW stores `offsets_` (size_t) and `internal_to_external_`.
  // If `id` is external ID, we can store it directly in lists if we don't map to internal ID.
  // BUT we need `offset` to get data from VectorStore.
  // So we need a map: `id -> offset`.
  // HNSW uses `id_to_offset_idx_` and `offsets_` vector.
  // We can do the same.
  // Mapping:
  // External ID -> Internal Offset (in member vector).
  // Inverted Lists -> Store External IDs? Or Internal IDs?
  // Storing External IDs in lists is easier for results.
  // But for re-ranking/scoring, we need VectorStore offset.
  // So we need `Map<ExtID, Offset>`.
  std::unordered_map<uint32_t, size_t> id_to_offset_; 
  // Wait, `offsets_` in HNSW was vector<size_t>.
  // Here just a map is enough?
  
  // Pre-training buffer
  std::vector<uint32_t> buffer_ids_;
  
  // Helpers
  Status TrainImpl(const std::unique_lock<std::shared_mutex>& lock);
  Status AddToBuffer(uint32_t id, size_t offset);
  Status AddToInvertedList(uint32_t id, const float* vec);
};

}  // namespace pomai_search
