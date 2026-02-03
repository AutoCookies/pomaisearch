#pragma once

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "core/index/index.h"
#include "core/clustering/kmeans.h"
#include "core/quantization/scalar_quantizer.h"
#include "core/vectorstore/vector_store.h"

namespace pomai_search {

class IvfSq8Index : public Index {
 public:
  struct Config {
      int nlist = 100;
      int nprobe = 10;
      size_t min_train_size = 0;
      float refine_factor = 3.0f; // Multiplier for topk in coarse search
      SearchEngineConfig::Similarity similarity = SearchEngineConfig::Similarity::Dot;
  };

  IvfSq8Index(const VectorStore* store, int dim, Config config);

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
  
  KMeans kmeans_;
  ScalarQuantizer quantizer_;
  
  std::vector<std::vector<uint32_t>> lists_;
  std::vector<std::vector<uint8_t>> codes_; // Flattened codes for each list
  
  std::unordered_map<uint32_t, size_t> id_to_offset_; 
  std::vector<uint32_t> buffer_ids_;
  
  Status TrainImpl(const std::unique_lock<std::shared_mutex>& lock);
  Status AddToInvertedList(uint32_t id, const float* vec);
};

}  // namespace pomai_search
