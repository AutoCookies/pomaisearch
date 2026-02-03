#pragma once

#include <vector>
#include <random>
#include "core/kernels/kernels.h"
#include "core/vectorstore/vector_store.h"
#include "pomai_search/status.h"
#include "pomai_search/search_engine.h"

namespace pomai_search {

// Simple CPU K-Means implementation (Lloyd's algorithm).
class KMeans {
 public:
  struct Config {
    int k;                  // Number of centroids
    int max_iterations = 20;
    int min_points_per_centroid = 39; // Suggested by Faiss (train set size = 39*k)
    uint32_t seed = 42;
    SearchEngineConfig::Similarity similarity = SearchEngineConfig::Similarity::Dot; // Usually L2 for IVF, but we support Dot
  };

  KMeans(Config config, int dim);

  // Train centroids using a subset of vectors from VectorStore.
  // indices: The list of VectorStore offsets (IDs) to train on.
  Status Train(const VectorStore& store, const std::vector<uint32_t>& indices);

  // Assign vectors to nearest centroid.
  // Returns: vector of centroid indices, distances.
  StatusOr<std::pair<std::vector<int>, std::vector<float>>> Assign(
      const float* data, int num_vectors) const;
      
  // Assign single vector
  int AssignOne(const float* vector) const;

  const std::vector<float>& centroids() const { return centroids_; }

 private:
  Config config_;
  int dim_;
  std::vector<float> centroids_; // k * dim
  std::mt19937 rng_;
};

}  // namespace pomai_search
