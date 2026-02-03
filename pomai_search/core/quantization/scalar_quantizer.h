#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>
#include "core/vectorstore/vector_store.h"
#include "core/kernels/kernels.h"
#include "pomai_search/status.h"

namespace pomai_search {

// Scalar Quantizer (SQ8)
// Compresses float vectors to uint8_t codes.
// Algorithm: Uniform per-dimension quantization.
class ScalarQuantizer {
 public:
  explicit ScalarQuantizer(int dim);

  // Compute min/max for each dimension from sample data.
  // indices: Offsets into VectorStore.
  void Train(const VectorStore& store, const std::vector<uint32_t>& indices);

  // Encode a vector into codes (size = dim).
  void Encode(const float* vec, uint8_t* out_codes) const;

  // Decode codes back to float (approximate).
  void Decode(const uint8_t* codes, float* out_vec) const;

  // Compute L2 squared distances between query (float) and multiple codes (uint8).
  // query: float[dim]
  // codes: uint8[n * dim] (contiguous)
  // out_dists: float[n]
  void ComputeDistancesL2(const float* query, const uint8_t* codes, int n, float* out_dists) const;

  // Compute Dot Product distances (Higher is better)
  void ComputeDotProducts(const float* query, const uint8_t* codes, int n, float* out_scores) const;
  
  Status Save(std::FILE* out) const;
  Status Load(std::FILE* in);

  size_t code_size() const { return dim_; }

 private:
  int dim_;
  std::vector<float> min_;  // min per dim
  std::vector<float> diff_; // max - min per dim
  std::vector<float> step_; // diff / 255.0
};

}  // namespace pomai_search
