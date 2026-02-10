#include "core/quantization/scalar_quantizer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace pomai_search {

ScalarQuantizer::ScalarQuantizer(int dim) : dim_(dim) {}

void ScalarQuantizer::Train(const VectorStore& store, const std::vector<uint32_t>& indices) {
    if (indices.empty()) return;

    min_.assign(dim_, std::numeric_limits<float>::max());
    std::vector<float> max(dim_, std::numeric_limits<float>::lowest());

    auto store_guard = store.AcquireRead();
    for (uint32_t idx : indices) {
        const float* vec = store.Get(idx, store_guard);
        for (int d = 0; d < dim_; ++d) {
            float v = vec[d];
            if (v < min_[d]) min_[d] = v;
            if (v > max[d]) max[d] = v;
        }
    }

    diff_.resize(dim_);
    step_.resize(dim_);
    for (int d = 0; d < dim_; ++d) {
        diff_[d] = max[d] - min_[d];
        if (diff_[d] < 1e-6f) diff_[d] = 1e-6f; // Avoid div by zero
        step_[d] = diff_[d] / 255.0f;
    }
}

void ScalarQuantizer::Encode(const float* vec, uint8_t* out_codes) const {
    if (min_.empty()) return; // Not trained
    for (int d = 0; d < dim_; ++d) {
        // formula: code = (val - min) / step
        // clamp to [0, 255]
        float v = (vec[d] - min_[d]) / step_[d];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        out_codes[d] = static_cast<uint8_t>(std::round(v));
    }
}

void ScalarQuantizer::Decode(const uint8_t* codes, float* out_vec) const {
    if (min_.empty()) return;
    for (int d = 0; d < dim_; ++d) {
        out_vec[d] = min_[d] + codes[d] * step_[d];
    }
}

void ScalarQuantizer::ComputeDistancesL2(const float* query, const uint8_t* codes, int n, float* out_dists) const {
    // Optimization opportunity: AVX2
    // For now: Reference implementation
    for (int i = 0; i < n; ++i) {
        const uint8_t* c = codes + i * dim_;
        float sum_sq = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float reconstructed = min_[d] + c[d] * step_[d];
            float diff = query[d] - reconstructed;
            sum_sq += diff * diff;
        }
        out_dists[i] = sum_sq;
    }
}


void ScalarQuantizer::ComputeDotProducts(const float* query, const uint8_t* codes, int n, float* out_scores) const {
    // Optimization opportunity: Precompute query_quantized? No, min/step are dimension specific.
    // Optimization: Unroll loops.
    for (int i = 0; i < n; ++i) {
        const uint8_t* c = codes + i * dim_;
        float dot = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float reconstructed = min_[d] + c[d] * step_[d];
            dot += query[d] * reconstructed;
        }
        out_scores[i] = dot;
    }
}

Status ScalarQuantizer::Save(std::FILE* out) const {
    if(!fwrite(&dim_, sizeof(dim_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // min_
    uint64_t sz = min_.size();
    if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(sz > 0 && fwrite(min_.data(), sizeof(float), sz, out) != sz) return Status(StatusCode::kInternal, "write failed");
    
    // diff_
    sz = diff_.size();
    if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(sz > 0 && fwrite(diff_.data(), sizeof(float), sz, out) != sz) return Status(StatusCode::kInternal, "write failed");
    
    // step_
    sz = step_.size();
    if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(sz > 0 && fwrite(step_.data(), sizeof(float), sz, out) != sz) return Status(StatusCode::kInternal, "write failed");
    
    return Status::Ok();
}

Status ScalarQuantizer::Load(std::FILE* in) {
    if(!fread(&dim_, sizeof(dim_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    
    uint64_t sz = 0;
    if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    min_.resize(sz);
    if(sz > 0 && fread(min_.data(), sizeof(float), sz, in) != sz) return Status(StatusCode::kInternal, "read failed");
    
    if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    diff_.resize(sz);
    if(sz > 0 && fread(diff_.data(), sizeof(float), sz, in) != sz) return Status(StatusCode::kInternal, "read failed");
    
    if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    step_.resize(sz);
    if(sz > 0 && fread(step_.data(), sizeof(float), sz, in) != sz) return Status(StatusCode::kInternal, "read failed");
    
    return Status::Ok();
}

}  // namespace pomai_search
