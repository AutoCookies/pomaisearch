#include "core/clustering/kmeans.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace pomai_search {

KMeans::KMeans(Config config, int dim) : config_(config), dim_(dim), rng_(config.seed) {}

Status KMeans::Train(const VectorStore& store, const std::vector<uint32_t>& indices) {
    if (indices.empty()) {
        return Status(StatusCode::kInvalidArgument, "No training data provided");
    }
    if (indices.size() < static_cast<size_t>(config_.k)) {
        return Status(StatusCode::kInvalidArgument, "Not enough points for K clusters");
    }

    int k = config_.k;
    int n = static_cast<int>(indices.size());
    centroids_.resize(k * dim_);

    // 1. Initialization (Random Sample)
    // Reservoir sampling or just shuffle indices if small?
    std::vector<uint32_t> sample_indices = indices;
    if (sample_indices.size() > static_cast<size_t>(k)) {
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng_);
        sample_indices.resize(k);
    }
    auto store_guard = store.AcquireRead();
    for (int i = 0; i < k; ++i) {
        // We need offset from indices
        // indices contains IDs or offsets? The API says 'indices' which likely mapped to offsets?
        // Wait, indices usually means valid IDs in VectorStore context.
        // Assuming indices are VectorStore offsets for simplicity here, or store has random access by ID?
        // VectorStore doesn't expose ID->Offset. It only has "Get(offset)".
        // So `indices` MUST be offsets.
        const float* src = store.Get(sample_indices[i], store_guard);
        std::copy(src, src + dim_, centroids_.data() + i * dim_);
    }

    // 2. Lloyd's Iterations
    std::vector<int> assignments(n);
    std::vector<float> new_centroids(k * dim_);
    std::vector<int> counts(k);

    DotFunc dot = GetDotFunc(CpuSupportsAvx2());

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Assignment Step
        int changed = 0;
        // Reset accumulators
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < n; ++i) {
            const float* vec = store.Get(indices[i], store_guard);
            int best_c = -1;
            float best_dist = -std::numeric_limits<float>::max(); // Dot product: max is best
            
            // If checking L2 distance (Euclidean): min is best.
            // Config has similarity.
            // If Dot/Cosine: Maximize dot.
            // If L2: Minimize L2.
            // WARNING: KMeans on Dot/Cosine usually requires Spherical KMeans (constantly normalize).
            // For now, implementing standard K-Means which minimizes L2 (Maximize -L2).
            // But if user wants Dot similarity search, should we cluster by Dot?
            // "Spherical K-Means" is standard for Cosine/Dot.
            // Let's assume Spherical if Cosine, Euclidean if L2?
            // SearchEngineConfig defaults to Dot. 
            // Let's implement generic Max Score assignment.
            
            // Find nearest centroid
            for (int c = 0; c < k; ++c) {
                float d = dot(vec, centroids_.data() + c * dim_, dim_);
                if (d > best_dist) {
                    best_dist = d;
                    best_c = c;
                }
            }
            
            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                changed++;
            }
            
            // Accumulate
            const float* c_ptr = vec;
            float* nc_ptr = new_centroids.data() + best_c * dim_;
            for (int d = 0; d < dim_; ++d) {
                nc_ptr[d] += c_ptr[d];
            }
            counts[best_c]++;
        }
        
        // Update Step
        for (int c = 0; c < k; ++c) {
            float* nc_ptr = new_centroids.data() + c * dim_;
            if (counts[c] > 0) {
                float scale = 1.0f / counts[c];
                for (int d = 0; d < dim_; ++d) {
                    nc_ptr[d] *= scale;
                }
                // Normalize if Spherical/Cosine
                if (config_.similarity == SearchEngineConfig::Similarity::Cosine ||
                    config_.similarity == SearchEngineConfig::Similarity::Dot) {
                     // Normalize centroid to unit length
                     float sum_sq = 0.0f;
                     for(int d=0; d<dim_; ++d) sum_sq += nc_ptr[d] * nc_ptr[d];
                     float norm = std::sqrt(sum_sq);
                     if (norm > 1e-6f) {
                         float inv = 1.0f / norm;
                         for(int d=0; d<dim_; ++d) nc_ptr[d] *= inv;
                     }
                }
            } else {
                 // Random re-init or leave as is?
                 // Leave as is from previous iteration (common heuristic)
                 const float* old = centroids_.data() + c * dim_;
                 std::copy(old, old + dim_, nc_ptr);
            }
        }
        
        // Swap
        centroids_ = new_centroids;
        
        if (changed == 0) break;
    }
    
    return Status::Ok();
}

int KMeans::AssignOne(const float* vector) const {
    DotFunc dot = GetDotFunc(CpuSupportsAvx2());
    int best_c = -1;
    float best_dist = -std::numeric_limits<float>::max();
    
    for (int c = 0; c < config_.k; ++c) {
        float d = dot(vector, centroids_.data() + c * dim_, dim_);
        if (d > best_dist) {
            best_dist = d;
            best_c = c;
        }
    }
    return best_c;
}

Status KMeans::Save(std::FILE* out) const {
    if(!fwrite(&dim_, sizeof(dim_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(!fwrite(&config_.k, sizeof(config_.k), 1, out)) return Status(StatusCode::kInternal, "write failed");
    // Write centroids
    uint64_t sz = centroids_.size();
    if(!fwrite(&sz, sizeof(sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(sz > 0 && fwrite(centroids_.data(), sizeof(float), sz, out) != sz) return Status(StatusCode::kInternal, "write failed");
    return Status::Ok();
}

Status KMeans::Load(std::FILE* in) {
    if(!fread(&dim_, sizeof(dim_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    int k = 0;
    if(!fread(&k, sizeof(k), 1, in)) return Status(StatusCode::kInternal, "read failed");
    // Ensure k matches config if config was set? Or overwrite config_.k?
    // We treat loaded data as authoritative for logic.
    config_.k = k;
    
    uint64_t sz = 0;
    if(!fread(&sz, sizeof(sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    centroids_.resize(sz);
    if(sz > 0 && fread(centroids_.data(), sizeof(float), sz, in) != sz) return Status(StatusCode::kInternal, "read failed");
    return Status::Ok();
}

}  // namespace pomai_search
