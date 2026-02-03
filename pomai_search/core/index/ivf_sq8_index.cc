#include "core/index/ivf_sq8_index.h"
#include <algorithm>
#include <queue>
#include <cmath>

namespace pomai_search {

IvfSq8Index::IvfSq8Index(const VectorStore* store, int dim, Config config)
    : store_(store), dim_(dim), config_(config), 
      kmeans_({config.nlist, 20, 39, 42, config.similarity}, dim),
      quantizer_(dim) {
    if (config_.min_train_size == 0) {
        config_.min_train_size = static_cast<size_t>(config_.nlist * 39);
        if (config_.min_train_size < 100) config_.min_train_size = 100;
    }
}

Status IvfSq8Index::Upsert(uint32_t id, size_t offset, float /*norm*/) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    id_to_offset_[id] = offset;
    
    if (trained_) {
        const float* vec = store_->Get(offset);
        return AddToInvertedList(id, vec);
    } else {
        buffer_ids_.push_back(id);
        if (buffer_ids_.size() >= config_.min_train_size) {
            return TrainImpl(lock);
        }
    }
    return Status::Ok();
}

Status IvfSq8Index::Delete(uint32_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!id_to_offset_.count(id)) return Status(StatusCode::kNotFound, "not found");
    id_to_offset_.erase(id);
    return Status::Ok();
}

Status IvfSq8Index::Train() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (trained_) return Status::Ok();
    return TrainImpl(lock);
}

Status IvfSq8Index::TrainImpl(const std::unique_lock<std::shared_mutex>& /*lock*/) {
    std::vector<uint32_t> training_offsets;
    training_offsets.reserve(buffer_ids_.size());
    for(uint32_t id : buffer_ids_) {
        if (id_to_offset_.count(id)) {
            training_offsets.push_back(static_cast<uint32_t>(id_to_offset_[id]));
        }
    }
    
    if (training_offsets.empty()) return Status::Ok();
    
    // Train KMeans
    Status s = kmeans_.Train(*store_, training_offsets);
    if (!s.ok()) return s;
    
    // Train Quantizer
    quantizer_.Train(*store_, training_offsets);
    
    // Init lists
    lists_.resize(config_.nlist);
    codes_.resize(config_.nlist);
    
    // Re-ingest
    for(uint32_t id : buffer_ids_) {
         if (id_to_offset_.count(id)) {
             const float* vec = store_->Get(id_to_offset_[id]);
             AddToInvertedList(id, vec);
         }
    }
    buffer_ids_.clear();
    trained_ = true;
    return Status::Ok();
}

Status IvfSq8Index::AddToInvertedList(uint32_t id, const float* vec) {
    int centroid = kmeans_.AssignOne(vec);
    if (centroid < 0 || centroid >= static_cast<int>(lists_.size())) {
        return Status(StatusCode::kInternal, "KMeans error");
    }
    
    lists_[centroid].push_back(id);
    
    // Encode
    size_t old_size = codes_[centroid].size();
    codes_[centroid].resize(old_size + dim_);
    quantizer_.Encode(vec, codes_[centroid].data() + old_size);
    
    return Status::Ok();
}

StatusOr<std::vector<Candidate>> IvfSq8Index::Search(VectorView q, int topk, const Filter& /*f*/) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // If not trained, fallback to exact search over buffer (Use Flat logic basically)
    if (!trained_) {
        // Simple exact scan over buffer
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> pq;
        DotFunc dot = GetDotFunc(CpuSupportsAvx2());
        for(uint32_t id : buffer_ids_) {
             if (!id_to_offset_.count(id)) continue;
             const float* vec = store_->Get(id_to_offset_.at(id));
             float score = dot(q.data, vec, dim_);
             if (pq.size() < static_cast<size_t>(topk)) pq.push({score, id});
             else if (score > pq.top().first) { pq.pop(); pq.push({score, id}); }
        }
        std::vector<Candidate> res;
        while(!pq.empty()) { res.push_back({pq.top().second, pq.top().first}); pq.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

    // 1. Find Centroids (Coarse)
    int nprobe = std::min(config_.nprobe, config_.nlist);
    std::vector<std::pair<float, int>> centroid_scores;
    centroid_scores.reserve(config_.nlist);
    DotFunc dot = GetDotFunc(CpuSupportsAvx2());
    const float* centroids = kmeans_.centroids().data();
    for(int c=0; c<config_.nlist; ++c) {
         float score = dot(q.data, centroids + c * dim_, dim_);
         centroid_scores.push_back({score, c});
    }
    std::partial_sort(centroid_scores.begin(), centroid_scores.begin() + nprobe, centroid_scores.end(), std::greater<>());
    
    // 2. Scan Codes (Fine - Approximate)
    // We collect refine_k candidates
    int refine_k = static_cast<int>(topk * config_.refine_factor);
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> approx_pq;
    
    std::vector<float> dists_buffer; // Reuse for distance/score calculation
    
    for(int i=0; i<nprobe; ++i) {
        int c = centroid_scores[i].second;
        const auto& ids = lists_[c];
        const auto& codes = codes_[c];
        int count = static_cast<int>(ids.size());
        if (count == 0) continue;
        
        dists_buffer.resize(count);
        
        if (config_.similarity == SearchEngineConfig::Similarity::Dot) {
            quantizer_.ComputeDotProducts(q.data, codes.data(), count, dists_buffer.data());
        } else {
            // L2: we want min distance. Use negative distance for max-heap logic?
            // SearchEngine assumes higher score = better.
            // L2 score = -distance^2 usually.
            // ComputeDistancesL2 returns distance squared.
            quantizer_.ComputeDistancesL2(q.data, codes.data(), count, dists_buffer.data());
             for(int j=0; j<count; ++j) dists_buffer[j] = -dists_buffer[j];
        }
        
        for(int j=0; j<count; ++j) {
            uint32_t id = ids[j];
            // Filter deletions (lazy)
            if (!id_to_offset_.count(id)) continue;
            
            float score = dists_buffer[j];
            if (approx_pq.size() < static_cast<size_t>(refine_k)) {
                approx_pq.push({score, id});
            } else if (score > approx_pq.top().first) {
                approx_pq.pop();
                approx_pq.push({score, id});
            }
        }
    }
    
    // 3. Refinement (Exact Re-ranking)
    std::vector<uint32_t> candidates_to_refine;
    candidates_to_refine.reserve(approx_pq.size());
    while(!approx_pq.empty()) {
        candidates_to_refine.push_back(approx_pq.top().second);
        approx_pq.pop();
    }
    
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> exact_pq;
    
    for(uint32_t id : candidates_to_refine) {
        if (!id_to_offset_.count(id)) continue; 
        const float* vec = store_->Get(id_to_offset_.at(id));
        float score = dot(q.data, vec, dim_); // Exact score (Dot/L2 logic needed? DotFunc handles Dot)
        // If L2, DotFunc might not be correct? 
        // SearchEngine::Similarity::Dot uses DotFunc.
        // SearchEngine::Similarity::Cosine uses DotFunc (normalized).
        // If user selected L2, we don't support it in Config yet? 
        // Types.h: enum class Similarity { Dot, Cosine }. No L2.
        // So DotFunc is correct.
        
        if (exact_pq.size() < static_cast<size_t>(topk)) {
            exact_pq.push({score, id});
        } else if (score > exact_pq.top().first) {
            exact_pq.pop();
            exact_pq.push({score, id});
        }
    }
    
    std::vector<Candidate> result;
    while(!exact_pq.empty()) {
        result.push_back({exact_pq.top().second, exact_pq.top().first});
        exact_pq.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

IndexStats IvfSq8Index::GetStats() const {
    IndexStats stats;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    stats.num_points = id_to_offset_.size();
    return stats;
}

}  // namespace pomai_search
