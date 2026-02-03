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
    std::shared_lock<std::shared_mutex> lock(mutex_);
    IndexStats stats;
    stats.num_points = id_to_offset_.size();
    stats.num_deleted = 0; // IVF doesn't track deletions separately
    return stats;
}

Status IvfSq8Index::Save(std::FILE* out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    if(!fwrite(&config_.nlist, sizeof(config_.nlist), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(!fwrite(&trained_, sizeof(trained_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    uint64_t buf_sz = buffer_ids_.size();
    if(!fwrite(&buf_sz, sizeof(buf_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(buf_sz > 0 && fwrite(buffer_ids_.data(), sizeof(uint32_t), buf_sz, out) != buf_sz) return Status(StatusCode::kInternal, "write failed");
    
    uint64_t map_sz = id_to_offset_.size();
    if(!fwrite(&map_sz, sizeof(map_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    for(const auto& pair : id_to_offset_) {
        if(!fwrite(&pair.first, sizeof(pair.first), 1, out)) return Status(StatusCode::kInternal, "write failed");
        if(!fwrite(&pair.second, sizeof(pair.second), 1, out)) return Status(StatusCode::kInternal, "write failed");
    }
    
    if (trained_) {
        Status s = kmeans_.Save(out);
        if (!s.ok()) return s;
        
        s = quantizer_.Save(out);
        if (!s.ok()) return s;
        
        uint64_t num_lists = lists_.size();
        if(!fwrite(&num_lists, sizeof(num_lists), 1, out)) return Status(StatusCode::kInternal, "write failed");
        for(const auto& lst : lists_) {
            uint64_t lsz = lst.size();
            if(!fwrite(&lsz, sizeof(lsz), 1, out)) return Status(StatusCode::kInternal, "write failed");
            if(lsz > 0 && fwrite(lst.data(), sizeof(uint32_t), lsz, out) != lsz) return Status(StatusCode::kInternal, "write failed");
        }
        
        // Codes
        uint64_t num_codes = codes_.size();
        if(!fwrite(&num_codes, sizeof(num_codes), 1, out)) return Status(StatusCode::kInternal, "write failed");
        for(const auto& code : codes_) {
            uint64_t csz = code.size();
            if(!fwrite(&csz, sizeof(csz), 1, out)) return Status(StatusCode::kInternal, "write failed");
            if(csz > 0 && fwrite(code.data(), sizeof(uint8_t), csz, out) != csz) return Status(StatusCode::kInternal, "write failed");
        }
    }
    return Status::Ok();
}

Status IvfSq8Index::Load(std::FILE* in) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    int nlist = 0;
    if(!fread(&nlist, sizeof(nlist), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(nlist != config_.nlist) return Status(StatusCode::kInternal, "nlist mismatch");
    
    if(!fread(&trained_, sizeof(trained_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    
    uint64_t buf_sz = 0;
    if(!fread(&buf_sz, sizeof(buf_sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    buffer_ids_.resize(buf_sz);
    if(buf_sz > 0 && fread(buffer_ids_.data(), sizeof(uint32_t), buf_sz, in) != buf_sz) return Status(StatusCode::kInternal, "read failed");
    
    uint64_t map_sz = 0;
    if(!fread(&map_sz, sizeof(map_sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    id_to_offset_.clear();
    id_to_offset_.reserve(map_sz);
    for(size_t i=0; i<map_sz; ++i) {
        uint32_t k;
        size_t v;
        if(!fread(&k, sizeof(k), 1, in)) return Status(StatusCode::kInternal, "read failed");
        if(!fread(&v, sizeof(v), 1, in)) return Status(StatusCode::kInternal, "read failed");
        id_to_offset_[k] = v;
    }
    
    if (trained_) {
        Status s = kmeans_.Load(in);
        if (!s.ok()) return s;
        
        s = quantizer_.Load(in);
        if (!s.ok()) return s;
        
        uint64_t num_lists = 0;
        if(!fread(&num_lists, sizeof(num_lists), 1, in)) return Status(StatusCode::kInternal, "read failed");
        lists_.resize(num_lists);
        for(size_t i=0; i<num_lists; ++i) {
            uint64_t lsz = 0;
            if(!fread(&lsz, sizeof(lsz), 1, in)) return Status(StatusCode::kInternal, "read failed");
            lists_[i].resize(lsz);
            if(lsz > 0 && fread(lists_[i].data(), sizeof(uint32_t), lsz, in) != lsz) return Status(StatusCode::kInternal, "read failed");
        }
        
        uint64_t num_codes = 0;
        if(!fread(&num_codes, sizeof(num_codes), 1, in)) return Status(StatusCode::kInternal, "read failed");
        codes_.resize(num_codes);
        for(size_t i=0; i<num_codes; ++i) {
            uint64_t csz = 0;
            if(!fread(&csz, sizeof(csz), 1, in)) return Status(StatusCode::kInternal, "read failed");
            codes_[i].resize(csz);
            if(csz > 0 && fread(codes_[i].data(), sizeof(uint8_t), csz, in) != csz) return Status(StatusCode::kInternal, "read failed");
        }
    }
    
    return Status::Ok();
}

}  // namespace pomai_search
