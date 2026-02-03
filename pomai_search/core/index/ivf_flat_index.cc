#include "core/index/ivf_flat_index.h"
#include "core/kernels/kernels.h"

#include <algorithm>
#include <queue>
#include <iostream>

namespace pomai_search {

IvfFlatIndex::IvfFlatIndex(const VectorStore* store, int dim, Config config)
    : store_(store), dim_(dim), config_(config), 
      kmeans_({config.nlist, 20, 39, 42, config.similarity}, dim) {
    if (config_.min_train_size == 0) {
        config_.min_train_size = static_cast<size_t>(config_.nlist * 39);
        if (config_.min_train_size < 100) config_.min_train_size = 100;
    }
}

Status IvfFlatIndex::Upsert(uint32_t id, size_t offset, float /*norm*/) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Always update map
    id_to_offset_[id] = offset;
    
    if (trained_) {
        // Quantize and add to list
        const float* vec = store_->Get(offset);
        return AddToInvertedList(id, vec);
    } else {
        // Add to buffer
        buffer_ids_.push_back(id);
        
        // Check for auto-train
        if (buffer_ids_.size() >= config_.min_train_size) {
            return TrainImpl(lock);
        }
    }
    return Status::Ok();
}

Status IvfFlatIndex::Delete(uint32_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!id_to_offset_.count(id)) return Status(StatusCode::kNotFound, "not found");
    
    // We cannot easily remove from inverted list without scanning (slow).
    // Or we mark as deleted in a separate set?
    // HNSW has `deleted_` vector.
    // Here we remove from map.
    // `Search` will check map existence?
    // If we remove from map, we lose Offset.
    // But `Search` iterates lists which contain ID. Then looks up Offset in Map.
    // If Map lookup fails -> deleted.
    id_to_offset_.erase(id);
    // We leave ID in lists/buffer (lazy delete).
    return Status::Ok();
}

Status IvfFlatIndex::Train() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (trained_) return Status::Ok();
    return TrainImpl(lock);
}

Status IvfFlatIndex::TrainImpl(const std::unique_lock<std::shared_mutex>& /*lock*/) {
    // 1. Train K-Means
    // Convert buffer_ids_ to vector of offsets?
    // buffer content is IDs.
    // map maps ID->Offset.
    std::vector<uint32_t> training_offsets;
    training_offsets.reserve(buffer_ids_.size());
    for(uint32_t id : buffer_ids_) {
        // If deleted, it won't be in map?
        // Wait, lazy delete in buffer means we should check map.
        if (id_to_offset_.count(id)) {
            training_offsets.push_back(static_cast<uint32_t>(id_to_offset_[id])); // Warning cast size_t to uint32?
            // VectorStore Offset is size_t. KMeans takes uint32_t indices?
            // KMeans API: `Train(..., const std::vector<uint32_t>& indices)`
            // This implies offsets must fit in uint32_t?
            // VectorStore offsets are float-offsets.
            // If > 4B floats (16GB), we have issue.
            // Assuming < 4B for now (Phase 2 limitation).
        }
    }
    
    if (training_offsets.empty()) return Status::Ok(); // All deleted?
    
    Status s = kmeans_.Train(*store_, training_offsets);
    if (!s.ok()) return s;
    
    // 2. Initialize Lists
    lists_.resize(config_.nlist);
    
    // 3. Re-ingest buffer
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

Status IvfFlatIndex::AddToInvertedList(uint32_t id, const float* vec) {
    int centroid = kmeans_.AssignOne(vec);
    if (centroid < 0 || centroid >= static_cast<int>(lists_.size())) {
        return Status(StatusCode::kInternal, "KMeans assignment out of bounds");
    }
    // Optimization: we could lock individual lists, but we have global lock.
    lists_[centroid].push_back(id);
    return Status::Ok();
}

StatusOr<std::vector<Candidate>> IvfFlatIndex::Search(VectorView q, int topk, const Filter& /*f*/) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    DotFunc dot = GetDotFunc(CpuSupportsAvx2()); // Use kernels
    
    // Result container: Min Heap of {score, id}
    // Keeps track of the K best results.
    // We want to discard the smallest score (if higher is better).
    // top() returns largest element in MaxHeap, smallest in MinHeap.
    // So MinHeap is correct for evicting smallest.
    using ScoredID = std::pair<float, uint32_t>;
    std::priority_queue<ScoredID, std::vector<ScoredID>, std::greater<ScoredID>> top_k;
    
    auto process = [&](uint32_t id) {
        auto it = id_to_offset_.find(id);
        if (it == id_to_offset_.end()) return; // Deleted
        
        const float* vec = store_->Get(it->second);
        float score = dot(q.data, vec, dim_);
        
        if (top_k.size() < static_cast<size_t>(topk)) {
            top_k.push({score, id});
        } else if (score > top_k.top().first) {
            top_k.pop();
            top_k.push({score, id});
        }
    };

    if (!trained_) {
        // Brute force over buffer
        for(uint32_t id : buffer_ids_) process(id);
    } else {
        // 1. Find nearest centroids
        // We can use KMeans::Assign assigning to just 1?
        // We need 'nprobe' centroids.
        // KMeans class doesn't expose "Find N nearest centroids".
        // It has `AssignOne`.
        // We need manual scan over centroids.
        int nprobe = std::min(config_.nprobe, config_.nlist);
        std::vector<std::pair<float, int>> centroid_scores;
        centroid_scores.reserve(config_.nlist);
        const float* centroids = kmeans_.centroids().data();
        
        for(int c=0; c<config_.nlist; ++c) {
             float score = dot(q.data, centroids + c * dim_, dim_);
             centroid_scores.push_back({score, c});
        }
        
        // Sort descending
        // Partial sort/TopK
        std::partial_sort(centroid_scores.begin(), centroid_scores.begin() + nprobe, centroid_scores.end(), std::greater<>());
        
        // 2. Scan lists
        for(int i=0; i<nprobe; ++i) {
            int c = centroid_scores[i].second;
            for(uint32_t id : lists_[c]) {
                process(id);
            }
        }
    }
    
    std::vector<Candidate> result;
    while(!top_k.empty()) {
        result.push_back({top_k.top().second, top_k.top().first});
        top_k.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

IndexStats IvfFlatIndex::GetStats() const {
    IndexStats stats;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    stats.num_points = id_to_offset_.size();
    return stats;
}

Status IvfFlatIndex::Save(std::FILE* out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Config validation?
    if(!fwrite(&config_.nlist, sizeof(config_.nlist), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // State
    if(!fwrite(&trained_, sizeof(trained_), 1, out)) return Status(StatusCode::kInternal, "write failed");
    
    // Buffer
    uint64_t buf_sz = buffer_ids_.size();
    if(!fwrite(&buf_sz, sizeof(buf_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if(buf_sz > 0 && fwrite(buffer_ids_.data(), sizeof(uint32_t), buf_sz, out) != buf_sz) return Status(StatusCode::kInternal, "write failed");
    
    // Map
    uint64_t map_sz = id_to_offset_.size();
    if(!fwrite(&map_sz, sizeof(map_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
    for(const auto& pair : id_to_offset_) {
        // pair.first is uint32_t, pair.second is size_t
        if(!fwrite(&pair.first, sizeof(pair.first), 1, out)) return Status(StatusCode::kInternal, "write failed");
        if(!fwrite(&pair.second, sizeof(pair.second), 1, out)) return Status(StatusCode::kInternal, "write failed");
    }
    
    if (trained_) {
        // KMeans
        Status s = kmeans_.Save(out);
        if (!s.ok()) return s;
        
        // Lists
        uint64_t num_lists = lists_.size();
        if(!fwrite(&num_lists, sizeof(num_lists), 1, out)) return Status(StatusCode::kInternal, "write failed");
        for(const auto& lst : lists_) {
            uint64_t lsz = lst.size();
            if(!fwrite(&lsz, sizeof(lsz), 1, out)) return Status(StatusCode::kInternal, "write failed");
            if(lsz > 0 && fwrite(lst.data(), sizeof(uint32_t), lsz, out) != lsz) return Status(StatusCode::kInternal, "write failed");
        }
    }
    
    return Status::Ok();
}

Status IvfFlatIndex::Load(std::FILE* in) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    int nlist = 0;
    if(!fread(&nlist, sizeof(nlist), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if(nlist != config_.nlist) return Status(StatusCode::kInternal, "nlist mismatch");
    
    if(!fread(&trained_, sizeof(trained_), 1, in)) return Status(StatusCode::kInternal, "read failed");
    
    // Buffer
    uint64_t buf_sz = 0;
    if(!fread(&buf_sz, sizeof(buf_sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
    buffer_ids_.resize(buf_sz);
    if(buf_sz > 0 && fread(buffer_ids_.data(), sizeof(uint32_t), buf_sz, in) != buf_sz) return Status(StatusCode::kInternal, "read failed");
    
    // Map
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
        
        uint64_t num_lists = 0;
        if(!fread(&num_lists, sizeof(num_lists), 1, in)) return Status(StatusCode::kInternal, "read failed");
        lists_.resize(num_lists);
        for(size_t i=0; i<num_lists; ++i) {
            uint64_t lsz = 0;
            if(!fread(&lsz, sizeof(lsz), 1, in)) return Status(StatusCode::kInternal, "read failed");
            lists_[i].resize(lsz);
            if(lsz > 0 && fread(lists_[i].data(), sizeof(uint32_t), lsz, in) != lsz) return Status(StatusCode::kInternal, "read failed");
        }
    }
    
    return Status::Ok();
}

}  // namespace pomai_search
