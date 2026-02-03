#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/index/flat_index.h"
#include "core/index/hnsw_index.h"
#include "core/index/ivf_flat_index.h"
#include "core/index/ivf_sq8_index.h"
#include "pomai_search/observability/metrics.h"
#include "pomai_search/scoring.h"
#include "pomai_search/search_engine.h"
#include "core/kernels/kernels.h"
#include "pomai_search/thread_pool.h"
#include "core/vectorstore/vector_store.h"
#include "core/serialize/snapshot.h"
#include "pomai_search/types.h"
#include "pomai_search/hash.h"

namespace pomai_search {

namespace {
// Helper functions
inline std::vector<std::string> Tokenize(std::string_view text) {
  std::vector<std::string> tokens;
  std::string current;
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      current.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    } else if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}
}  // namespace

class KeywordIndex {
 public:
  void Upsert(uint32_t id, const std::optional<std::string>& text) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    DeleteInternal(id);
    if (!text || text->empty()) {
      return;
    }
    std::vector<std::string> tokens = Tokenize(*text);
    if (tokens.empty()) {
      return;
    }
    std::unordered_map<std::string, int> term_freq;
    for (const auto& token : tokens) {
      ++term_freq[token];
    }
    doc_tokens_[id] = term_freq;
    ++doc_count_;
    for (const auto& pair : term_freq) {
      auto& posting = postings_[pair.first];
      posting[id] = pair.second;
    }
    for (const auto& pair : term_freq) {
      doc_freq_[pair.first] = static_cast<int>(postings_[pair.first].size());
    }
  }

  void Delete(uint32_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    DeleteInternal(id);
  }

  std::vector<Candidate> Search(std::string_view query, int topk) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::string> tokens = Tokenize(query);
    if (tokens.empty() || doc_count_ == 0) {
      return {};
    }
    std::unordered_map<uint32_t, float> scores;
    for (const auto& token : tokens) {
      auto posting_it = postings_.find(token);
      if (posting_it == postings_.end()) {
        continue;
      }
      int df = 1;
      auto df_it = doc_freq_.find(token);
      if (df_it != doc_freq_.end()) {
        df = std::max(1, df_it->second);
      }
      float idf =
          std::log(1.0f + static_cast<float>(doc_count_) / static_cast<float>(df));
      for (const auto& entry : posting_it->second) {
        scores[entry.first] += static_cast<float>(entry.second) * idf;
      }
    }
    std::vector<Candidate> results;
    results.reserve(scores.size());
    for (const auto& pair : scores) {
      results.push_back(Candidate{pair.first, pair.second});
    }
    std::sort(
        results.begin(), results.end(), [](const Candidate& a, const Candidate& b) {
          float score_a = SanitizeScore(a.score);
          float score_b = SanitizeScore(b.score);
          if (score_a != score_b) {
            return score_a > score_b;
          }
          return a.id < b.id;
        });
    if (static_cast<int>(results.size()) > topk) {
      results.resize(static_cast<size_t>(topk));
    }
    return results;
  }


  
  Status Save(std::FILE* out) const;
  Status Load(std::FILE* in);

 private:
  void DeleteInternal(uint32_t id) {
    auto it = doc_tokens_.find(id);
    if (it == doc_tokens_.end()) {
      return;
    }
    for (const auto& pair : it->second) {
      auto posting_it = postings_.find(pair.first);
      if (posting_it != postings_.end()) {
        posting_it->second.erase(id);
        doc_freq_[pair.first] = static_cast<int>(posting_it->second.size());
        if (posting_it->second.empty()) {
          postings_.erase(posting_it);
          doc_freq_.erase(pair.first);
        }
      }
    }
    doc_tokens_.erase(it);
    if (doc_count_ > 0) {
      --doc_count_;
    }
  }

  uint64_t doc_count_ = 0;
  std::unordered_map<uint32_t, std::unordered_map<std::string, int>> doc_tokens_;
  std::unordered_map<std::string, std::unordered_map<uint32_t, int>> postings_;
  std::unordered_map<std::string, int> doc_freq_;
  mutable std::shared_mutex mutex_;
};

struct ShardDoc {
  std::string key;
  Metadata meta;
  size_t offset = 0;
  bool deleted = false;
  std::optional<std::chrono::system_clock::time_point> expiry;
  std::optional<std::string> text;
};

class Shard {
 public:
  Shard(int dim, size_t alignment, size_t reserve_vectors,
        const pomai_search::SearchEngineConfig& cfg)
      : dim_(dim),
        store_(dim, alignment, reserve_vectors),
        similarity_(cfg.similarity),
        dot_func_(pomai_search::GetDotFunc(cfg.enable_avx2)),
        max_points_(cfg.max_points_per_shard) {
    if (cfg.index_type == pomai_search::SearchEngineConfig::IndexType::Hnsw) {
      uint32_t seed =
          static_cast<uint32_t>(cfg.global_seed ^ pomai_search::StableHash64("hnsw_seed"));
      index_ = std::make_unique<pomai_search::HnswIndex>(
          &store_, dim, cfg.similarity, dot_func_, max_points_,
          cfg.hnsw_m, cfg.hnsw_ef_construction, cfg.hnsw_ef_search, seed);
    } else if (cfg.index_type == pomai_search::SearchEngineConfig::IndexType::IvfFlat) {
      pomai_search::IvfFlatIndex::Config ivf_cfg;
      ivf_cfg.nlist = cfg.ivf_nlist;
      ivf_cfg.nprobe = cfg.ivf_nprobe;
      ivf_cfg.similarity = cfg.similarity;
      index_ = std::make_unique<pomai_search::IvfFlatIndex>(&store_, dim, ivf_cfg);
    } else if (cfg.index_type == pomai_search::SearchEngineConfig::IndexType::IvfSq8) {
      pomai_search::IvfSq8Index::Config sq_cfg;
      sq_cfg.nlist = cfg.ivf_nlist;
      sq_cfg.nprobe = cfg.ivf_nprobe;
      sq_cfg.similarity = cfg.similarity;
      // Refine factor default is 3.0, maybe expose in config?
      // For now hardcoded or use default.
      index_ = std::make_unique<pomai_search::IvfSq8Index>(&store_, dim, sq_cfg);
    } else {
      index_ = std::make_unique<pomai_search::FlatIndex>(&store_, dim, cfg.similarity, dot_func_,
                                           max_points_);
    }
  }

  Status Upsert(std::string_view key, VectorView vec, Metadata meta,
                std::optional<std::chrono::system_clock::time_point> expiry,
                std::optional<std::string> text) {
    if (vec.dim != dim_ || vec.data == nullptr) {
      return Status(StatusCode::kInvalidArgument, "dimension mismatch");
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    uint32_t id;
    bool exists = false;
    auto it = key_to_id_.find(std::string(key));
    if (it != key_to_id_.end()) {
      id = it->second;
      exists = true;
    } else {
      if (max_points_ > 0 && docs_.size() >= max_points_) {
        return Status(StatusCode::kResourceExhausted, "shard at capacity");
      }
      id = static_cast<uint32_t>(docs_.size());
    }

    // Append to VectorStore
    size_t offset = store_.Append(vec.data);
    
    // Compute norm
    float norm = 0.0f;
    float dot_prod = dot_func_(vec.data, vec.data, dim_);
    norm = std::sqrt(dot_prod);

    // Upsert to Index
    Status status = index_->Upsert(id, offset, norm);
    if (!status.ok()) {
      return status;
    }

    if (exists) {
      auto& doc = docs_[id];
      doc.meta = std::move(meta);
      doc.deleted = false;
      doc.expiry = expiry;
      doc.text = text;
      doc.offset = offset;
    } else {
      ShardDoc doc;
      doc.key = std::string(key);
      doc.meta = std::move(meta);
      doc.deleted = false;
      doc.expiry = expiry;
      doc.text = text;
      doc.offset = offset;
      key_to_id_.emplace(doc.key, id);
      docs_.push_back(doc);
    }
    
    // Upsert Keyword Index (thread-safe)
    keyword_index_.Upsert(id, text);
    
    return Status::Ok();
  }

  Status Delete(std::string_view key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = key_to_id_.find(std::string(key));
    if (it == key_to_id_.end()) {
      return Status(StatusCode::kNotFound, "key not found");
    }
    uint32_t id = it->second;
    docs_[id].deleted = true;
    keyword_index_.Delete(id);
    return index_->Delete(id);
  }

  StatusOr<bool> Exists(std::string_view key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = key_to_id_.find(std::string(key));
    if (it == key_to_id_.end()) {
      return Status(StatusCode::kNotFound, "key not found");
    }
    const auto& doc = docs_[it->second];
    bool expired = IsExpired(doc.expiry);
    bool exists = !doc.deleted && !expired;
    return StatusOr<bool>(exists);
  }

  StatusOr<std::vector<ResultItem>> Search(VectorView query, int topk,
                                           const Filter& filter) const {
    auto candidates = index_->Search(query, topk, filter);
    if (!candidates.ok()) {
      return candidates.status();
    }
    std::vector<ResultItem> results;
    results.reserve(candidates.value().size());
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& cand : candidates.value()) {
      if (cand.id >= docs_.size()) {
        continue;
      }
      const auto& doc = docs_[cand.id];
      if (doc.deleted) {
        continue;
      }
      if (IsExpired(doc.expiry)) {
          continue;
      }
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      ResultItem item;
      item.key = doc.key;
      item.score = SanitizeScore(cand.score);
      item.meta = doc.meta;
      item.internal_id = cand.id;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(),
              [](const ResultItem& a, const ResultItem& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.internal_id < b.internal_id;
              });
    return StatusOr<std::vector<ResultItem>>(std::move(results));
  }
  
  // Reusable strict weak ordering, but kept inline for simplicity in this extraction
  
  struct CandidateResult {
    uint32_t id = 0;
    std::string key;
    pomai_search::Metadata meta;
    float score = 0.0f;
  };

  StatusOr<std::vector<CandidateResult>> SearchCandidates(
      VectorView query, int topk, const Filter& filter) const {
    auto candidates = index_->Search(query, topk, filter);
    if (!candidates.ok()) {
      return candidates.status();
    }
    std::vector<CandidateResult> results;
    results.reserve(candidates.value().size());
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& cand : candidates.value()) {
      if (cand.id >= docs_.size()) {
        continue;
      }
      const auto& doc = docs_[cand.id];
      if (doc.deleted) continue;
      if (IsExpired(doc.expiry)) continue;
      
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      CandidateResult item;
      item.id = cand.id;
      item.key = doc.key;
      item.score = SanitizeScore(cand.score);
      item.meta = doc.meta;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(),
              [](const CandidateResult& a, const CandidateResult& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.id < b.id;
              });
    return StatusOr<std::vector<CandidateResult>>(std::move(results));
  }

  std::vector<Candidate> KeywordCandidates(std::string_view text, int topk) const {
    return keyword_index_.Search(text, topk);
  }

  StatusOr<std::vector<CandidateResult>> KeywordCandidatesWithDocs(std::string_view text, int topk,
                                                                   const Filter& filter) const {
    std::vector<Candidate> candidates = keyword_index_.Search(text, topk);
    std::vector<CandidateResult> results;
    results.reserve(candidates.size());
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& cand : candidates) {
      if (cand.id >= docs_.size()) {
        continue;
      }
      const auto& doc = docs_[cand.id];
      if (doc.deleted) continue;
      if (IsExpired(doc.expiry)) continue;
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      CandidateResult item;
      item.id = cand.id;
      item.key = doc.key;
      item.score = SanitizeScore(cand.score);
      item.meta = doc.meta;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(),
              [](const CandidateResult& a, const CandidateResult& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.id < b.id;
              });
    return StatusOr<std::vector<CandidateResult>>(std::move(results));
  }

  StatusOr<std::vector<ResultItem>> SearchHybrid(std::string_view text,
                                                 VectorView query, bool has_vector,
                                                 float alpha, int topk,
                                                 const Filter& filter) const {
    std::vector<Candidate> keyword_candidates;
    if (!text.empty()) {
      keyword_candidates = keyword_index_.Search(text, topk * 2);
    }
    std::vector<Candidate> vector_candidates;
    if (has_vector) {
      auto vector_results = index_->Search(query, topk * 2, filter);
      if (!vector_results.ok()) {
        return vector_results.status();
      }
      vector_candidates = vector_results.value();
    }
    std::unordered_map<uint32_t, std::pair<float, float>> scores;
    for (const auto& cand : keyword_candidates) {
      scores[cand.id].second = cand.score;
    }
    for (const auto& cand : vector_candidates) {
      scores[cand.id].first = cand.score;
    }
    std::vector<ResultItem> results;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& pair : scores) {
      uint32_t id = pair.first;
      if (id >= docs_.size()) {
        continue;
      }
      const auto& doc = docs_[id];
      if (doc.deleted) continue;
       if (IsExpired(doc.expiry)) continue;
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      float score = alpha * pair.second.first + (1.0f - alpha) * pair.second.second;
      ResultItem item;
      item.key = doc.key;
      item.score = SanitizeScore(score);
      item.meta = doc.meta;
      item.internal_id = id;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(),
              [](const ResultItem& a, const ResultItem& b) {
                float score_a = SanitizeScore(a.score);
                float score_b = SanitizeScore(b.score);
                if (score_a != score_b) {
                  return score_a > score_b;
                }
                if (a.key != b.key) {
                  return a.key < b.key;
                }
                return a.internal_id < b.internal_id;
              });
    if (static_cast<int>(results.size()) > topk) {
      results.resize(static_cast<size_t>(topk));
    }
    return StatusOr<std::vector<ResultItem>>(std::move(results));
  }

  void ExportRecords(std::vector<pomai_search::SnapshotRecord>* out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (size_t i = 0; i < docs_.size(); ++i) {
      const auto& doc = docs_[i];
      if (doc.deleted) continue;
      if (pomai_search::IsExpired(doc.expiry)) continue;
      pomai_search::SnapshotRecord record;
      record.key = doc.key;
      record.meta = doc.meta;
      record.expiry = doc.expiry;
      record.text = doc.text;
      record.vector.resize(static_cast<size_t>(dim_));
      
      const float* data = store_.Get(doc.offset);
      std::copy(data, data + dim_, record.vector.begin());
      out->push_back(std::move(record));
    }
  }

  StatusOr<std::vector<float>> GetVectorCopy(std::string_view key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = key_to_id_.find(std::string(key));
    if (it == key_to_id_.end()) {
      return Status(StatusCode::kNotFound, "key not found");
    }
    const auto& doc = docs_[it->second];
    if (doc.deleted) return Status(StatusCode::kNotFound, "key not found");
    if (IsExpired(doc.expiry)) return Status(StatusCode::kNotFound, "key not found");
    
    std::vector<float> vec(dim_);
    const float* data = store_.Get(doc.offset);
    std::copy(data, data + dim_, vec.begin());
    return StatusOr<std::vector<float>>(std::move(vec));
  }

  IndexStats GetStats() const { return index_->GetStats(); }

  Status Save(std::FILE* out) const;
  Status Load(std::FILE* in);

 private:
  int dim_;
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, uint32_t> key_to_id_;
  std::vector<ShardDoc> docs_;
  VectorStore store_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;
  std::unique_ptr<Index> index_;
  KeywordIndex keyword_index_;
};

struct SearchEngine::Impl {
  SearchEngineConfig cfg;
  std::vector<std::unique_ptr<Shard>> shards;
  std::unique_ptr<ThreadPool> query_pool;
  std::unique_ptr<ThreadPool> ingest_pool;
  Metrics metrics;
};

}  // namespace pomai_search
