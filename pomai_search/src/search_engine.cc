#include "pomai_search/search_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "pomai_search/hash.h"
#include "pomai_search/index/flat_index.h"
#include "pomai_search/index/hnsw_index.h"
#include "pomai_search/logging.h"
#include "pomai_search/observability/metrics.h"
#include "pomai_search/scoring.h"
#include "pomai_search/simd/kernels.h"
#include "pomai_search/thread_pool.h"
#include "pomai_search/vector_arena.h"

namespace pomai_search {

namespace {

std::vector<std::string> Tokenize(std::string_view text) {
  std::vector<std::string> tokens;
  std::string current;
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
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

class KeywordIndex {
 public:
  void Upsert(uint32_t id, const std::optional<std::string>& text) {
    Delete(id);
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

  std::vector<Candidate> Search(std::string_view query, int topk) const {
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
      float idf = std::log(1.0f + static_cast<float>(doc_count_) / static_cast<float>(df));
      for (const auto& entry : posting_it->second) {
        scores[entry.first] += static_cast<float>(entry.second) * idf;
      }
    }
    std::vector<Candidate> results;
    results.reserve(scores.size());
    for (const auto& pair : scores) {
      results.push_back(Candidate{pair.first, pair.second});
    }
    std::sort(results.begin(), results.end(), [](const Candidate& a, const Candidate& b) {
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

 private:
  uint64_t doc_count_ = 0;
  std::unordered_map<uint32_t, std::unordered_map<std::string, int>> doc_tokens_;
  std::unordered_map<std::string, std::unordered_map<uint32_t, int>> postings_;
  std::unordered_map<std::string, int> doc_freq_;
};

struct ShardDoc {
  std::string key;
  Metadata meta;
  size_t offset = 0;
  bool deleted = false;
  std::optional<std::chrono::steady_clock::time_point> expiry;
  std::optional<std::string> text;
};

class Shard {
 public:
  Shard(int dim, size_t alignment, size_t reserve_vectors, const SearchEngineConfig& cfg)
      : dim_(dim),
        arena_(dim, alignment, reserve_vectors),
        similarity_(cfg.similarity),
        dot_func_(GetDotFunc(cfg.enable_avx2)),
        max_points_(cfg.max_points_per_shard) {
    if (cfg.index_type == SearchEngineConfig::IndexType::Hnsw) {
      index_ = std::make_unique<HnswIndex>(dim, cfg.similarity, dot_func_, alignment,
                                           reserve_vectors, max_points_, cfg.hnsw_m,
                                           cfg.hnsw_ef_construction, cfg.hnsw_ef_search,
                                           cfg.hnsw_seed);
    } else {
      index_ = std::make_unique<FlatIndex>(dim, cfg.similarity, dot_func_, alignment,
                                           reserve_vectors, max_points_);
    }
  }

  Status Upsert(std::string_view key, VectorView vec, Metadata meta,
                std::optional<std::chrono::steady_clock::time_point> expiry,
                std::optional<std::string> text) {
    if (vec.dim != dim_ || vec.data == nullptr) {
      return Status(StatusCode::kInvalidArgument, "dimension mismatch");
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = key_to_id_.find(std::string(key));
    if (it != key_to_id_.end()) {
      uint32_t id = it->second;
      auto& doc = docs_[id];
      doc.meta = std::move(meta);
      doc.deleted = false;
      doc.expiry = expiry;
      doc.text = text;
      doc.offset = arena_.Append(vec.data);
      keyword_index_.Upsert(id, doc.text);
      return index_->Upsert(id, vec);
    }
    uint32_t id = static_cast<uint32_t>(docs_.size());
    if (max_points_ > 0 && docs_.size() >= max_points_) {
      return Status(StatusCode::kResourceExhausted, "shard at capacity");
    }
    ShardDoc doc;
    doc.key = std::string(key);
    doc.meta = std::move(meta);
    doc.deleted = false;
    doc.expiry = expiry;
    doc.text = text;
    doc.offset = arena_.Append(vec.data);
    key_to_id_.emplace(doc.key, id);
    docs_.push_back(doc);
    keyword_index_.Upsert(id, doc.text);
    return index_->Upsert(id, vec);
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
    bool exists = !doc.deleted && !IsExpired(doc.expiry);
    return StatusOr<bool>(exists);
  }

  StatusOr<std::vector<ResultItem>> Search(VectorView query, int topk, const Filter& filter) const {
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
      if (doc.deleted || IsExpired(doc.expiry)) {
        continue;
      }
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      ResultItem item;
      item.key = doc.key;
      item.score = SanitizeScore(cand.score);
      item.meta = doc.meta;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(), [](const ResultItem& a, const ResultItem& b) {
      float score_a = SanitizeScore(a.score);
      float score_b = SanitizeScore(b.score);
      if (score_a != score_b) {
        return score_a > score_b;
      }
      return a.key < b.key;
    });
    return StatusOr<std::vector<ResultItem>>(std::move(results));
  }

  StatusOr<std::vector<ResultItem>> SearchHybrid(std::string_view text, VectorView query,
                                                 bool has_vector, float alpha, int topk,
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
      if (doc.deleted || IsExpired(doc.expiry)) {
        continue;
      }
      if (!MetadataFilterMatch(filter, doc.meta)) {
        continue;
      }
      float score = alpha * pair.second.first + (1.0f - alpha) * pair.second.second;
      ResultItem item;
      item.key = doc.key;
      item.score = SanitizeScore(score);
      item.meta = doc.meta;
      results.push_back(std::move(item));
    }
    std::sort(results.begin(), results.end(), [](const ResultItem& a, const ResultItem& b) {
      float score_a = SanitizeScore(a.score);
      float score_b = SanitizeScore(b.score);
      if (score_a != score_b) {
        return score_a > score_b;
      }
      return a.key < b.key;
    });
    if (static_cast<int>(results.size()) > topk) {
      results.resize(static_cast<size_t>(topk));
    }
    return StatusOr<std::vector<ResultItem>>(std::move(results));
  }

  StatusOr<std::vector<float>> GetVectorCopy(std::string_view key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = key_to_id_.find(std::string(key));
    if (it == key_to_id_.end()) {
      return Status(StatusCode::kNotFound, "key not found");
    }
    const auto& doc = docs_[it->second];
    if (doc.deleted || IsExpired(doc.expiry)) {
      return Status(StatusCode::kNotFound, "key not found");
    }
    std::vector<float> vec(dim_);
    const float* data = arena_.Get(doc.offset);
    std::copy(data, data + dim_, vec.begin());
    return StatusOr<std::vector<float>>(std::move(vec));
  }

  IndexStats GetStats() const { return index_->GetStats(); }

 private:
  int dim_;
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, uint32_t> key_to_id_;
  std::vector<ShardDoc> docs_;
  VectorArena arena_;
  SearchEngineConfig::Similarity similarity_;
  DotFunc dot_func_;
  size_t max_points_;
  std::unique_ptr<Index> index_;
  KeywordIndex keyword_index_;
};

}  // namespace

struct SearchEngine::Impl {
  SearchEngineConfig cfg;
  std::vector<std::unique_ptr<Shard>> shards;
  std::unique_ptr<ThreadPool> query_pool;
  std::unique_ptr<ThreadPool> ingest_pool;
  Metrics metrics;
};

SearchEngine::SearchEngine() = default;

SearchEngine::~SearchEngine() {
  Close();
}

StatusOr<std::unique_ptr<SearchEngine>> SearchEngine::Open(const SearchEngineConfig& cfg) {
  auto engine = std::unique_ptr<SearchEngine>(new SearchEngine());
  Status status = engine->Initialize(cfg);
  if (!status.ok()) {
    return status;
  }
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

Status SearchEngine::Initialize(const SearchEngineConfig& cfg) {
  if (cfg.dim <= 0 || cfg.num_shards <= 0) {
    return Status(StatusCode::kInvalidArgument, "invalid config");
  }
  impl_ = std::make_unique<Impl>();
  impl_->cfg = cfg;
  impl_->shards.reserve(static_cast<size_t>(cfg.num_shards));
  size_t reserve_vectors = cfg.max_points_per_shard > 0 ? cfg.max_points_per_shard : 0;
  for (int i = 0; i < cfg.num_shards; ++i) {
    impl_->shards.emplace_back(
        std::make_unique<Shard>(cfg.dim, cfg.memory_alignment, reserve_vectors, cfg));
  }
  if (cfg.query_threads > 0) {
    impl_->query_pool = std::make_unique<ThreadPool>(static_cast<size_t>(cfg.query_threads));
  }
  if (cfg.ingest_threads > 0) {
    impl_->ingest_pool = std::make_unique<ThreadPool>(static_cast<size_t>(cfg.ingest_threads));
  }
  return Status::Ok();
}

Status SearchEngine::Upsert(std::string_view key, VectorView vec, Metadata meta,
                            std::optional<std::chrono::milliseconds> ttl,
                            std::optional<std::string> text) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  auto expiry = ComputeExpiry(ttl);
  impl_->metrics.IncrementUpserts();
  return impl_->shards[shard_index]->Upsert(key, vec, std::move(meta), expiry, std::move(text));
}

Status SearchEngine::Delete(std::string_view key) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  impl_->metrics.IncrementDeletes();
  return impl_->shards[shard_index]->Delete(key);
}

StatusOr<bool> SearchEngine::Exists(std::string_view key) const {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  return impl_->shards[shard_index]->Exists(key);
}

StatusOr<std::vector<ResultItem>> SearchEngine::Search(VectorView q, QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  impl_->metrics.IncrementQueries();
  auto start = std::chrono::steady_clock::now();
  std::vector<std::vector<ResultItem>> shard_results(impl_->shards.size());
  std::vector<std::future<StatusOr<std::vector<ResultItem>>>> futures;
  if (opt.scope == QueryOptions::Scope::Local) {
    auto result = impl_->shards.front()->Search(q, topk, opt.filter);
    if (!result.ok()) {
      return result.status();
    }
    auto end = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    impl_->metrics.RecordQueryLatency(ms);
    return result;
  }
  if (impl_->query_pool) {
    futures.reserve(impl_->shards.size());
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto submitted = impl_->query_pool->Submit([
          shard = impl_->shards[i].get(), q, topk, filter = opt.filter]() {
        return shard->Search(q, topk, filter);
      });
      if (!submitted.ok()) {
        return submitted.status();
      }
      futures.push_back(std::move(submitted.value()));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      auto result = futures[i].get();
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  } else {
    for (size_t i = 0; i < impl_->shards.size(); ++i) {
      auto result = impl_->shards[i]->Search(q, topk, opt.filter);
      if (!result.ok()) {
        return result.status();
      }
      shard_results[i] = std::move(result.value());
    }
  }
  std::vector<ResultItem> merged;
  for (const auto& shard_result : shard_results) {
    merged.insert(merged.end(), shard_result.begin(), shard_result.end());
  }
  std::sort(merged.begin(), merged.end(), [](const ResultItem& a, const ResultItem& b) {
    float score_a = SanitizeScore(a.score);
    float score_b = SanitizeScore(b.score);
    if (score_a != score_b) {
      return score_a > score_b;
    }
    return a.key < b.key;
  });
  if (static_cast<int>(merged.size()) > topk) {
    merged.resize(static_cast<size_t>(topk));
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<std::vector<ResultItem>>(std::move(merged));
}

StatusOr<std::vector<ResultItem>> SearchEngine::SearchByKey(std::string_view key, QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  impl_->metrics.IncrementQueries();
  size_t shard_index = StableHash64(key) % impl_->shards.size();
  auto vec_or = impl_->shards[shard_index]->GetVectorCopy(key);
  if (!vec_or.ok()) {
    return vec_or.status();
  }
  VectorView view{vec_or.value().data(), impl_->cfg.dim};
  if (opt.scope == QueryOptions::Scope::Local) {
    auto start = std::chrono::steady_clock::now();
    auto result = impl_->shards[shard_index]->Search(
        view, opt.topk > 0 ? opt.topk : impl_->cfg.topk_default, opt.filter);
    if (result.ok()) {
      auto end = std::chrono::steady_clock::now();
      double ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
      impl_->metrics.RecordQueryLatency(ms);
    }
    return result;
  }
  return Search(view, opt);
}

StatusOr<std::vector<ResultItem>> SearchEngine::SearchHybrid(const HybridQuery& query,
                                                             QueryOptions opt) {
  if (!impl_) {
    return Status(StatusCode::kInternal, "engine not initialized");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->cfg.topk_default;
  impl_->metrics.IncrementQueries();
  auto start = std::chrono::steady_clock::now();
  bool has_vector = query.vector_query.has_value();
  VectorView view{};
  if (has_vector) {
    if (static_cast<int>(query.vector_query->size()) != impl_->cfg.dim) {
      return Status(StatusCode::kInvalidArgument, "dimension mismatch");
    }
    view = VectorView{query.vector_query->data(), impl_->cfg.dim};
  }
  std::vector<ResultItem> merged;
  if (opt.scope == QueryOptions::Scope::Local) {
    auto result = impl_->shards.front()->SearchHybrid(query.text_query.value_or(""), view, has_vector,
                                                      query.alpha, topk, opt.filter);
    if (!result.ok()) {
      return result.status();
    }
    merged = std::move(result.value());
    auto end = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    impl_->metrics.RecordQueryLatency(ms);
  } else {
    std::vector<std::vector<ResultItem>> shard_results(impl_->shards.size());
    if (impl_->query_pool) {
      std::vector<std::future<StatusOr<std::vector<ResultItem>>>> futures;
      futures.reserve(impl_->shards.size());
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto submitted = impl_->query_pool->Submit([
            shard = impl_->shards[i].get(), text = query.text_query.value_or(""), view, has_vector,
            alpha = query.alpha, topk, filter = opt.filter]() {
          return shard->SearchHybrid(text, view, has_vector, alpha, topk, filter);
        });
        if (!submitted.ok()) {
          return submitted.status();
        }
        futures.push_back(std::move(submitted.value()));
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    } else {
      for (size_t i = 0; i < impl_->shards.size(); ++i) {
        auto result = impl_->shards[i]->SearchHybrid(query.text_query.value_or(""), view, has_vector,
                                                     query.alpha, topk, opt.filter);
        if (!result.ok()) {
          return result.status();
        }
        shard_results[i] = std::move(result.value());
      }
    }
    for (const auto& shard_result : shard_results) {
      merged.insert(merged.end(), shard_result.begin(), shard_result.end());
    }
    std::sort(merged.begin(), merged.end(), [](const ResultItem& a, const ResultItem& b) {
      float score_a = SanitizeScore(a.score);
      float score_b = SanitizeScore(b.score);
      if (score_a != score_b) {
        return score_a > score_b;
      }
      return a.key < b.key;
    });
    if (static_cast<int>(merged.size()) > topk) {
      merged.resize(static_cast<size_t>(topk));
    }
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  impl_->metrics.RecordQueryLatency(ms);
  return StatusOr<std::vector<ResultItem>>(std::move(merged));
}

SearchEngine::Stats SearchEngine::GetStats() const {
  Stats stats;
  if (!impl_) {
    return stats;
  }
  for (const auto& shard : impl_->shards) {
    auto shard_stats = shard->GetStats();
    stats.num_points += shard_stats.num_points;
    stats.num_deleted += shard_stats.num_deleted;
  }
  return stats;
}

std::string SearchEngine::MetricsJson() const {
  if (!impl_) {
    return "{}";
  }
  return impl_->metrics.ToJson();
}

Status SearchEngine::Close() {
  if (!impl_) {
    return Status::Ok();
  }
  impl_->query_pool.reset();
  impl_->ingest_pool.reset();
  impl_.reset();
  return Status::Ok();
}

}  // namespace pomai_search
