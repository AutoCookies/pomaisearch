#include "pomai_search/search_engine.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <thread>

#include "pomai_search/logging.h"
#include "shard.h"
#include "thread_pool.h"

namespace pomai_search {

struct SearchEngine::Impl {
  SearchEngineConfig config;
  std::vector<std::unique_ptr<Shard>> shards;
  std::unique_ptr<ThreadPool> query_pool;
  DotFunc dot_func = DotScalar;
  bool avx2_enabled = false;
  mutable std::mutex stats_mutex;
  Stats stats;
};

SearchEngine::SearchEngine() = default;
SearchEngine::~SearchEngine() = default;

StatusOr<std::unique_ptr<SearchEngine>> SearchEngine::Open(const SearchEngineConfig& cfg) {
  auto engine = std::unique_ptr<SearchEngine>(new SearchEngine());
  Status status = engine->Initialize(cfg);
  if (!status.ok()) {
    return StatusOr<std::unique_ptr<SearchEngine>>(status);
  }
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

Status SearchEngine::Initialize(const SearchEngineConfig& cfg) {
  if (cfg.dim <= 0) {
    return Status(StatusCode::kInvalidArgument, "dim must be positive");
  }
  impl_ = std::make_unique<Impl>();
  impl_->config = cfg;
  if (impl_->config.num_shards <= 0) {
    impl_->config.num_shards = static_cast<int>(std::thread::hardware_concurrency());
    if (impl_->config.num_shards <= 0) {
      impl_->config.num_shards = 1;
    }
  }
  impl_->config.num_shards = std::max(1, std::min(impl_->config.num_shards, 64));
  if (impl_->config.query_threads <= 0) {
    impl_->config.query_threads = impl_->config.num_shards;
  }
  if (impl_->config.ingest_threads <= 0) {
    impl_->config.ingest_threads = impl_->config.num_shards;
  }
  impl_->config.topk_default = std::max(1, impl_->config.topk_default);
  impl_->config.memory_alignment = std::max<size_t>(16, impl_->config.memory_alignment);

  impl_->avx2_enabled = impl_->config.enable_avx2 && CpuSupportsAvx2();
  if (impl_->avx2_enabled) {
    impl_->dot_func = DotAvx2;
    POMAI_LOG_INFO("AVX2 enabled for dot product");
  } else {
    impl_->dot_func = DotScalar;
    POMAI_LOG_INFO("Using scalar dot product");
  }

  impl_->query_pool = std::make_unique<ThreadPool>(
      static_cast<size_t>(std::max(1, impl_->config.query_threads)));

  size_t reserve_vectors = impl_->config.max_points_per_shard;
  impl_->shards.reserve(impl_->config.num_shards);
  for (int i = 0; i < impl_->config.num_shards; ++i) {
    impl_->shards.emplace_back(std::make_unique<Shard>(impl_->config.dim,
                                                      impl_->config.memory_alignment,
                                                      reserve_vectors,
                                                      impl_->config.similarity,
                                                      impl_->dot_func,
                                                      impl_->config.max_points_per_shard));
  }
  return Status::Ok();
}

static size_t HashKey(std::string_view key) {
  return std::hash<std::string_view>{}(key);
}

static int ShardForKey(std::string_view key, int num_shards) {
  return static_cast<int>(HashKey(key) % static_cast<size_t>(num_shards));
}

Status SearchEngine::Upsert(std::string_view key, VectorView vec, Metadata meta,
                            std::optional<std::chrono::milliseconds> ttl) {
  if (key.empty()) {
    return Status(StatusCode::kInvalidArgument, "key empty");
  }
  auto expiry = ComputeExpiry(ttl);
  int shard_id = ShardForKey(key, impl_->config.num_shards);
  return impl_->shards[shard_id]->Upsert(key, vec, std::move(meta), expiry);
}

Status SearchEngine::Delete(std::string_view key) {
  if (key.empty()) {
    return Status(StatusCode::kInvalidArgument, "key empty");
  }
  int shard_id = ShardForKey(key, impl_->config.num_shards);
  return impl_->shards[shard_id]->Delete(key);
}

StatusOr<bool> SearchEngine::Exists(std::string_view key) const {
  if (key.empty()) {
    return Status(StatusCode::kInvalidArgument, "key empty");
  }
  int shard_id = ShardForKey(key, impl_->config.num_shards);
  return impl_->shards[shard_id]->Exists(key);
}

StatusOr<std::vector<ResultItem>> SearchEngine::Search(VectorView q, QueryOptions opt) {
  if (q.dim != impl_->config.dim || q.data == nullptr) {
    return Status(StatusCode::kInvalidArgument, "dimension mismatch");
  }
  int topk = opt.topk > 0 ? opt.topk : impl_->config.topk_default;
  float query_norm = 0.0f;
  if (impl_->config.similarity == SearchEngineConfig::Similarity::Cosine) {
    query_norm = std::sqrt(impl_->dot_func(q.data, q.data, q.dim));
  }
  auto start = std::chrono::steady_clock::now();
  std::vector<std::future<StatusOr<std::vector<ResultItem>>>> futures;
  futures.reserve(impl_->shards.size());
  for (const auto& shard : impl_->shards) {
    futures.emplace_back(impl_->query_pool->Submit([shard_ptr = shard.get(), q, topk, opt, query_norm]() {
      return shard_ptr->Search(q, topk, opt.filter_equals, query_norm);
    }));
  }
  std::vector<ResultItem> merged;
  for (auto& future : futures) {
    auto result = future.get();
    if (!result.ok()) {
      return result.status();
    }
    auto& items = result.value();
    merged.insert(merged.end(), items.begin(), items.end());
  }
  std::sort(merged.begin(), merged.end(), [](const ResultItem& a, const ResultItem& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.key < b.key;
  });
  if (static_cast<int>(merged.size()) > topk) {
    merged.resize(static_cast<size_t>(topk));
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    impl_->stats.last_query_ms_p50 = ms;
  }
  return StatusOr<std::vector<ResultItem>>(std::move(merged));
}

StatusOr<std::vector<ResultItem>> SearchEngine::SearchByKey(std::string_view key, QueryOptions opt) {
  if (key.empty()) {
    return Status(StatusCode::kInvalidArgument, "key empty");
  }
  int shard_id = ShardForKey(key, impl_->config.num_shards);
  auto vec_result = impl_->shards[shard_id]->GetVectorCopy(key);
  if (!vec_result.ok()) {
    return vec_result.status();
  }
  auto& vec = vec_result.value();
  VectorView view{vec.data(), impl_->config.dim};
  return Search(view, opt);
}

SearchEngine::Stats SearchEngine::GetStats() const {
  Stats stats;
  stats.last_query_ms_p50 = impl_->stats.last_query_ms_p50;
  uint64_t points = 0;
  uint64_t deleted = 0;
  for (const auto& shard : impl_->shards) {
    points += shard->num_points();
    deleted += shard->num_deleted();
  }
  stats.num_points = points;
  stats.num_deleted = deleted;
  return stats;
}

Status SearchEngine::Close() {
  impl_.reset();
  return Status::Ok();
}

}  // namespace pomai_search
