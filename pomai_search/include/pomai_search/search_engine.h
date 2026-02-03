#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pomai_search/status.h"
#include "pomai_search/types.h"

namespace pomai_search {

class SearchEngine {
 public:
  struct QueryOptions {
    int topk = 0;
    enum class Scope { Local, Global };
    Scope scope = Scope::Global;
    Filter filter;
    QueryOptions() = default;
  };

  struct HybridQuery {
    std::optional<std::string> text_query;
    std::optional<std::vector<float>> vector_query;
    float alpha = 0.5f;
  };

  struct Stats {
    uint64_t num_points = 0;
    uint64_t num_deleted = 0;
  };

  static StatusOr<std::unique_ptr<SearchEngine>> Open(const SearchEngineConfig& cfg);

  Status Upsert(std::string_view key, VectorView vec, Metadata meta = {},
                std::optional<std::chrono::milliseconds> ttl = std::nullopt,
                std::optional<std::string> text = std::nullopt);
  Status Delete(std::string_view key);
  StatusOr<bool> Exists(std::string_view key) const;

  StatusOr<std::vector<ResultItem>> Search(VectorView q, QueryOptions opt);
  StatusOr<std::vector<ResultItem>> Search(VectorView q) { return Search(q, QueryOptions{}); }
  StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key, QueryOptions opt);
  StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key) {
    return SearchByKey(key, QueryOptions{});
  }
  StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query, QueryOptions opt);
  StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query) {
    return SearchHybrid(query, QueryOptions{});
  }
  StatusOr<SearchResponse> SearchWithExplain(VectorView q, QueryOptions opt, QueryPolicy policy);
  StatusOr<SearchResponse> SearchHybridWithExplain(const HybridQuery& query, QueryOptions opt,
                                                   QueryPolicy policy);

  Stats GetStats() const;
  std::string MetricsJson() const;
  Status Close();

  ~SearchEngine();

 private:
  SearchEngine();
  Status Initialize(const SearchEngineConfig& cfg);

  friend class SnapshotWriter;
  friend class SnapshotReader;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pomai_search
