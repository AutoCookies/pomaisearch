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
    int topk;
    // Local scope restricts vector searches to shard 0. SearchByKey uses the key owner shard.
    enum class Scope { Local, Global };
    Scope scope;
    Metadata filter_equals;
    QueryOptions() : topk(0), scope(Scope::Global), filter_equals() {}
  };

  struct Stats {
    uint64_t num_points = 0;
    uint64_t num_deleted = 0;
    double last_query_ms_p50 = 0.0;
  };

  static StatusOr<std::unique_ptr<SearchEngine>> Open(const SearchEngineConfig& cfg);

  Status Upsert(std::string_view key, VectorView vec, Metadata meta = {},
                std::optional<std::chrono::milliseconds> ttl = std::nullopt);
  Status Delete(std::string_view key);
  StatusOr<bool> Exists(std::string_view key) const;

  StatusOr<std::vector<ResultItem>> Search(VectorView q, QueryOptions opt = {});
  StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key, QueryOptions opt = {});

  Stats GetStats() const;
  Status Close();

  ~SearchEngine();

 private:
  SearchEngine();
  Status Initialize(const SearchEngineConfig& cfg);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pomai_search
