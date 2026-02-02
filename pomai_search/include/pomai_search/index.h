#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pomai_search/status.h"
#include "pomai_search/types.h"

namespace pomai_search {

struct IndexSearchParams {
  int topk = 10;
  float rerank_weight = 1.0f;
};

class VectorIndex {
 public:
  virtual ~VectorIndex() = default;
  virtual Status Add(std::string_view key, VectorView vec, const Metadata& meta) = 0;
  virtual Status Delete(std::string_view key) = 0;
  virtual StatusOr<std::vector<ResultItem>> Search(VectorView query, const IndexSearchParams& params,
                                                   const Metadata& filter) const = 0;
};

class SnapshotStore {
 public:
  virtual ~SnapshotStore() = default;
  virtual Status SaveSnapshot(std::string_view path) const = 0;
  virtual Status LoadSnapshot(std::string_view path) = 0;
};

class ScoreFusion {
 public:
  virtual ~ScoreFusion() = default;
  virtual float Fuse(float vector_score, float bm25_score, float weight) const = 0;
};

struct QueryPolicyDecision {
  bool use_cache = false;
  bool use_db = false;
  float rerank_weight = 1.0f;
};

class QueryPolicy {
 public:
  virtual ~QueryPolicy() = default;
  virtual QueryPolicyDecision Decide(std::string_view query) const = 0;
};

class MetricsSink {
 public:
  virtual ~MetricsSink() = default;
  virtual void IncrementCounter(std::string_view name, uint64_t value) = 0;
  virtual void RecordLatency(std::string_view name, std::chrono::milliseconds value) = 0;
};

}  // namespace pomai_search
