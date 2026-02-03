#include "pomai_search/observability/metrics.h"

#include <sstream>

namespace pomai_search {

Metrics::Metrics() {
  for (auto& bucket : histogram_) {
    bucket.store(0, std::memory_order_relaxed);
  }
}

const std::vector<double>& Metrics::Buckets() {
  static const std::vector<double> buckets = {1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000};
  return buckets;
}

void Metrics::RecordQueryLatency(double ms) {
  size_t idx = 0;
  const auto& buckets = Buckets();
  while (idx < buckets.size() && ms > buckets[idx]) {
    ++idx;
  }
  histogram_[idx].fetch_add(1, std::memory_order_relaxed);
}

void Metrics::IncrementQueries() {
  queries_total_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::IncrementUpserts() {
  upserts_total_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::IncrementDeletes() {
  deletes_total_.fetch_add(1, std::memory_order_relaxed);
}

std::string Metrics::ToJson() const {
  std::ostringstream out;
  out << "{\"queries_total\":" << queries_total_.load() << ",";
  out << "\"upserts_total\":" << upserts_total_.load() << ",";
  out << "\"deletes_total\":" << deletes_total_.load() << ",";
  out << "\"query_latency_ms\":{";
  const auto& buckets = Buckets();
  for (size_t i = 0; i < buckets.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    uint64_t count = histogram_[i].load();
    out << "\"le_" << buckets[i] << "\":" << count;
  }
  uint64_t overflow = histogram_.back().load();
  out << ",\"overflow\":" << overflow << "}}";
  return out.str();
}

}  // namespace pomai_search
