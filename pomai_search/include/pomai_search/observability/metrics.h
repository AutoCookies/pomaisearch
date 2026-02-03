#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace pomai_search {

class Metrics {
 public:
  Metrics();

  void RecordQueryLatency(double ms);

  void IncrementQueries();
  void IncrementUpserts();
  void IncrementDeletes();

  std::string ToJson() const;

 private:
  static const std::vector<double>& Buckets();

  std::atomic<uint64_t> queries_total_{0};
  std::atomic<uint64_t> upserts_total_{0};
  std::atomic<uint64_t> deletes_total_{0};
  static constexpr size_t kHistogramSize = 12;
  std::array<std::atomic<uint64_t>, kHistogramSize> histogram_;
};

}  // namespace pomai_search
