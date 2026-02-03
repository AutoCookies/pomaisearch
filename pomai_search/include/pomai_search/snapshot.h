#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pomai_search/status.h"
#include "pomai_search/types.h"

namespace pomai_search {

struct SnapshotRecord {
  std::string key;
  std::vector<float> vector;
  Metadata meta;
  std::optional<std::string> text;
  std::optional<std::chrono::system_clock::time_point> expiry;
};

class SearchEngine;

class SnapshotWriter {
 public:
  static Status Write(const SearchEngine& engine, std::string_view path);
};

class SnapshotReader {
 public:
  static StatusOr<std::unique_ptr<SearchEngine>> Read(std::string_view path);
};

}  // namespace pomai_search
