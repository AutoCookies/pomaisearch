#pragma once

#include <cstdint>
#include <string_view>

namespace pomai_search {

inline uint64_t StableHash64(std::string_view key) {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t hash = kOffsetBasis;
  for (char c : key) {
    hash ^= static_cast<uint8_t>(c);
    hash *= kPrime;
  }
  return hash;
}

}  // namespace pomai_search
