#pragma once

#include <cstdint>
#include <cstddef>
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

inline uint64_t StableHash64Bytes(const void* data, size_t size, uint64_t seed = 0) {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t hash = seed == 0 ? kOffsetBasis : seed;
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

inline uint64_t StableHash64Combine(uint64_t seed, std::string_view value) {
  return StableHash64Bytes(value.data(), value.size(), seed);
}

}  // namespace pomai_search
