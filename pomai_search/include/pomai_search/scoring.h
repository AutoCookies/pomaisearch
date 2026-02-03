#pragma once

#include <cmath>
#include <limits>

namespace pomai_search {

inline float SanitizeScore(float score) {
  if (std::isnan(score)) {
    return -std::numeric_limits<float>::infinity();
  }
  return score;
}

}  // namespace pomai_search
