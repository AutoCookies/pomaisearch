#include "core/kernels/kernels.h"

namespace pomai_search {

float DotScalar(const float* a, const float* b, int dim) {
  float sum = 0.0f;
  for (int i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace pomai_search
