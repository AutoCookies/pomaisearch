#include "simd/kernels.h"

#include <immintrin.h>

namespace pomai_search {

float DotAvx2(const float* a, const float* b, int dim) {
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 7 < dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  float buffer[8];
  _mm256_storeu_ps(buffer, sum);
  float result = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

}  // namespace pomai_search
