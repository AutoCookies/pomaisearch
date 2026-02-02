#include "simd/kernels.h"

#include <immintrin.h>

namespace pomai_search {

float DotAvx2(const float* a, const float* b, int dim) {
  const int kStride = 8;
  int i = 0;
  __m256 sum = _mm256_setzero_ps();
  for (; i + kStride <= dim; i += kStride) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  alignas(32) float buf[8];
  _mm256_store_ps(buf, sum);
  float total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
  for (; i < dim; ++i) {
    total += a[i] * b[i];
  }
  return total;
}

}  // namespace pomai_search
