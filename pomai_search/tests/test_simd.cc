#include <random>
#include <vector>

#include "simd/kernels.h"
#include "test_framework.h"

using pomai_search::CpuSupportsAvx2;
using pomai_search::DotAvx2;
using pomai_search::DotScalar;

POMAI_TEST(TestScalarDot) {
  std::vector<float> a{1.0f, 2.0f, 3.0f};
  std::vector<float> b{4.0f, 5.0f, 6.0f};
  EXPECT_NEAR(DotScalar(a.data(), b.data(), 3), 32.0f, 1e-5f);
  return true;
}

POMAI_TEST(TestAvx2MatchesScalar) {
  if (!CpuSupportsAvx2()) {
    return true;
  }
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(64);
  std::vector<float> b(64);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  float scalar = DotScalar(a.data(), b.data(), static_cast<int>(a.size()));
  float avx2 = DotAvx2(a.data(), b.data(), static_cast<int>(a.size()));
  EXPECT_NEAR(scalar, avx2, 1e-4f);
  return true;
}
