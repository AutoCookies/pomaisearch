#include "pomai_search/simd/kernels.h"
#include "tests/test_framework.h"

namespace pomai_search::test {

POMAI_TEST(SimdDotScalar) {
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float result = DotScalar(a, b, 4);
  EXPECT_NEAR(result, 70.0f, 1e-5f);
  return true;
}

POMAI_TEST(SimdDispatch) {
  DotFunc func = GetDotFunc(false);
  float a[2] = {1, 2};
  float b[2] = {3, 4};
  EXPECT_NEAR(func(a, b, 2), 11.0f, 1e-5f);
  return true;
}

}  // namespace pomai_search::test
