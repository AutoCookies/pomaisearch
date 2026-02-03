#include "pomai_search/hash.h"
#include "tests/test_framework.h"

namespace pomai_search::test {

POMAI_TEST(StableHash64Deterministic) {
  EXPECT_EQ(StableHash64("alpha"), 0x8ac625bb85ed202bULL);
  EXPECT_EQ(StableHash64("pomai"), 0x9869323559e44f71ULL);
  EXPECT_EQ(StableHash64("key123"), 0x0a2eab35557fa774ULL);
  return true;
}

}  // namespace pomai_search::test
