#include "core/vectorstore/vector_arena.h"
#include "tests/test_framework.h"

#include <cstdint>

namespace pomai_search::test {

POMAI_TEST(VectorArenaAppend) {
  VectorArena arena(3, 32, 2);
  float a[3] = {1.0f, 2.0f, 3.0f};
  float b[3] = {4.0f, 5.0f, 6.0f};
  arena.Append(a);
  arena.Append(b);
  const float* data = arena.Get(0);
  EXPECT_EQ(data[0], 1.0f);
  EXPECT_EQ(data[1], 2.0f);
  EXPECT_EQ(data[2], 3.0f);
  const float* data2 = arena.Get(3);
  EXPECT_EQ(data2[0], 4.0f);
  EXPECT_EQ(data2[1], 5.0f);
  EXPECT_EQ(data2[2], 6.0f);
  return true;
}

POMAI_TEST(VectorArenaAlignment) {
  VectorArena arena(4, 64, 1);
  float v[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  arena.Append(v);
  const float* data = arena.Get(0);
  auto addr = reinterpret_cast<std::uintptr_t>(data);
  EXPECT_TRUE(addr % 32 == 0);
  return true;
}

}  // namespace pomai_search::test
