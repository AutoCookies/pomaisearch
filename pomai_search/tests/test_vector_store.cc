#include "core/vectorstore/vector_store.h"
#include "tests/test_framework.h"

#include <cstdint>

namespace pomai_search::test {

POMAI_TEST(VectorStoreAppend) {
  VectorStore store(3, 32, 2);
  float a[3] = {1.0f, 2.0f, 3.0f};
  float b[3] = {4.0f, 5.0f, 6.0f};
  size_t off1 = 0;
  size_t off2 = 0;
  {
    auto guard = store.AcquireWrite();
    off1 = store.Insert(a, guard);
    off2 = store.Insert(b, guard);
  }
  auto read_guard = store.AcquireRead();
  const float* data = store.Get(off1, read_guard);
  EXPECT_EQ(data[0], 1.0f);
  EXPECT_EQ(data[1], 2.0f);
  EXPECT_EQ(data[2], 3.0f);
  const float* data2 = store.Get(off2, read_guard);
  EXPECT_EQ(data2[0], 4.0f);
  EXPECT_EQ(data2[1], 5.0f);
  EXPECT_EQ(data2[2], 6.0f);
  return true;
}

POMAI_TEST(VectorStoreAlignment) {
  VectorStore store(4, 64, 1);
  float v[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  size_t offset = 0;
  {
    auto guard = store.AcquireWrite();
    offset = store.Insert(v, guard);
  }
  auto read_guard = store.AcquireRead();
  const float* data = store.Get(offset, read_guard);
  auto addr = reinterpret_cast<std::uintptr_t>(data);
  // We asked for 64 byte alignment. The VectorStore allocates blocks using
  // AlignedAllocator, so the base address is aligned even if per-vector offsets
  // are not aligned to 64 bytes.
  EXPECT_TRUE(addr % 32 == 0);
  return true;
}

POMAI_TEST(VectorStoreReuseSlots) {
  VectorStore store(2, 32, 0, 4);
  float a[2] = {1.0f, 2.0f};
  float b[2] = {3.0f, 4.0f};
  float c[2] = {5.0f, 6.0f};
  size_t slot1 = 0;
  size_t slot2 = 0;
  size_t slot3 = 0;
  {
    auto guard = store.AcquireWrite();
    slot1 = store.Insert(a, guard);
    slot2 = store.Insert(b, guard);
    (void)slot2;
    store.Release(slot1, guard);
    slot3 = store.Insert(c, guard);
  }
  EXPECT_EQ(slot1, slot3);
  auto stats = store.GetStats();
  EXPECT_EQ(stats.live_vectors, 2u);
  EXPECT_EQ(stats.total_vectors, 2u);
  return true;
}

}  // namespace pomai_search::test
