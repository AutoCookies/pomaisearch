#include "pomai_search/vector_store.h"
#include "tests/test_framework.h"

#include <cstdint>

namespace pomai_search::test {

POMAI_TEST(VectorStoreAppend) {
  VectorStore store(3, 32, 2);
  float a[3] = {1.0f, 2.0f, 3.0f};
  float b[3] = {4.0f, 5.0f, 6.0f};
  size_t off1 = store.Append(a);
  size_t off2 = store.Append(b);
  const float* data = store.Get(off1);
  EXPECT_EQ(data[0], 1.0f);
  EXPECT_EQ(data[1], 2.0f);
  EXPECT_EQ(data[2], 3.0f);
  const float* data2 = store.Get(off2);
  EXPECT_EQ(data2[0], 4.0f);
  EXPECT_EQ(data2[1], 5.0f);
  EXPECT_EQ(data2[2], 6.0f);
  return true;
}

POMAI_TEST(VectorStoreAlignment) {
  VectorStore store(4, 64, 1);
  float v[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  size_t offset = store.Append(v);
  const float* data = store.Get(offset);
  auto addr = reinterpret_cast<std::uintptr_t>(data);
  // We asked for 64 byte alignment. AlignedAllocator usually respects this.
  // Note: VectorStore allocates `std::vector` using `AlignedAllocator`.
  // The beginning of the vector is aligned.
  // BUT `data_.data()` is aligned.
  // `offset` logic in VectorStore: `data_.data() + offset`.
  // `offset` is index (count * dim).
  // 4 floats = 16 bytes.
  // If count is 0, offset is 0 -> aligned.
  // If count is 1, offset is 4 -> addr + 16.
  // 16 is NOT 64-byte aligned (if base is 64).
  // So alignment check on `Get(offset)` for subsequent items depends on `dim`.
  // Test code checked `Get(0)`.
  // Wait, `test_vector_arena.cc` checked `Get(0)`.
  // `EXPECT_TRUE(addr % 32 == 0);`. (32 bytes?).
  // For `store(4, 64, 1)`, we expect base to be 64-byte aligned.
  // 64 % 32 == 0. So it should pass.
  EXPECT_TRUE(addr % 32 == 0);
  return true;
}

}  // namespace pomai_search::test
