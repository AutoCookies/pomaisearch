#include "core/kernels/kernels.h"

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace pomai_search {

namespace {

void Cpuid(int regs[4], int function_id, int subfunction_id) {
#if defined(_MSC_VER)
  __cpuidex(regs, function_id, subfunction_id);
#else
  unsigned int a = 0;
  unsigned int b = 0;
  unsigned int c = 0;
  unsigned int d = 0;
  __cpuid_count(function_id, subfunction_id, a, b, c, d);
  regs[0] = static_cast<int>(a);
  regs[1] = static_cast<int>(b);
  regs[2] = static_cast<int>(c);
  regs[3] = static_cast<int>(d);
#endif
}

uint64_t Xgetbv(unsigned int index) {
#if defined(_MSC_VER)
  return _xgetbv(index);
#else
  unsigned int eax = 0;
  unsigned int edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

}  // namespace

bool CpuSupportsAvx2() {
  int regs[4] = {0, 0, 0, 0};
  Cpuid(regs, 0, 0);
  const int max_id = regs[0];
  if (max_id < 7) {
    return false;
  }
  Cpuid(regs, 1, 0);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  if (!(osxsave && avx)) {
    return false;
  }
  uint64_t xcr0 = Xgetbv(0);
  const bool ymm_state = (xcr0 & 0x6) == 0x6;
  if (!ymm_state) {
    return false;
  }
  Cpuid(regs, 7, 0);
  const bool avx2 = (regs[1] & (1 << 5)) != 0;
  return avx2;
}

DotFunc GetDotFunc(bool enable_avx2) {
#if defined(__AVX2__)
  if (enable_avx2 && CpuSupportsAvx2()) {
    return &DotAvx2;
  }
#else
  (void)enable_avx2;
#endif
  return &DotScalar;
}

}  // namespace pomai_search
