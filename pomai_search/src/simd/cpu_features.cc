#include "simd/kernels.h"

#include <array>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace pomai_search {

bool CpuSupportsAvx2() {
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid_max(0, nullptr)) {
    return false;
  }
  __cpuid_count(1, 0, eax, ebx, ecx, edx);
  bool osxsave = (ecx & (1 << 27)) != 0;
  bool avx = (ecx & (1 << 28)) != 0;
  if (!osxsave || !avx) {
    return false;
  }
  unsigned int xcr0_low = 0, xcr0_high = 0;
  asm volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
  if ((xcr0_low & 0x6) != 0x6) {
    return false;
  }
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  bool avx2 = (ebx & (1 << 5)) != 0;
  return avx2;
#else
  return false;
#endif
}

}  // namespace pomai_search
