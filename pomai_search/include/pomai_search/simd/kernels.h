#pragma once

#include <cstddef>

namespace pomai_search {

using DotFunc = float (*)(const float*, const float*, int);

float DotScalar(const float* a, const float* b, int dim);
float DotAvx2(const float* a, const float* b, int dim);

bool CpuSupportsAvx2();
DotFunc GetDotFunc(bool enable_avx2);

}  // namespace pomai_search
