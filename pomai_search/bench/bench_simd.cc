#include "pomai_search/simd/kernels.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace pomai_search {
namespace {

double MeasureDot(DotFunc func, const std::vector<float>& a, const std::vector<float>& b,
                  int dim, int iters) {
  auto start = std::chrono::steady_clock::now();
  float sink = 0.0f;
  for (int i = 0; i < iters; ++i) {
    sink += func(a.data(), b.data(), dim);
  }
  auto end = std::chrono::steady_clock::now();
  double ns = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(end - start).count();
  if (sink == 0.123456f) {
    std::cerr << "sink " << sink << "\n";
  }
  return ns / static_cast<double>(iters);
}

std::vector<float> RandomVector(int dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> vec(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    vec[static_cast<size_t>(i)] = dist(rng);
  }
  return vec;
}

}  // namespace
}  // namespace pomai_search

int main() {
  using namespace pomai_search;
  std::mt19937 rng(7);
  std::vector<int> dims = {128, 256, 512, 768};
  bool avx2 = CpuSupportsAvx2();
  std::cout << "bench_simd\n";
  for (int dim : dims) {
    auto a = RandomVector(dim, rng);
    auto b = RandomVector(dim, rng);
    double scalar_ns = MeasureDot(DotScalar, a, b, dim, 20000);
    double avx2_ns = avx2 ? MeasureDot(DotAvx2, a, b, dim, 20000) : scalar_ns;
    double speedup = avx2 ? scalar_ns / avx2_ns : 1.0;
    std::cout << "dim=" << dim << " scalar_ns=" << scalar_ns << " avx2_ns=" << avx2_ns
              << " speedup=" << speedup << "\n";
  }
  return 0;
}
