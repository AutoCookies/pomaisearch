#include "pomai_search/index/flat_index.h"
#include "tests/test_framework.h"

#include <random>

namespace pomai_search::test {

POMAI_TEST(FlatIndexMatchesBruteForce) {
  constexpr int kDim = 4;
  FlatIndex index(kDim, SearchEngineConfig::Similarity::Dot, &DotScalar, 32, 0, 0);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data;
  for (uint32_t i = 0; i < 50; ++i) {
    std::vector<float> vec(kDim);
    for (int d = 0; d < kDim; ++d) {
      vec[d] = dist(rng);
    }
    data.push_back(vec);
    VectorView view{vec.data(), kDim};
    index.Upsert(i, view);
  }
  std::vector<float> query(kDim);
  for (int d = 0; d < kDim; ++d) {
    query[d] = dist(rng);
  }
  VectorView q{query.data(), kDim};
  auto result = index.Search(q, 5, Filter{});
  EXPECT_TRUE(result.ok());
  std::vector<Candidate> brute;
  for (uint32_t i = 0; i < data.size(); ++i) {
    float score = DotScalar(query.data(), data[i].data(), kDim);
    brute.push_back(Candidate{i, score});
  }
  std::sort(brute.begin(), brute.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  });
  for (size_t i = 0; i < result.value().size(); ++i) {
    EXPECT_EQ(result.value()[i].id, brute[i].id);
  }
  return true;
}

}  // namespace pomai_search::test
