#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace pomai_search::test {

using TestFunc = bool (*)();

struct TestCase {
  const char* name;
  TestFunc func;
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct TestRegistrar {
  TestRegistrar(const char* name, TestFunc func) {
    Registry().push_back({name, func});
  }
};

#define POMAI_TEST(name)                                      \
  static bool name();                                         \
  static ::pomai_search::test::TestRegistrar name##_registrar(#name, name); \
  static bool name()

#define EXPECT_TRUE(cond)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::cerr << "Expectation failed: " #cond "\n";                        \
      return false;                                                           \
    }                                                                         \
  } while (0)

#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))
#define EXPECT_NEAR(a, b, eps) EXPECT_TRUE(std::fabs((a) - (b)) <= (eps))

inline int RunAll() {
  for (const auto& test : Registry()) {
    if (!test.func()) {
      std::cerr << "Test failed: " << test.name << "\n";
      return 1;
    }
  }
  std::cout << "All tests passed\n";
  return 0;
}

}  // namespace pomai_search::test
