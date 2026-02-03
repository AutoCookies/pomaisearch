#include "pomai_search/thread_pool.h"
#include "tests/test_framework.h"

#include <atomic>

namespace pomai_search::test {

POMAI_TEST(ThreadPoolExecutes) {
  ThreadPool pool(2);
  auto future_or = pool.Submit([]() { return 7; });
  EXPECT_TRUE(future_or.ok());
  EXPECT_EQ(future_or.value().get(), 7);
  return true;
}

POMAI_TEST(ThreadPoolRejectsAfterShutdown) {
  ThreadPool pool(1);
  pool.Shutdown();
  auto future_or = pool.Submit([]() { return 1; });
  EXPECT_TRUE(!future_or.ok());
  return true;
}

POMAI_TEST(ThreadPoolQueueSize) {
  ThreadPool pool(1);
  std::atomic<int> gate{0};
  auto future_or = pool.Submit([&gate]() {
    while (gate.load() == 0) {
    }
    return 1;
  });
  EXPECT_TRUE(future_or.ok());
  EXPECT_TRUE(pool.QueueSize() <= 1);
  gate.store(1);
  future_or.value().get();
  return true;
}

}  // namespace pomai_search::test
