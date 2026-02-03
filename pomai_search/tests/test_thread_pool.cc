#include "pomai_search/thread_pool.h"
#include "tests/test_framework.h"

#include <atomic>
#include <chrono>
#include <thread>

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
        std::this_thread::yield();
    }
    return 1;
  });
  EXPECT_TRUE(future_or.ok());
  // Task is running or queued.
  gate.store(1);
  future_or.value().get();
  return true;
}

POMAI_TEST(ThreadPoolBackpressure) {
    // 1 worker, max_queue_size = 1
    // This allows 1 task executing + 1 task queued. 3rd task should fail.
    ThreadPool pool(1, 1);
    
    std::atomic<bool> release{false};
    std::atomic<int> running{0};
    
    // Task 1: Blocks worker
    auto t1 = pool.Submit([&]() {
        running++;
        while (!release) std::this_thread::yield();
        return 1;
    });
    EXPECT_TRUE(t1.ok());
    
    // Wait for T1 to start running (occupy worker)
    while (running == 0) std::this_thread::yield();
    
    // Task 2: Fills queue (size 1)
    auto t2 = pool.Submit([&]() { return 2; });
    EXPECT_TRUE(t2.ok());
    
    // Task 3: Should be rejected
    auto t3 = pool.Submit([&]() { return 3; });
    
    bool rejected = !t3.ok() && t3.status().code() == StatusCode::kResourceExhausted;
    
    release = true;
    t1.value().get();
    t2.value().get();
    
    EXPECT_TRUE(rejected);
    return true;
}

}  // namespace pomai_search::test
