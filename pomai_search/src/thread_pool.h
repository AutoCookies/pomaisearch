#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace pomai_search {

class ThreadPool {
 public:
  explicit ThreadPool(size_t threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <typename F>
  auto Submit(F&& func) -> std::future<decltype(func())> {
    using Result = decltype(func());
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(func));
    std::future<Result> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

 private:
  void WorkerLoop();

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
};

}  // namespace pomai_search
