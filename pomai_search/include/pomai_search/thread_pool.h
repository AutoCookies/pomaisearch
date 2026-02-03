#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "pomai_search/status.h"

namespace pomai_search {

class ThreadPool {
 public:
  explicit ThreadPool(size_t threads, size_t max_queue_size = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <typename F>
  auto Submit(F&& func) -> StatusOr<std::future<decltype(func())>> {
    using Result = decltype(func());
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(func));
    std::future<Result> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        return Status(StatusCode::kUnavailable, "thread pool stopping");
      }
      if (max_queue_size_ > 0 && tasks_.size() >= max_queue_size_) {
        return Status(StatusCode::kResourceExhausted, "thread pool saturated");
      }
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return StatusOr<std::future<Result>>(std::move(future));
  }

  Status Shutdown();
  size_t QueueSize() const;

 private:
  void WorkerLoop();

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
  size_t max_queue_size_ = 0;
};

}  // namespace pomai_search
