#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "pomai_search/thread_pool.h"

namespace pomai_search {

struct HttpRequest {
  std::string method;
  std::string path;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;
};

class HttpServer {
 public:
  using Handler = std::function<HttpResponse(const HttpRequest&)>;

  HttpServer();
  ~HttpServer();

  bool Start(int port, Handler handler, int worker_threads, size_t max_inflight,
             size_t max_body_bytes = 0, int timeout_ms = 0);
  void Wait();
  void Stop();

 private:
  void AcceptLoop();
  void HandleClient(int client_fd);

  std::atomic<bool> running_{false};
  int server_fd_ = -1;
  std::thread accept_thread_;
  Handler handler_;
  std::unique_ptr<ThreadPool> pool_;
  std::atomic<size_t> inflight_{0};
  size_t max_inflight_ = 0;
  size_t max_body_bytes_ = 0;
  int timeout_ms_ = 0;
};

}  // namespace pomai_search
