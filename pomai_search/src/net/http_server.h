#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <memory>

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

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  bool Start(int port, Handler handler, int worker_threads);
  void Wait();
  void Stop();

 private:
  void AcceptLoop();
  void HandleClient(int client_fd);

  int server_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  Handler handler_;
  int worker_threads_ = 0;
  std::unique_ptr<class ThreadPool> pool_;
};

}  // namespace pomai_search
