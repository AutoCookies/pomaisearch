#pragma once

#include <functional>
#include <string>
#include <unordered_map>

namespace pomai_search {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

struct HttpResponse {
  int status = 200;
  std::string body;
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
 public:
  explicit HttpServer(int port);
  void AddHandler(const std::string& method, const std::string& path, HttpHandler handler);
  bool Start();

 private:
  int port_;
  std::unordered_map<std::string, HttpHandler> handlers_;
  HttpResponse HandleRequest(const HttpRequest& request);
};

}  // namespace pomai_search
