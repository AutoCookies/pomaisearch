#include "http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>

namespace pomai_search {

HttpServer::HttpServer(int port) : port_(port) {}

void HttpServer::AddHandler(const std::string& method, const std::string& path, HttpHandler handler) {
  handlers_[method + " " + path] = std::move(handler);
}

HttpResponse HttpServer::HandleRequest(const HttpRequest& request) {
  auto it = handlers_.find(request.method + " " + request.path);
  if (it == handlers_.end()) {
    return HttpResponse{404, "{\"ok\":false,\"message\":\"not found\"}"};
  }
  return it->second(request);
}

static bool ReadLine(int fd, std::string* out) {
  out->clear();
  char c = 0;
  while (true) {
    ssize_t n = read(fd, &c, 1);
    if (n <= 0) {
      return false;
    }
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      return true;
    }
    out->push_back(c);
  }
}

bool HttpServer::Start() {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    return false;
  }
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port_));

  if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(server_fd);
    return false;
  }
  if (listen(server_fd, 16) < 0) {
    close(server_fd);
    return false;
  }

  while (true) {
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
      continue;
    }
    std::string line;
    if (!ReadLine(client_fd, &line)) {
      close(client_fd);
      continue;
    }
    std::istringstream request_line(line);
    HttpRequest request;
    request_line >> request.method;
    request_line >> request.path;

    size_t content_length = 0;
    while (ReadLine(client_fd, &line)) {
      if (line.empty()) {
        break;
      }
      auto pos = line.find(":");
      if (pos == std::string::npos) {
        continue;
      }
      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1);
      if (key == "Content-Length") {
        content_length = static_cast<size_t>(std::stoul(value));
      }
    }
    if (content_length > 0) {
      request.body.resize(content_length);
      size_t read_total = 0;
      while (read_total < content_length) {
        ssize_t n = read(client_fd, &request.body[read_total], content_length - read_total);
        if (n <= 0) {
          break;
        }
        read_total += static_cast<size_t>(n);
      }
    }

    HttpResponse response = HandleRequest(request);
    std::ostringstream oss;
    oss << "HTTP/1.1 " << response.status << " OK\r\n";
    oss << "Content-Type: application/json\r\n";
    oss << "Content-Length: " << response.body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << response.body;
    std::string payload = oss.str();
    send(client_fd, payload.c_str(), payload.size(), 0);
    close(client_fd);
  }
  return true;
}

}  // namespace pomai_search
