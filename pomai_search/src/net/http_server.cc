#include "net/http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <strings.h>
#include <sstream>

#include "thread_pool.h"

namespace pomai_search {

namespace {

std::string Trim(std::string value) {
  size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

const char* ReasonPhrase(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 500:
      return "Internal Server Error";
    default:
      return "OK";
  }
}

}  // namespace

HttpServer::HttpServer() = default;

HttpServer::~HttpServer() {
  Stop();
}

bool HttpServer::Start(int port, Handler handler, int worker_threads) {
  if (running_.load()) {
    return false;
  }
  server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    return false;
  }
  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(server_fd_);
    server_fd_ = -1;
    return false;
  }
  if (listen(server_fd_, 64) < 0) {
    ::close(server_fd_);
    server_fd_ = -1;
    return false;
  }
  handler_ = std::move(handler);
  worker_threads_ = worker_threads > 0 ? worker_threads : 4;
  pool_ = std::make_unique<ThreadPool>(static_cast<size_t>(worker_threads_));
  running_.store(true);
  accept_thread_ = std::thread([this]() { AcceptLoop(); });
  return true;
}

void HttpServer::Wait() {
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
}

void HttpServer::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  if (server_fd_ >= 0) {
    ::shutdown(server_fd_, SHUT_RDWR);
    ::close(server_fd_);
    server_fd_ = -1;
  }
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  if (pool_) {
    pool_->Shutdown();
  }
}

void HttpServer::AcceptLoop() {
  while (running_.load()) {
    sockaddr_in client{};
    socklen_t len = sizeof(client);
    int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client), &len);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (running_.load()) {
        continue;
      }
      break;
    }
    auto submitted = pool_->Submit([this, client_fd]() { HandleClient(client_fd); });
    if (!submitted.ok()) {
      ::close(client_fd);
    }
  }
}

void HttpServer::HandleClient(int client_fd) {
  std::string request;
  char buffer[4096];
  ssize_t n = 0;
  while ((n = ::recv(client_fd, buffer, sizeof(buffer), 0)) > 0) {
    request.append(buffer, buffer + n);
    if (request.find("\r\n\r\n") != std::string::npos) {
      break;
    }
  }
  HttpRequest req;
  HttpResponse resp;
  bool parse_ok = true;
  if (n <= 0) {
    ::close(client_fd);
    return;
  }
  size_t header_end = request.find("\r\n\r\n");
  std::string headers_block = request.substr(0, header_end);
  std::istringstream header_stream(headers_block);
  std::string request_line;
  if (!std::getline(header_stream, request_line)) {
    parse_ok = false;
  } else {
    if (!request_line.empty() && request_line.back() == '\r') {
      request_line.pop_back();
    }
    std::istringstream line_stream(request_line);
    line_stream >> req.method >> req.path;
    if (req.method.empty() || req.path.empty()) {
      parse_ok = false;
    }
  }
  std::string line;
  size_t content_length = 0;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = Trim(line.substr(0, pos));
    std::string value = Trim(line.substr(pos + 1));
    if (strcasecmp(key.c_str(), "Content-Length") == 0) {
      try {
        content_length = static_cast<size_t>(std::stoul(value));
      } catch (...) {
        content_length = 0;
      }
    }
    req.headers.emplace(std::move(key), std::move(value));
  }
  std::string body;
  if (header_end != std::string::npos) {
    body = request.substr(header_end + 4);
  }
  while (body.size() < content_length) {
    n = ::recv(client_fd, buffer, sizeof(buffer), 0);
    if (n <= 0) {
      break;
    }
    body.append(buffer, buffer + n);
  }
  req.body = std::move(body);
  if (!parse_ok) {
    resp.status = 400;
    resp.body = "{\"error\":\"bad request\"}";
  } else {
    resp = handler_(req);
  }
  std::ostringstream response_stream;
  response_stream << "HTTP/1.1 " << resp.status << " " << ReasonPhrase(resp.status) << "\r\n";
  response_stream << "Content-Type: " << resp.content_type << "\r\n";
  response_stream << "Content-Length: " << resp.body.size() << "\r\n";
  response_stream << "Connection: close\r\n\r\n";
  response_stream << resp.body;
  std::string response = response_stream.str();
  ::send(client_fd, response.data(), response.size(), 0);
  ::close(client_fd);
}

}  // namespace pomai_search
