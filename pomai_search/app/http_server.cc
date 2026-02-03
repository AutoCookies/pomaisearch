#include "app/http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstring>
#include <sstream>
#include <strings.h>

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
    case 429:
      return "Too Many Requests";
    case 408:
      return "Request Timeout";
    case 413:
      return "Payload Too Large";
    case 500:
      return "Internal Server Error";
    case 503:
      return "Service Unavailable";
    default:
      return "OK";
  }
}

}  // namespace

HttpServer::HttpServer() = default;

HttpServer::~HttpServer() {
  Stop();
}

bool HttpServer::Start(int port, Handler handler, int worker_threads, size_t max_inflight,
                       size_t max_body_bytes, int timeout_ms) {
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
  int threads = worker_threads > 0 ? worker_threads : 4;
  pool_ = std::make_unique<ThreadPool>(static_cast<size_t>(threads));
  max_inflight_ = max_inflight > 0 ? max_inflight : static_cast<size_t>(threads) * 4;
  max_body_bytes_ = max_body_bytes > 0 ? max_body_bytes : 1024 * 1024;
  timeout_ms_ = timeout_ms;
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
    if (inflight_.load() >= max_inflight_) {
      std::string response =
          "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      ::send(client_fd, response.data(), response.size(), 0);
      ::close(client_fd);
      continue;
    }
    inflight_.fetch_add(1);
    auto submitted = pool_->Submit([this, client_fd]() {
      HandleClient(client_fd);
      inflight_.fetch_sub(1);
    });
    if (!submitted.ok()) {
      inflight_.fetch_sub(1);
      ::close(client_fd);
    }
  }
}

void HttpServer::HandleClient(int client_fd) {
  if (timeout_ms_ > 0) {
    timeval tv{};
    tv.tv_sec = timeout_ms_ / 1000;
    tv.tv_usec = (timeout_ms_ % 1000) * 1000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
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
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      resp.status = 408;
      resp.body = "{\"ok\":false,\"code\":\"TIMEOUT\",\"message\":\"request timeout\"}";
      std::ostringstream response_stream;
      response_stream << "HTTP/1.1 " << resp.status << " " << ReasonPhrase(resp.status) << "\r\n";
      response_stream << "Content-Type: " << resp.content_type << "\r\n";
      response_stream << "Content-Length: " << resp.body.size() << "\r\n";
      response_stream << "Connection: close\r\n\r\n";
      response_stream << resp.body;
      std::string response = response_stream.str();
      ::send(client_fd, response.data(), response.size(), 0);
    }
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
  bool length_ok = true;
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
      size_t parsed = 0;
      auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
      if (result.ec != std::errc()) {
        length_ok = false;
      } else {
        content_length = parsed;
      }
    }
    req.headers.emplace(std::move(key), std::move(value));
  }
  if (!length_ok || content_length > max_body_bytes_) {
    resp.status = length_ok ? 413 : 400;
    if (resp.status == 413) {
      resp.body = "{\"ok\":false,\"code\":\"PAYLOAD_TOO_LARGE\",\"message\":\"payload too large\"}";
    } else {
      resp.body = "{\"ok\":false,\"code\":\"INVALID_ARGUMENT\",\"message\":\"invalid content length\"}";
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
    return;
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
    resp.body = "{\"ok\":false,\"code\":\"INVALID_ARGUMENT\",\"message\":\"bad request\"}";
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
