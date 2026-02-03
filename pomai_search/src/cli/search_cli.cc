#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai_search/net/json.h"

namespace pomai_search {

namespace {

struct CliConfig {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string command;
  std::string key;
  std::string text;
  std::vector<float> vector;
  std::unordered_map<std::string, std::string> metadata;
  int topk = 10;
  float alpha = 0.5f;
};

std::vector<float> ParseVector(std::string_view input, bool* ok) {
  std::vector<float> vec;
  std::string token;
  std::istringstream stream{std::string(input)};
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    char* end = nullptr;
    float value = std::strtof(token.c_str(), &end);
    if (end == token.c_str()) {
      *ok = false;
      return {};
    }
    vec.push_back(value);
  }
  *ok = !vec.empty();
  return vec;
}

std::unordered_map<std::string, std::string> ParseMetadata(std::string_view input) {
  std::unordered_map<std::string, std::string> map;
  std::string pair;
  std::istringstream stream{std::string(input)};
  while (std::getline(stream, pair, ',')) {
    auto pos = pair.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = pair.substr(0, pos);
    std::string value = pair.substr(pos + 1);
    if (!key.empty()) {
      map.emplace(std::move(key), std::move(value));
    }
  }
  return map;
}

bool ParseArgs(int argc, char** argv, CliConfig* cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        return "";
      }
      return argv[++i];
    };
    if (arg == "--host") {
      cfg->host = next();
    } else if (arg == "--port") {
      cfg->port = std::stoi(next());
    } else if (arg == "--command") {
      cfg->command = next();
    } else if (arg == "--key") {
      cfg->key = next();
    } else if (arg == "--vector") {
      bool ok = false;
      cfg->vector = ParseVector(next(), &ok);
      if (!ok) {
        return false;
      }
    } else if (arg == "--metadata") {
      cfg->metadata = ParseMetadata(next());
    } else if (arg == "--text") {
      cfg->text = next();
    } else if (arg == "--topk") {
      cfg->topk = std::stoi(next());
    } else if (arg == "--alpha") {
      cfg->alpha = std::stof(next());
    }
  }
  return !cfg->command.empty();
}

std::string BuildRequestBody(const CliConfig& cfg) {
  if (cfg.command == "upsert") {
    std::ostringstream oss;
    oss << "{\"key\":\"" << JsonEscape(cfg.key) << "\",\"vector\":[";
    for (size_t i = 0; i < cfg.vector.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << cfg.vector[i];
    }
    oss << "],\"metadata\":{";
    bool first = true;
    for (const auto& pair : cfg.metadata) {
      if (!first) {
        oss << ",";
      }
      first = false;
      oss << "\"" << JsonEscape(pair.first) << "\":\"" << JsonEscape(pair.second) << "\"";
    }
    oss << "}";
    if (!cfg.text.empty()) {
      oss << ",\"text\":\"" << JsonEscape(cfg.text) << "\"";
    }
    oss << "}";
    return oss.str();
  }
  if (cfg.command == "search") {
    std::ostringstream oss;
    oss << "{\"vector\":[";
    for (size_t i = 0; i < cfg.vector.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << cfg.vector[i];
    }
    oss << "],\"topk\":" << cfg.topk << "}";
    return oss.str();
  }
  if (cfg.command == "search_hybrid") {
    std::ostringstream oss;
    oss << "{\"topk\":" << cfg.topk << ",\"alpha\":" << cfg.alpha;
    if (!cfg.text.empty()) {
      oss << ",\"text_query\":\"" << JsonEscape(cfg.text) << "\"";
    }
    if (!cfg.vector.empty()) {
      oss << ",\"vector\":[";
      for (size_t i = 0; i < cfg.vector.size(); ++i) {
        if (i > 0) {
          oss << ",";
        }
        oss << cfg.vector[i];
      }
      oss << "]";
    }
    oss << "}";
    return oss.str();
  }
  if (cfg.command == "delete") {
    return "{\"key\":\"" + JsonEscape(cfg.key) + "\"}";
  }
  return "{}";
}

std::string RequestPath(const std::string& command) {
  if (command == "upsert") {
    return "/v1/upsert";
  }
  if (command == "search") {
    return "/v1/search";
  }
  if (command == "search_hybrid") {
    return "/v1/search_hybrid";
  }
  if (command == "delete") {
    return "/v1/delete";
  }
  return "/";
}

bool SendRequest(const CliConfig& cfg, const std::string& body) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(cfg.host.c_str(), std::to_string(cfg.port).c_str(), &hints, &res) != 0) {
    return false;
  }
  int sock = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (sock < 0) {
    freeaddrinfo(res);
    return false;
  }
  if (::connect(sock, res->ai_addr, res->ai_addrlen) < 0) {
    ::close(sock);
    freeaddrinfo(res);
    return false;
  }
  freeaddrinfo(res);
  std::ostringstream request;
  request << "POST " << RequestPath(cfg.command) << " HTTP/1.1\r\n";
  request << "Host: " << cfg.host << "\r\n";
  request << "Content-Type: application/json\r\n";
  request << "Content-Length: " << body.size() << "\r\n";
  request << "Connection: close\r\n\r\n";
  request << body;
  std::string request_str = request.str();
  ::send(sock, request_str.data(), request_str.size(), 0);
  std::string response;
  char buffer[4096];
  ssize_t n = 0;
  while ((n = ::recv(sock, buffer, sizeof(buffer), 0)) > 0) {
    response.append(buffer, buffer + n);
  }
  ::close(sock);
  std::cout << response << "\n";
  return true;
}

}  // namespace

}  // namespace pomai_search

int main(int argc, char** argv) {
  pomai_search::CliConfig cfg;
  if (!pomai_search::ParseArgs(argc, argv, &cfg)) {
    std::cerr << "Usage: pomai-search --command upsert|search|search_hybrid|delete [--host] [--port]\n";
    return 1;
  }
  if ((cfg.command == "upsert" || cfg.command == "delete") && cfg.key.empty()) {
    std::cerr << "key required\n";
    return 1;
  }
  if ((cfg.command == "upsert" || cfg.command == "search" || cfg.command == "search_hybrid") &&
      cfg.vector.empty() && cfg.command != "search_hybrid") {
    std::cerr << "vector required\n";
    return 1;
  }
  std::string body = pomai_search::BuildRequestBody(cfg);
  if (!pomai_search::SendRequest(cfg, body)) {
    std::cerr << "request failed\n";
    return 1;
  }
  return 0;
}
