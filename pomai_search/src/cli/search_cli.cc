#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "net/json.h"

namespace {

struct CliOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  int dim = 0;
};

std::vector<float> ParseVector(const std::string& text) {
  std::vector<float> values;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    values.push_back(std::stof(token));
  }
  return values;
}

bool SendRequest(const CliOptions& options, const std::string& method, const std::string& path,
                 const std::string& body, std::string* response_body) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  std::string port_str = std::to_string(options.port);
  if (getaddrinfo(options.host.c_str(), port_str.c_str(), &hints, &result) != 0) {
    return false;
  }
  int sock = -1;
  for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    sock = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock == -1) {
      continue;
    }
    if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
      break;
    }
    ::close(sock);
    sock = -1;
  }
  freeaddrinfo(result);
  if (sock == -1) {
    return false;
  }
  std::ostringstream req;
  req << method << " " << path << " HTTP/1.1\r\n";
  req << "Host: " << options.host << "\r\n";
  if (!body.empty()) {
    req << "Content-Type: application/json\r\n";
  }
  req << "Content-Length: " << body.size() << "\r\n";
  req << "Connection: close\r\n\r\n";
  req << body;
  std::string req_str = req.str();
  ::send(sock, req_str.data(), req_str.size(), 0);
  std::string response;
  char buffer[4096];
  ssize_t n = 0;
  while ((n = ::recv(sock, buffer, sizeof(buffer), 0)) > 0) {
    response.append(buffer, buffer + n);
  }
  ::close(sock);
  size_t header_end = response.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    *response_body = response;
  } else {
    *response_body = response.substr(header_end + 4);
  }
  return true;
}

void PrintUsage() {
  std::cout << "Usage: pomai-search [--host HOST] [--port PORT] [--dim DIM] <command> [options]\n"
            << "Commands:\n"
            << "  upsert --key KEY --vec 0.1,0.2 --tag TAG --source SRC --lang LANG\n"
            << "  delete --key KEY\n"
            << "  search --vec 0.1,0.2 [--topk K] [--scope local|global]\n"
            << "  search_by_key --key KEY [--topk K] [--scope local|global]\n"
            << "  stats\n";
}

}  // namespace

int main(int argc, char** argv) {
  CliOptions options;
  if (argc < 2) {
    PrintUsage();
    return 1;
  }
  int idx = 1;
  for (; idx < argc; ++idx) {
    std::string arg = argv[idx];
    if (arg == "--host" && idx + 1 < argc) {
      options.host = argv[++idx];
    } else if (arg == "--port" && idx + 1 < argc) {
      options.port = std::atoi(argv[++idx]);
    } else if (arg == "--dim" && idx + 1 < argc) {
      options.dim = std::atoi(argv[++idx]);
    } else {
      break;
    }
  }
  if (idx >= argc) {
    PrintUsage();
    return 1;
  }
  std::string command = argv[idx++];
  std::string key;
  std::string vec_text;
  std::string tag;
  std::string source;
  std::string lang;
  int topk = 0;
  std::string scope;
  for (; idx < argc; ++idx) {
    std::string arg = argv[idx];
    if (arg == "--key" && idx + 1 < argc) {
      key = argv[++idx];
    } else if (arg == "--vec" && idx + 1 < argc) {
      vec_text = argv[++idx];
    } else if (arg == "--tag" && idx + 1 < argc) {
      tag = argv[++idx];
    } else if (arg == "--source" && idx + 1 < argc) {
      source = argv[++idx];
    } else if (arg == "--lang" && idx + 1 < argc) {
      lang = argv[++idx];
    } else if (arg == "--topk" && idx + 1 < argc) {
      topk = std::atoi(argv[++idx]);
    } else if (arg == "--scope" && idx + 1 < argc) {
      scope = argv[++idx];
    }
  }

  std::string body;
  std::string path;
  if (command == "upsert") {
    if (key.empty() || vec_text.empty()) {
      std::cerr << "upsert requires --key and --vec" << std::endl;
      return 1;
    }
    auto vec = ParseVector(vec_text);
    if (options.dim > 0 && static_cast<int>(vec.size()) != options.dim) {
      std::cerr << "vector dimension mismatch" << std::endl;
      return 1;
    }
    std::ostringstream oss;
    oss << "{\"key\":\"" << pomai_search::JsonEscape(key) << "\",\"vector\":[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << vec[i];
    }
    oss << "]";
    if (!tag.empty() || !source.empty() || !lang.empty()) {
      oss << ",\"metadata\":{";
      bool first = true;
      if (!tag.empty()) {
        oss << "\"tag\":\"" << pomai_search::JsonEscape(tag) << "\"";
        first = false;
      }
      if (!source.empty()) {
        if (!first) {
          oss << ",";
        }
        oss << "\"source\":\"" << pomai_search::JsonEscape(source) << "\"";
        first = false;
      }
      if (!lang.empty()) {
        if (!first) {
          oss << ",";
        }
        oss << "\"lang\":\"" << pomai_search::JsonEscape(lang) << "\"";
      }
      oss << "}";
    }
    oss << "}";
    body = oss.str();
    path = "/v1/upsert";
  } else if (command == "delete") {
    if (key.empty()) {
      std::cerr << "delete requires --key" << std::endl;
      return 1;
    }
    body = "{\"key\":\"" + pomai_search::JsonEscape(key) + "\"}";
    path = "/v1/delete";
  } else if (command == "search") {
    if (vec_text.empty()) {
      std::cerr << "search requires --vec" << std::endl;
      return 1;
    }
    auto vec = ParseVector(vec_text);
    if (options.dim > 0 && static_cast<int>(vec.size()) != options.dim) {
      std::cerr << "vector dimension mismatch" << std::endl;
      return 1;
    }
    std::ostringstream oss;
    oss << "{\"vector\":[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << vec[i];
    }
    oss << "]";
    if (topk > 0) {
      oss << ",\"topk\":" << topk;
    }
    if (!scope.empty()) {
      oss << ",\"scope\":\"" << pomai_search::JsonEscape(scope) << "\"";
    }
    oss << "}";
    body = oss.str();
    path = "/v1/search";
  } else if (command == "search_by_key") {
    if (key.empty()) {
      std::cerr << "search_by_key requires --key" << std::endl;
      return 1;
    }
    std::ostringstream oss;
    oss << "{\"key\":\"" << pomai_search::JsonEscape(key) << "\"";
    if (topk > 0) {
      oss << ",\"topk\":" << topk;
    }
    if (!scope.empty()) {
      oss << ",\"scope\":\"" << pomai_search::JsonEscape(scope) << "\"";
    }
    oss << "}";
    body = oss.str();
    path = "/v1/search_by_key";
  } else if (command == "stats") {
    path = "/v1/stats";
  } else {
    PrintUsage();
    return 1;
  }

  std::string response_body;
  if (!SendRequest(options, command == "stats" ? "GET" : "POST", path, body, &response_body)) {
    std::cerr << "Failed to connect to server" << std::endl;
    return 1;
  }
  std::cout << response_body << std::endl;
  return 0;
}
