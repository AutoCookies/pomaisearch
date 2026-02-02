#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CliConfig {
  std::string host = "127.0.0.1";
  int port = 8080;
};

bool SendRequest(const CliConfig& config, const std::string& method, const std::string& path,
                 const std::string& body, std::string* response_out) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(config.port));
  if (inet_pton(AF_INET, config.host.c_str(), &addr.sin_addr) <= 0) {
    close(sock);
    return false;
  }
  if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(sock);
    return false;
  }
  std::ostringstream oss;
  oss << method << " " << path << " HTTP/1.1\r\n";
  oss << "Host: " << config.host << "\r\n";
  if (!body.empty()) {
    oss << "Content-Type: application/json\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
  } else {
    oss << "Content-Length: 0\r\n";
  }
  oss << "Connection: close\r\n\r\n";
  oss << body;
  std::string payload = oss.str();
  send(sock, payload.data(), payload.size(), 0);

  char buffer[4096];
  std::string response;
  ssize_t n = 0;
  while ((n = read(sock, buffer, sizeof(buffer))) > 0) {
    response.append(buffer, buffer + n);
  }
  close(sock);
  auto pos = response.find("\r\n\r\n");
  if (pos != std::string::npos) {
    *response_out = response.substr(pos + 4);
  } else {
    *response_out = response;
  }
  return true;
}

std::vector<float> ParseVectorArg(std::string_view input) {
  std::vector<float> vec;
  std::stringstream ss{std::string(input)};
  std::string token;
  while (std::getline(ss, token, ',')) {
    vec.push_back(std::stof(token));
  }
  return vec;
}

std::string VectorToJson(const std::vector<float>& vec) {
  std::string json = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    json += std::to_string(vec[i]);
    if (i + 1 < vec.size()) {
      json += ",";
    }
  }
  json += "]";
  return json;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: pomai-search <command> [options]\n";
    return 1;
  }
  std::string command(argv[1]);
  CliConfig config;
  std::string key;
  std::vector<float> vec;
  std::string tag;
  int topk = 0;

  for (int i = 2; i < argc; ++i) {
    std::string_view arg(argv[i]);
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) {
        return {};
      }
      return std::string_view(argv[++i]);
    };
    if (arg == "--host") {
      config.host = std::string(next());
    } else if (arg == "--port") {
      config.port = std::stoi(std::string(next()));
    } else if (arg == "--key") {
      key = std::string(next());
    } else if (arg == "--vec") {
      vec = ParseVectorArg(next());
    } else if (arg == "--tag") {
      tag = std::string(next());
    } else if (arg == "--topk") {
      topk = std::stoi(std::string(next()));
    }
  }

  std::string response;
  if (command == "upsert") {
    if (key.empty() || vec.empty()) {
      std::cerr << "upsert requires --key and --vec\n";
      return 1;
    }
    std::string body = "{\"key\":\"" + key + "\",\"vector\":" + VectorToJson(vec);
    if (!tag.empty()) {
      body += ",\"meta\":{\"tag\":\"" + tag + "\"}";
    }
    body += "}";
    if (!SendRequest(config, "POST", "/v1/upsert", body, &response)) {
      std::cerr << "request failed\n";
      return 1;
    }
  } else if (command == "search") {
    if (vec.empty()) {
      std::cerr << "search requires --vec\n";
      return 1;
    }
    std::string body = "{\"vector\":" + VectorToJson(vec);
    if (topk > 0) {
      body += ",\"topk\":" + std::to_string(topk);
    }
    body += "}";
    if (!SendRequest(config, "POST", "/v1/search", body, &response)) {
      std::cerr << "request failed\n";
      return 1;
    }
  } else if (command == "stats") {
    if (!SendRequest(config, "GET", "/v1/stats", "", &response)) {
      std::cerr << "request failed\n";
      return 1;
    }
  } else {
    std::cerr << "Unknown command: " << command << "\n";
    return 1;
  }

  std::cout << response << "\n";
  return 0;
}
