#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pomai_search/logging.h"
#include "pomai_search/search_engine.h"
#include "pomai_search/status.h"
#include "pomai_search/types.h"
#include "server/http_server.h"
#include "server/json.h"

namespace pomai_search {

static std::string StatusToJson(const Status& status) {
  return std::string("{\"ok\":false,\"code\":\"") + status.ToString() + "\"}";
}

static std::optional<std::string> GetString(const JsonValue& value) {
  if (value.type != JsonValue::Type::String) {
    return std::nullopt;
  }
  return value.string_value;
}

static std::optional<double> GetNumber(const JsonValue& value) {
  if (value.type != JsonValue::Type::Number) {
    return std::nullopt;
  }
  return value.number_value;
}

static std::vector<float> ParseVector(const JsonValue& value) {
  std::vector<float> out;
  if (value.type != JsonValue::Type::Array) {
    return out;
  }
  out.reserve(value.array_value.size());
  for (const auto& element : value.array_value) {
    if (element.type != JsonValue::Type::Number) {
      return {};
    }
    out.push_back(static_cast<float>(element.number_value));
  }
  return out;
}

static Metadata ParseMetadata(const JsonValue& value) {
  Metadata meta;
  if (value.type != JsonValue::Type::Object) {
    return meta;
  }
  for (const auto& pair : value.object_value) {
    if (pair.second.type == JsonValue::Type::String) {
      meta.emplace(pair.first, pair.second.string_value);
    }
  }
  return meta;
}

static std::string ResultsToJson(const std::vector<ResultItem>& results) {
  std::string json = "{\"results\":[";
  bool first = true;
  for (const auto& item : results) {
    if (!first) {
      json += ",";
    }
    first = false;
    json += "{\"key\":\"" + item.key + "\",\"score\":" + std::to_string(item.score) + ",\"meta\":{";
    bool first_meta = true;
    for (const auto& meta : item.meta) {
      if (!first_meta) {
        json += ",";
      }
      first_meta = false;
      json += "\"" + meta.first + "\":\"" + meta.second + "\"";
    }
    json += "}}";
  }
  json += "]}";
  return json;
}

static std::string StatsToJson(const SearchEngine::Stats& stats) {
  std::string json = "{";
  json += "\"num_points\":" + std::to_string(stats.num_points) + ",";
  json += "\"num_deleted\":" + std::to_string(stats.num_deleted) + ",";
  json += "\"last_query_ms_p50\":" + std::to_string(stats.last_query_ms_p50);
  json += "}";
  return json;
}

}  // namespace pomai_search

int main(int argc, char** argv) {
  using namespace pomai_search;
  SearchEngineConfig config;
  config.dim = 0;
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    auto next = [&]() -> std::optional<std::string_view> {
      if (i + 1 >= argc) {
        return std::nullopt;
      }
      return std::string_view(argv[++i]);
    };
    if (arg == "--dim") {
      auto value = next();
      if (!value) {
        std::cerr << "--dim requires value\n";
        return 1;
      }
      config.dim = std::stoi(std::string(*value));
    } else if (arg == "--shards") {
      auto value = next();
      if (value) {
        config.num_shards = std::stoi(std::string(*value));
      }
    } else if (arg == "--similarity") {
      auto value = next();
      if (value && *value == "cosine") {
        config.similarity = SearchEngineConfig::Similarity::Cosine;
      } else {
        config.similarity = SearchEngineConfig::Similarity::Dot;
      }
    } else if (arg == "--port") {
      auto value = next();
      if (value) {
        port = std::stoi(std::string(*value));
      }
    } else if (arg == "--threads") {
      auto value = next();
      if (value) {
        config.query_threads = std::stoi(std::string(*value));
      }
    } else if (arg == "--avx2") {
      auto value = next();
      if (value && *value == "off") {
        config.enable_avx2 = false;
      }
    }
  }
  if (config.dim <= 0) {
    std::cerr << "--dim is required\n";
    return 1;
  }

  auto engine_result = SearchEngine::Open(config);
  if (!engine_result.ok()) {
    std::cerr << engine_result.status().ToString() << "\n";
    return 1;
  }
  auto engine = std::move(engine_result.value());

  HttpServer server(port);
  server.AddHandler("POST", "/v1/upsert", [engine_ptr = engine.get()](const HttpRequest& request) {
    auto json = ParseJson(request.body);
    if (!json.ok()) {
      return HttpResponse{400, StatusToJson(json.status())};
    }
    if (json.value().type != JsonValue::Type::Object) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid json\"}"};
    }
    const auto& obj = json.value().object_value;
    auto key_it = obj.find("key");
    auto vector_it = obj.find("vector");
    if (key_it == obj.end() || vector_it == obj.end()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"missing fields\"}"};
    }
    auto key = GetString(key_it->second);
    if (!key) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid key\"}"};
    }
    std::vector<float> vec = ParseVector(vector_it->second);
    if (vec.empty()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid vector\"}"};
    }
    Metadata meta;
    auto meta_it = obj.find("meta");
    if (meta_it != obj.end()) {
      meta = ParseMetadata(meta_it->second);
    }
    std::optional<std::chrono::milliseconds> ttl;
    auto ttl_it = obj.find("ttl_ms");
    if (ttl_it != obj.end()) {
      auto ttl_val = GetNumber(ttl_it->second);
      if (ttl_val) {
        ttl = std::chrono::milliseconds(static_cast<int64_t>(*ttl_val));
      }
    }
    VectorView view{vec.data(), static_cast<int>(vec.size())};
    Status status = engine_ptr->Upsert(*key, view, std::move(meta), ttl);
    if (!status.ok()) {
      return HttpResponse{400, StatusToJson(status)};
    }
    return HttpResponse{200, "{\"ok\":true}"};
  });

  server.AddHandler("POST", "/v1/delete", [engine_ptr = engine.get()](const HttpRequest& request) {
    auto json = ParseJson(request.body);
    if (!json.ok()) {
      return HttpResponse{400, StatusToJson(json.status())};
    }
    if (json.value().type != JsonValue::Type::Object) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid json\"}"};
    }
    const auto& obj = json.value().object_value;
    auto key_it = obj.find("key");
    if (key_it == obj.end()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"missing key\"}"};
    }
    auto key = GetString(key_it->second);
    if (!key) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid key\"}"};
    }
    Status status = engine_ptr->Delete(*key);
    if (!status.ok()) {
      return HttpResponse{404, StatusToJson(status)};
    }
    return HttpResponse{200, "{\"ok\":true}"};
  });

  server.AddHandler("POST", "/v1/search", [engine_ptr = engine.get()](const HttpRequest& request) {
    auto json = ParseJson(request.body);
    if (!json.ok()) {
      return HttpResponse{400, StatusToJson(json.status())};
    }
    if (json.value().type != JsonValue::Type::Object) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid json\"}"};
    }
    const auto& obj = json.value().object_value;
    auto vector_it = obj.find("vector");
    if (vector_it == obj.end()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"missing vector\"}"};
    }
    std::vector<float> vec = ParseVector(vector_it->second);
    if (vec.empty()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid vector\"}"};
    }
    SearchEngine::QueryOptions options;
    auto topk_it = obj.find("topk");
    if (topk_it != obj.end()) {
      auto val = GetNumber(topk_it->second);
      if (val) {
        options.topk = static_cast<int>(*val);
      }
    }
    auto filter_it = obj.find("filter");
    if (filter_it != obj.end()) {
      options.filter_equals = ParseMetadata(filter_it->second);
    }
    VectorView view{vec.data(), static_cast<int>(vec.size())};
    auto result = engine_ptr->Search(view, options);
    if (!result.ok()) {
      return HttpResponse{400, StatusToJson(result.status())};
    }
    return HttpResponse{200, ResultsToJson(result.value())};
  });

  server.AddHandler("POST", "/v1/search_by_key", [engine_ptr = engine.get()](const HttpRequest& request) {
    auto json = ParseJson(request.body);
    if (!json.ok()) {
      return HttpResponse{400, StatusToJson(json.status())};
    }
    if (json.value().type != JsonValue::Type::Object) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid json\"}"};
    }
    const auto& obj = json.value().object_value;
    auto key_it = obj.find("key");
    if (key_it == obj.end()) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"missing key\"}"};
    }
    auto key = GetString(key_it->second);
    if (!key) {
      return HttpResponse{400, "{\"ok\":false,\"message\":\"invalid key\"}"};
    }
    SearchEngine::QueryOptions options;
    auto topk_it = obj.find("topk");
    if (topk_it != obj.end()) {
      auto val = GetNumber(topk_it->second);
      if (val) {
        options.topk = static_cast<int>(*val);
      }
    }
    auto filter_it = obj.find("filter");
    if (filter_it != obj.end()) {
      options.filter_equals = ParseMetadata(filter_it->second);
    }
    auto result = engine_ptr->SearchByKey(*key, options);
    if (!result.ok()) {
      return HttpResponse{400, StatusToJson(result.status())};
    }
    return HttpResponse{200, ResultsToJson(result.value())};
  });

  server.AddHandler("GET", "/v1/stats", [engine_ptr = engine.get()](const HttpRequest&) {
    auto stats = engine_ptr->GetStats();
    return HttpResponse{200, StatsToJson(stats)};
  });

  POMAI_LOG_INFO("pomai-searchd listening on port " + std::to_string(port));
  if (!server.Start()) {
    std::cerr << "Failed to start server\n";
    return 1;
  }
  return 0;
}
