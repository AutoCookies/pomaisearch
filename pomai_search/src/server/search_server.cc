#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>

#include "net/http_server.h"
#include "net/json.h"
#include "pomai_search/search_engine.h"
#include "pomai_search/types.h"

using pomai_search::HttpRequest;
using pomai_search::HttpResponse;
using pomai_search::JsonEscape;
using pomai_search::JsonValue;
using pomai_search::ParseJson;
using pomai_search::SearchEngine;
using pomai_search::SearchEngineConfig;

namespace {

std::string ErrorBody(const std::string& message) {
  return "{\"error\":\"" + JsonEscape(message) + "\"}";
}

pomai_search::SearchEngine::QueryOptions::Scope ParseScope(const JsonValue& root) {
  std::string scope;
  if (pomai_search::GetStringField(root, "scope", &scope)) {
    if (scope == "local") {
      return pomai_search::SearchEngine::QueryOptions::Scope::Local;
    }
  }
  return pomai_search::SearchEngine::QueryOptions::Scope::Global;
}

bool ParseVector(const JsonValue& root, int dim, std::vector<float>* vec, std::string* error) {
  if (!pomai_search::GetFloatArrayField(root, "vector", vec)) {
    *error = "missing vector";
    return false;
  }
  if (static_cast<int>(vec->size()) != dim) {
    *error = "vector dimension mismatch";
    return false;
  }
  return true;
}

pomai_search::Metadata ParseMetadata(const JsonValue& root) {
  pomai_search::Metadata meta;
  pomai_search::GetStringMapField(root, "metadata", &meta);
  return meta;
}

std::string SerializeResults(const std::vector<pomai_search::ResultItem>& results) {
  std::ostringstream oss;
  oss << "{\"results\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& item = results[i];
    if (i > 0) {
      oss << ",";
    }
    oss << "{\"key\":\"" << JsonEscape(item.key) << "\",\"score\":" << item.score;
    if (!item.meta.empty()) {
      oss << ",\"metadata\":{";
      bool first = true;
      for (const auto& [k, v] : item.meta) {
        if (!first) {
          oss << ",";
        }
        first = false;
        oss << "\"" << JsonEscape(k) << "\":\"" << JsonEscape(v) << "\"";
      }
      oss << "}";
    }
    oss << "}";
  }
  oss << "]}";
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  SearchEngineConfig cfg;
  int port = 8080;
  cfg.num_shards = 4;
  cfg.query_threads = 4;
  cfg.ingest_threads = 4;
  cfg.topk_default = 10;
  cfg.similarity = SearchEngineConfig::Similarity::Dot;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dim" && i + 1 < argc) {
      cfg.dim = std::atoi(argv[++i]);
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (arg == "--shards" && i + 1 < argc) {
      cfg.num_shards = std::atoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      cfg.query_threads = std::atoi(argv[++i]);
      cfg.ingest_threads = cfg.query_threads;
    } else if (arg == "--topk" && i + 1 < argc) {
      cfg.topk_default = std::atoi(argv[++i]);
    } else if (arg == "--similarity" && i + 1 < argc) {
      std::string sim = argv[++i];
      if (sim == "cosine") {
        cfg.similarity = SearchEngineConfig::Similarity::Cosine;
      }
    } else if (arg == "--avx2" && i + 1 < argc) {
      std::string enabled = argv[++i];
      cfg.enable_avx2 = (enabled == "on");
    }
  }
  if (cfg.dim <= 0) {
    std::cerr << "--dim is required" << std::endl;
    return 1;
  }
  auto engine_result = SearchEngine::Open(cfg);
  if (!engine_result.ok()) {
    std::cerr << engine_result.status().ToString() << std::endl;
    return 1;
  }
  auto engine = std::move(engine_result.value());
  pomai_search::HttpServer server;
  auto handler = [engine_ptr = engine.get(), cfg](const HttpRequest& req) {
    HttpResponse resp;
    if (req.path == "/v1/stats") {
      auto stats = engine_ptr->GetStats();
      std::ostringstream oss;
      oss << "{\"num_points\":" << stats.num_points << ",\"num_deleted\":" << stats.num_deleted
          << ",\"last_query_ms_p50\":" << stats.last_query_ms_p50 << "}";
      resp.body = oss.str();
      return resp;
    }
    if (req.method != "POST") {
      resp.status = 404;
      resp.body = ErrorBody("not found");
      return resp;
    }
    JsonValue root;
    std::string error;
    if (!ParseJson(req.body, &root, &error)) {
      resp.status = 400;
      resp.body = ErrorBody(error);
      return resp;
    }
    if (req.path == "/v1/upsert") {
      std::string key;
      if (!pomai_search::GetStringField(root, "key", &key)) {
        resp.status = 400;
        resp.body = ErrorBody("missing key");
        return resp;
      }
      std::vector<float> vec;
      if (!ParseVector(root, cfg.dim, &vec, &error)) {
        resp.status = 400;
        resp.body = ErrorBody(error);
        return resp;
      }
      int ttl_ms = 0;
      std::optional<std::chrono::milliseconds> ttl;
      if (pomai_search::GetIntField(root, "ttl_ms", &ttl_ms)) {
        ttl = std::chrono::milliseconds(ttl_ms);
      }
      auto status = engine_ptr->Upsert(key, {vec.data(), cfg.dim}, ParseMetadata(root), ttl);
      if (!status.ok()) {
        resp.status = 400;
        resp.body = ErrorBody(status.ToString());
      } else {
        resp.body = "{\"status\":\"ok\"}";
      }
      return resp;
    }
    if (req.path == "/v1/delete") {
      std::string key;
      if (!pomai_search::GetStringField(root, "key", &key)) {
        resp.status = 400;
        resp.body = ErrorBody("missing key");
        return resp;
      }
      auto status = engine_ptr->Delete(key);
      if (!status.ok()) {
        resp.status = 404;
        resp.body = ErrorBody(status.ToString());
      } else {
        resp.body = "{\"status\":\"ok\"}";
      }
      return resp;
    }
    if (req.path == "/v1/search" || req.path == "/v1/search_by_key") {
      pomai_search::SearchEngine::QueryOptions options;
      options.scope = ParseScope(root);
      pomai_search::GetIntField(root, "topk", &options.topk);
      pomai_search::GetStringMapField(root, "filter", &options.filter_equals);
      if (req.path == "/v1/search") {
        std::vector<float> vec;
        if (!ParseVector(root, cfg.dim, &vec, &error)) {
          resp.status = 400;
          resp.body = ErrorBody(error);
          return resp;
        }
        auto result = engine_ptr->Search({vec.data(), cfg.dim}, options);
        if (!result.ok()) {
          resp.status = 400;
          resp.body = ErrorBody(result.status().ToString());
          return resp;
        }
        resp.body = SerializeResults(result.value());
        return resp;
      }
      std::string key;
      if (!pomai_search::GetStringField(root, "key", &key)) {
        resp.status = 400;
        resp.body = ErrorBody("missing key");
        return resp;
      }
      auto result = engine_ptr->SearchByKey(key, options);
      if (!result.ok()) {
        resp.status = 400;
        resp.body = ErrorBody(result.status().ToString());
        return resp;
      }
      resp.body = SerializeResults(result.value());
      return resp;
    }
    resp.status = 404;
    resp.body = ErrorBody("not found");
    return resp;
  };
  if (!server.Start(port, handler, cfg.query_threads)) {
    std::cerr << "Failed to start server" << std::endl;
    return 1;
  }
  std::cout << "pomai-searchd listening on port " << port << std::endl;
  server.Wait();
  return 0;
}
