#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "pomai_search/net/http_server.h"
#include "pomai_search/net/json.h"
#include "pomai_search/search_engine.h"

namespace pomai_search {

namespace {

struct ServerConfig {
  int port = 8080;
  SearchEngineConfig engine;
  int worker_threads = 4;
  size_t max_inflight = 0;
};

bool ParseArgs(int argc, char** argv, ServerConfig* cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        return "";
      }
      return argv[++i];
    };
    if (arg == "--port") {
      cfg->port = std::stoi(next());
    } else if (arg == "--dim") {
      cfg->engine.dim = std::stoi(next());
    } else if (arg == "--shards") {
      cfg->engine.num_shards = std::stoi(next());
    } else if (arg == "--index") {
      std::string value = next();
      cfg->engine.index_type =
          value == "hnsw" ? SearchEngineConfig::IndexType::Hnsw : SearchEngineConfig::IndexType::Flat;
    } else if (arg == "--hnsw_m") {
      cfg->engine.hnsw_m = std::stoi(next());
    } else if (arg == "--hnsw_ef_construction") {
      cfg->engine.hnsw_ef_construction = std::stoi(next());
    } else if (arg == "--hnsw_ef_search") {
      cfg->engine.hnsw_ef_search = std::stoi(next());
    } else if (arg == "--threads") {
      cfg->worker_threads = std::stoi(next());
    } else if (arg == "--max_inflight") {
      cfg->max_inflight = static_cast<size_t>(std::stoul(next()));
    } else if (arg == "--similarity") {
      std::string value = next();
      cfg->engine.similarity = value == "cosine" ? SearchEngineConfig::Similarity::Cosine
                                                  : SearchEngineConfig::Similarity::Dot;
    } else if (arg == "--enable_avx2") {
      cfg->engine.enable_avx2 = next() == "on";
    } else if (arg == "--query_threads") {
      cfg->engine.query_threads = std::stoi(next());
    } else if (arg == "--ingest_threads") {
      cfg->engine.ingest_threads = std::stoi(next());
    }
  }
  return cfg->engine.dim > 0 && cfg->engine.num_shards > 0;
}

HttpResponse JsonError(int status, std::string_view code, std::string_view message) {
  HttpResponse resp;
  resp.status = status;
  resp.content_type = "application/json";
  resp.body = "{\"ok\":false,\"code\":\"" + std::string(code) + "\",\"message\":\"" +
              JsonEscape(message) + "\"}";
  return resp;
}

Filter ParseFilter(const JsonValue& object) {
  Filter filter;
  std::string value;
  if (GetStringField(object, "tag", &value)) {
    filter.tag = value;
  }
  if (GetStringField(object, "source", &value)) {
    filter.source = value;
  }
  if (GetStringField(object, "lang", &value)) {
    filter.lang = value;
  }
  return filter;
}

HttpResponse HandleSearch(SearchEngine* engine, const HttpRequest& req) {
  JsonValue root;
  std::string error;
  if (!ParseJson(req.body, &root, &error) || root.type != JsonValue::Type::kObject) {
    return JsonError(400, "INVALID_ARGUMENT", "invalid JSON");
  }
  std::vector<float> vector;
  if (!GetFloatArrayField(root, "vector", &vector)) {
    return JsonError(400, "INVALID_ARGUMENT", "vector required");
  }
  int topk = 0;
  GetIntField(root, "topk", &topk);
  Filter filter;
  const JsonValue* filter_field = FindField(root, "filter");
  if (filter_field && filter_field->type == JsonValue::Type::kObject) {
    filter = ParseFilter(*filter_field);
  }
  SearchEngine::QueryOptions options;
  options.topk = topk;
  options.filter = filter;
  VectorView view{vector.data(), static_cast<int>(vector.size())};
  auto result = engine->Search(view, options);
  if (!result.ok()) {
    return JsonError(400, "INVALID_ARGUMENT", result.status().message());
  }
  std::string body = "{\"ok\":true,\"results\":[";
  bool first = true;
  for (const auto& item : result.value()) {
    if (!first) {
      body += ",";
    }
    first = false;
    body += "{\"key\":\"" + JsonEscape(item.key) + "\",\"score\":" +
            std::to_string(item.score) + ",\"metadata\":{";
    bool first_meta = true;
    for (const auto& meta : item.meta) {
      if (!first_meta) {
        body += ",";
      }
      first_meta = false;
      body += "\"" + JsonEscape(meta.first) + "\":\"" + JsonEscape(meta.second) + "\"";
    }
    body += "}}";
  }
  body += "]}";
  HttpResponse resp;
  resp.status = 200;
  resp.body = std::move(body);
  return resp;
}

HttpResponse HandleHybrid(SearchEngine* engine, const HttpRequest& req) {
  JsonValue root;
  std::string error;
  if (!ParseJson(req.body, &root, &error) || root.type != JsonValue::Type::kObject) {
    return JsonError(400, "INVALID_ARGUMENT", "invalid JSON");
  }
  SearchEngine::HybridQuery query;
  std::string text;
  if (GetStringField(root, "text_query", &text)) {
    query.text_query = text;
  }
  std::vector<float> vector;
  if (GetFloatArrayField(root, "vector", &vector)) {
    query.vector_query = vector;
  }
  int topk = 0;
  GetIntField(root, "topk", &topk);
  const JsonValue* alpha_field = FindField(root, "alpha");
  if (alpha_field && alpha_field->type == JsonValue::Type::kNumber) {
    query.alpha = static_cast<float>(alpha_field->number);
  }
  Filter filter;
  const JsonValue* filter_field = FindField(root, "filter");
  if (filter_field && filter_field->type == JsonValue::Type::kObject) {
    filter = ParseFilter(*filter_field);
  }
  SearchEngine::QueryOptions options;
  options.topk = topk;
  options.filter = filter;
  auto result = engine->SearchHybrid(query, options);
  if (!result.ok()) {
    return JsonError(400, "INVALID_ARGUMENT", result.status().message());
  }
  std::string body = "{\"ok\":true,\"results\":[";
  bool first = true;
  for (const auto& item : result.value()) {
    if (!first) {
      body += ",";
    }
    first = false;
    body += "{\"key\":\"" + JsonEscape(item.key) + "\",\"score\":" +
            std::to_string(item.score) + ",\"metadata\":{";
    bool first_meta = true;
    for (const auto& meta : item.meta) {
      if (!first_meta) {
        body += ",";
      }
      first_meta = false;
      body += "\"" + JsonEscape(meta.first) + "\":\"" + JsonEscape(meta.second) + "\"";
    }
    body += "}}";
  }
  body += "]}";
  HttpResponse resp;
  resp.status = 200;
  resp.body = std::move(body);
  return resp;
}

HttpResponse HandleUpsert(SearchEngine* engine, const HttpRequest& req) {
  JsonValue root;
  std::string error;
  if (!ParseJson(req.body, &root, &error) || root.type != JsonValue::Type::kObject) {
    return JsonError(400, "INVALID_ARGUMENT", "invalid JSON");
  }
  std::string key;
  if (!GetStringField(root, "key", &key)) {
    return JsonError(400, "INVALID_ARGUMENT", "key required");
  }
  std::vector<float> vector;
  if (!GetFloatArrayField(root, "vector", &vector)) {
    return JsonError(400, "INVALID_ARGUMENT", "vector required");
  }
  Metadata meta;
  GetStringMapField(root, "metadata", &meta);
  std::string text;
  std::optional<std::string> text_opt;
  if (GetStringField(root, "text", &text)) {
    text_opt = text;
  }
  VectorView view{vector.data(), static_cast<int>(vector.size())};
  Status status = engine->Upsert(key, view, std::move(meta), std::nullopt, text_opt);
  if (!status.ok()) {
    return JsonError(400, "INVALID_ARGUMENT", status.message());
  }
  HttpResponse resp;
  resp.status = 200;
  resp.body = "{\"ok\":true}";
  return resp;
}

HttpResponse HandleDelete(SearchEngine* engine, const HttpRequest& req) {
  JsonValue root;
  std::string error;
  if (!ParseJson(req.body, &root, &error) || root.type != JsonValue::Type::kObject) {
    return JsonError(400, "INVALID_ARGUMENT", "invalid JSON");
  }
  std::string key;
  if (!GetStringField(root, "key", &key)) {
    return JsonError(400, "INVALID_ARGUMENT", "key required");
  }
  Status status = engine->Delete(key);
  if (!status.ok()) {
    return JsonError(404, "NOT_FOUND", status.message());
  }
  HttpResponse resp;
  resp.status = 200;
  resp.body = "{\"ok\":true}";
  return resp;
}

}  // namespace

}  // namespace pomai_search

int main(int argc, char** argv) {
  pomai_search::ServerConfig cfg;
  if (!pomai_search::ParseArgs(argc, argv, &cfg)) {
    std::cerr << "Usage: pomai-searchd --dim <dim> --shards <n> [--port <p>] [--index flat|hnsw]\n";
    return 1;
  }
  auto engine_or = pomai_search::SearchEngine::Open(cfg.engine);
  if (!engine_or.ok()) {
    std::cerr << "Failed to init engine: " << engine_or.status().ToString() << "\n";
    return 1;
  }
  auto engine = std::move(engine_or.value());
  pomai_search::HttpServer server;
  bool started = server.Start(
      cfg.port,
      [engine_ptr = engine.get()](const pomai_search::HttpRequest& req) {
        if (req.method == "GET" && req.path == "/v1/healthz") {
          pomai_search::HttpResponse resp;
          resp.status = 200;
          resp.body = "{\"ok\":true}";
          return resp;
        }
        if (req.method == "GET" && req.path == "/v1/readyz") {
          pomai_search::HttpResponse resp;
          resp.status = 200;
          resp.body = "{\"ok\":true}";
          return resp;
        }
        if (req.method == "GET" && req.path == "/v1/metrics") {
          pomai_search::HttpResponse resp;
          resp.status = 200;
          resp.body = engine_ptr->MetricsJson();
          return resp;
        }
        if (req.method == "POST" && req.path == "/v1/search") {
          return pomai_search::HandleSearch(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/search_hybrid") {
          return pomai_search::HandleHybrid(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/upsert") {
          return pomai_search::HandleUpsert(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/delete") {
          return pomai_search::HandleDelete(engine_ptr, req);
        }
        return pomai_search::JsonError(404, "NOT_FOUND", "unknown endpoint");
      },
      cfg.worker_threads, cfg.max_inflight);
  if (!started) {
    std::cerr << "Failed to start server" << "\n";
    return 1;
  }
  server.Wait();
  return 0;
}
