#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "app/http_server.h"
#include "app/json.h"
#include "pomai_search/search_engine.h"

namespace pomai_search {

namespace {

struct ServerConfig {
  int port = 8080;
  SearchEngineConfig engine;
  int worker_threads = 4;
  size_t max_inflight = 0;
  size_t max_body_bytes = 0;
  int timeout_ms = 0;
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
    } else if (arg == "--max_body_bytes") {
      cfg->max_body_bytes = static_cast<size_t>(std::stoul(next()));
    } else if (arg == "--timeout_ms") {
      cfg->timeout_ms = std::stoi(next());
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
    } else if (arg == "--global_seed") {
      cfg->engine.global_seed = static_cast<uint64_t>(std::stoull(next()));
    } else if (arg == "--contract_version") {
      cfg->engine.contract_version = static_cast<uint32_t>(std::stoul(next()));
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

QueryPolicy ParsePolicy(const JsonValue& object) {
  QueryPolicy policy;
  int value = 0;
  if (GetIntField(object, "max_latency_ms", &value)) {
    policy.max_latency_ms = value;
  }
  if (GetIntField(object, "max_candidates", &value)) {
    policy.max_candidates = value;
  }
  const JsonValue* recall_field = FindField(object, "recall_bias");
  if (recall_field && recall_field->type == JsonValue::Type::kNumber) {
    policy.recall_bias = static_cast<float>(recall_field->number);
  }
  std::string fusion;
  if (GetStringField(object, "fusion_method", &fusion)) {
    if (fusion == "rrf") {
      policy.fusion_method = FusionMethod::Rrf;
    } else {
      policy.fusion_method = FusionMethod::WeightedSum;
    }
  }
  return policy;
}

void AppendMetadataJson(const Metadata& meta, std::string* body) {
  std::vector<std::string> keys;
  keys.reserve(meta.size());
  for (const auto& pair : meta) {
    keys.push_back(pair.first);
  }
  std::sort(keys.begin(), keys.end());
  bool first_meta = true;
  for (const auto& key : keys) {
    auto it = meta.find(key);
    if (it == meta.end()) {
      continue;
    }
    if (!first_meta) {
      *body += ",";
    }
    first_meta = false;
    *body += "\"" + JsonEscape(it->first) + "\":\"" + JsonEscape(it->second) + "\"";
  }
}

void AppendResultsJson(const std::vector<ResultItem>& results, std::string* body) {
  bool first = true;
  for (const auto& item : results) {
    if (!first) {
      *body += ",";
    }
    first = false;
    *body += "{\"key\":\"" + JsonEscape(item.key) + "\",\"score\":" +
             std::to_string(item.score) + ",\"metadata\":{";
    AppendMetadataJson(item.meta, body);
    *body += "}}";
  }
}

void AppendExplainJson(const QueryExplain& explain, std::string* body) {
  *body += "\"explain\":{";
  *body += "\"query_id\":\"" + std::to_string(explain.query_id) + "\",";
  *body += "\"snapshot_id\":\"" + JsonEscape(explain.snapshot_id) + "\",";
  *body += "\"global_seed\":\"" + std::to_string(explain.global_seed) + "\",";
  *body += "\"contract_version\":" + std::to_string(explain.contract_version) + ",";
  *body += "\"execution_plan\":[";
  bool first_stage = true;
  for (const auto& stage : explain.execution_plan) {
    if (!first_stage) {
      *body += ",";
    }
    first_stage = false;
    *body += "{";
    *body += "\"name\":\"" + JsonEscape(stage.name) + "\",";
    *body += "\"index\":\"" + JsonEscape(stage.index) + "\",";
    *body += "\"ef_search\":" + std::to_string(stage.ef_search) + ",";
    *body += "\"nprobe\":" + std::to_string(stage.nprobe) + ",";
    *body += "\"max_candidates\":" + std::to_string(stage.max_candidates) + ",";
    *body += "\"time_ms\":" + std::to_string(stage.time_ms) + ",";
    *body += "\"candidates_out\":" + std::to_string(stage.candidates_out);
    *body += "}";
  }
  *body += "],\"results\":[";
  bool first_detail = true;
  for (const auto& detail : explain.result_details) {
    if (!first_detail) {
      *body += ",";
    }
    first_detail = false;
    *body += "{";
    *body += "\"vector_score_raw\":" + std::to_string(detail.vector_score_raw) + ",";
    *body += "\"vector_score_normed\":" + std::to_string(detail.vector_score_normed) + ",";
    *body += "\"keyword_score_raw\":" + std::to_string(detail.keyword_score_raw) + ",";
    *body += "\"keyword_score_normed\":" + std::to_string(detail.keyword_score_normed) + ",";
    *body += "\"fusion_method\":\"" + JsonEscape(detail.fusion_method) + "\",";
    *body += "\"final_score\":" + std::to_string(detail.final_score) + ",";
    *body += "\"rank_before_fusion\":" + std::to_string(detail.rank_before_fusion) + ",";
    *body += "\"rank_after_fusion\":" + std::to_string(detail.rank_after_fusion);
    *body += "}";
  }
  *body += "]}";
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
  AppendResultsJson(result.value(), &body);
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
  AppendResultsJson(result.value(), &body);
  body += "]}";
  HttpResponse resp;
  resp.status = 200;
  resp.body = std::move(body);
  return resp;
}

HttpResponse HandleSearchExplain(SearchEngine* engine, const HttpRequest& req) {
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
  QueryPolicy policy;
  const JsonValue* policy_field = FindField(root, "policy");
  if (policy_field && policy_field->type == JsonValue::Type::kObject) {
    policy = ParsePolicy(*policy_field);
  }
  SearchEngine::QueryOptions options;
  options.topk = topk;
  options.filter = filter;
  VectorView view{vector.data(), static_cast<int>(vector.size())};
  auto result = engine->SearchWithExplain(view, options, policy);
  if (!result.ok()) {
    return JsonError(400, "INVALID_ARGUMENT", result.status().message());
  }
  std::string body = "{\"ok\":true,\"results\":[";
  AppendResultsJson(result.value().results, &body);
  body += "],";
  AppendExplainJson(result.value().explain, &body);
  body += "}";
  HttpResponse resp;
  resp.status = 200;
  resp.body = std::move(body);
  return resp;
}

HttpResponse HandleHybridExplain(SearchEngine* engine, const HttpRequest& req) {
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
  QueryPolicy policy;
  const JsonValue* policy_field = FindField(root, "policy");
  if (policy_field && policy_field->type == JsonValue::Type::kObject) {
    policy = ParsePolicy(*policy_field);
  }
  SearchEngine::QueryOptions options;
  options.topk = topk;
  options.filter = filter;
  auto result = engine->SearchHybridWithExplain(query, options, policy);
  if (!result.ok()) {
    return JsonError(400, "INVALID_ARGUMENT", result.status().message());
  }
  std::string body = "{\"ok\":true,\"results\":[";
  AppendResultsJson(result.value().results, &body);
  body += "],";
  AppendExplainJson(result.value().explain, &body);
  body += "}";
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
        if (req.method == "POST" && req.path == "/v1/search_explain") {
          return pomai_search::HandleSearchExplain(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/search_hybrid") {
          return pomai_search::HandleHybrid(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/search_hybrid_explain") {
          return pomai_search::HandleHybridExplain(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/upsert") {
          return pomai_search::HandleUpsert(engine_ptr, req);
        }
        if (req.method == "POST" && req.path == "/v1/delete") {
          return pomai_search::HandleDelete(engine_ptr, req);
        }
        return pomai_search::JsonError(404, "NOT_FOUND", "unknown endpoint");
      },
      cfg.worker_threads, cfg.max_inflight, cfg.max_body_bytes, cfg.timeout_ms);
  if (!started) {
    std::cerr << "Failed to start server" << "\n";
    return 1;
  }
  server.Wait();
  return 0;
}
