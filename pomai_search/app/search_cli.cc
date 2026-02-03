#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "app/json.h"
#include "pomai_search/search_engine.h"
#include "core/serialize/snapshot.h"

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
  std::string dataset_path;
  std::string queries_path;
  std::string out_dir;
  std::string contract_dir;
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
    } else if (arg == "--dataset") {
      cfg->dataset_path = next();
    } else if (arg == "--queries") {
      cfg->queries_path = next();
    } else if (arg == "--out") {
      cfg->out_dir = next();
    } else if (arg == "--dir") {
      cfg->contract_dir = next();
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

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}

bool WriteFile(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  out << content;
  return out.good();
}

bool ParseConfig(const JsonValue& object, SearchEngineConfig* cfg) {
  if (object.type != JsonValue::Type::kObject) {
    return false;
  }
  int dim = 0;
  int shards = 1;
  if (!GetIntField(object, "dim", &dim)) {
    return false;
  }
  GetIntField(object, "num_shards", &shards);
  cfg->dim = dim;
  cfg->num_shards = shards;
  std::string index_type;
  if (GetStringField(object, "index_type", &index_type)) {
    cfg->index_type = index_type == "hnsw" ? SearchEngineConfig::IndexType::Hnsw
                                           : SearchEngineConfig::IndexType::Flat;
  }
  const JsonValue* seed_field = FindField(object, "global_seed");
  if (seed_field && seed_field->type == JsonValue::Type::kNumber) {
    cfg->global_seed = static_cast<uint64_t>(seed_field->number);
  }
  int contract_version = 0;
  if (GetIntField(object, "contract_version", &contract_version)) {
    cfg->contract_version = static_cast<uint32_t>(contract_version);
  }
  return true;
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
    policy.fusion_method = fusion == "rrf" ? FusionMethod::Rrf : FusionMethod::WeightedSum;
  }
  return policy;
}

bool LoadDataset(const std::string& dataset_path, SearchEngineConfig* cfg,
                 std::vector<SnapshotRecord>* records) {
  std::string dataset_json;
  if (!ReadFile(dataset_path, &dataset_json)) {
    return false;
  }
  JsonValue root;
  std::string error;
  if (!ParseJson(dataset_json, &root, &error)) {
    return false;
  }
  const JsonValue* config_field = FindField(root, "config");
  const JsonValue* records_field = FindField(root, "records");
  if (!config_field || !records_field) {
    return false;
  }
  if (!ParseConfig(*config_field, cfg)) {
    return false;
  }
  if (records_field->type != JsonValue::Type::kArray) {
    return false;
  }
  for (const auto& record : records_field->array) {
    if (record.type != JsonValue::Type::kObject) {
      return false;
    }
    SnapshotRecord item;
    if (!GetStringField(record, "key", &item.key)) {
      return false;
    }
    if (!GetFloatArrayField(record, "vector", &item.vector)) {
      return false;
    }
    GetStringMapField(record, "metadata", &item.meta);
    std::string text;
    if (GetStringField(record, "text", &text)) {
      item.text = text;
    }
    records->push_back(std::move(item));
  }
  return true;
}

bool HandleContractRecord(const CliConfig& cfg) {
  SearchEngineConfig engine_cfg;
  std::vector<SnapshotRecord> records;
  if (!LoadDataset(cfg.dataset_path, &engine_cfg, &records)) {
    return false;
  }
  std::string queries_json;
  if (!ReadFile(cfg.queries_path, &queries_json)) {
    return false;
  }
  JsonValue queries_root;
  std::string error;
  if (!ParseJson(queries_json, &queries_root, &error)) {
    return false;
  }
  const JsonValue* queries_field = FindField(queries_root, "queries");
  if (!queries_field || queries_field->type != JsonValue::Type::kArray) {
    return false;
  }
  auto engine_or = SearchEngine::Open(engine_cfg);
  if (!engine_or.ok()) {
    return false;
  }
  auto engine = std::move(engine_or.value());
  for (const auto& record : records) {
    VectorView view{record.vector.data(), static_cast<int>(record.vector.size())};
    Status status = engine->Upsert(record.key, view, record.meta, std::nullopt, record.text);
    if (!status.ok()) {
      return false;
    }
  }
  std::string expected = "{\"queries\":[";
  bool first_query = true;
  for (const auto& query_obj : queries_field->array) {
    if (!first_query) {
      expected += ",";
    }
    first_query = false;
    if (query_obj.type != JsonValue::Type::kObject) {
      return false;
    }
    std::vector<float> vector;
    GetFloatArrayField(query_obj, "vector", &vector);
    std::string text_query;
    bool has_text = GetStringField(query_obj, "text_query", &text_query);
    int topk = 0;
    GetIntField(query_obj, "topk", &topk);
    float alpha = 0.5f;
    const JsonValue* alpha_field = FindField(query_obj, "alpha");
    if (alpha_field && alpha_field->type == JsonValue::Type::kNumber) {
      alpha = static_cast<float>(alpha_field->number);
    }
    QueryPolicy policy;
    const JsonValue* policy_field = FindField(query_obj, "policy");
    if (policy_field && policy_field->type == JsonValue::Type::kObject) {
      policy = ParsePolicy(*policy_field);
    }
    SearchEngine::QueryOptions options;
    options.topk = topk;
    StatusOr<SearchResponse> response_or(Status(StatusCode::kInternal, "uninitialized"));
    if (has_text) {
      SearchEngine::HybridQuery query;
      query.text_query = text_query;
      if (!vector.empty()) {
        query.vector_query = vector;
      }
      query.alpha = alpha;
      response_or = engine->SearchHybridWithExplain(query, options, policy);
    } else {
      VectorView view{vector.data(), static_cast<int>(vector.size())};
      response_or = engine->SearchWithExplain(view, options, policy);
    }
    if (!response_or.ok()) {
      return false;
    }
    expected += "{\"results\":[";
    bool first_result = true;
    for (const auto& item : response_or.value().results) {
      if (!first_result) {
        expected += ",";
      }
      first_result = false;
      expected += "{\"key\":\"" + JsonEscape(item.key) + "\",\"score\":" +
                  std::to_string(item.score) + "}";
    }
    expected += "]}";
  }
  expected += "]}";
  std::string out_path = cfg.out_dir + "/expected.json";
  return WriteFile(out_path, expected);
}

bool HandleContractVerify(const CliConfig& cfg) {
  std::string dataset_path = cfg.contract_dir + "/dataset.json";
  std::string queries_path = cfg.contract_dir + "/queries.json";
  std::string expected_path = cfg.contract_dir + "/expected.json";
  SearchEngineConfig engine_cfg;
  std::vector<SnapshotRecord> records;
  if (!LoadDataset(dataset_path, &engine_cfg, &records)) {
    return false;
  }
  std::string queries_json;
  std::string expected_json;
  if (!ReadFile(queries_path, &queries_json) || !ReadFile(expected_path, &expected_json)) {
    return false;
  }
  JsonValue queries_root;
  JsonValue expected_root;
  std::string error;
  if (!ParseJson(queries_json, &queries_root, &error) ||
      !ParseJson(expected_json, &expected_root, &error)) {
    return false;
  }
  const JsonValue* queries_field = FindField(queries_root, "queries");
  const JsonValue* expected_field = FindField(expected_root, "queries");
  if (!queries_field || !expected_field || queries_field->type != JsonValue::Type::kArray ||
      expected_field->type != JsonValue::Type::kArray) {
    return false;
  }
  auto engine_or = SearchEngine::Open(engine_cfg);
  if (!engine_or.ok()) {
    return false;
  }
  auto engine = std::move(engine_or.value());
  for (const auto& record : records) {
    VectorView view{record.vector.data(), static_cast<int>(record.vector.size())};
    Status status = engine->Upsert(record.key, view, record.meta, std::nullopt, record.text);
    if (!status.ok()) {
      return false;
    }
  }
  if (queries_field->array.size() != expected_field->array.size()) {
    return false;
  }
  for (size_t i = 0; i < queries_field->array.size(); ++i) {
    const auto& query_obj = queries_field->array[i];
    const auto& expected_obj = expected_field->array[i];
    if (query_obj.type != JsonValue::Type::kObject || expected_obj.type != JsonValue::Type::kObject) {
      return false;
    }
    std::vector<float> vector;
    GetFloatArrayField(query_obj, "vector", &vector);
    std::string text_query;
    bool has_text = GetStringField(query_obj, "text_query", &text_query);
    int topk = 0;
    GetIntField(query_obj, "topk", &topk);
    float alpha = 0.5f;
    const JsonValue* alpha_field = FindField(query_obj, "alpha");
    if (alpha_field && alpha_field->type == JsonValue::Type::kNumber) {
      alpha = static_cast<float>(alpha_field->number);
    }
    QueryPolicy policy;
    const JsonValue* policy_field = FindField(query_obj, "policy");
    if (policy_field && policy_field->type == JsonValue::Type::kObject) {
      policy = ParsePolicy(*policy_field);
    }
    SearchEngine::QueryOptions options;
    options.topk = topk;
    StatusOr<SearchResponse> response_or(Status(StatusCode::kInternal, "uninitialized"));
    if (has_text) {
      SearchEngine::HybridQuery query;
      query.text_query = text_query;
      if (!vector.empty()) {
        query.vector_query = vector;
      }
      query.alpha = alpha;
      response_or = engine->SearchHybridWithExplain(query, options, policy);
    } else {
      VectorView view{vector.data(), static_cast<int>(vector.size())};
      response_or = engine->SearchWithExplain(view, options, policy);
    }
    if (!response_or.ok()) {
      return false;
    }
    const JsonValue* results_field = FindField(expected_obj, "results");
    if (!results_field || results_field->type != JsonValue::Type::kArray) {
      return false;
    }
    if (results_field->array.size() != response_or.value().results.size()) {
      return false;
    }
    for (size_t r = 0; r < results_field->array.size(); ++r) {
      const auto& expected_result = results_field->array[r];
      if (expected_result.type != JsonValue::Type::kObject) {
        return false;
      }
      std::string key;
      if (!GetStringField(expected_result, "key", &key)) {
        return false;
      }
      const JsonValue* score_field = FindField(expected_result, "score");
      if (!score_field || score_field->type != JsonValue::Type::kNumber) {
        return false;
      }
      if (response_or.value().results[r].key != key) {
        return false;
      }
      float expected_score = static_cast<float>(score_field->number);
      float actual_score = response_or.value().results[r].score;
      if (std::fabs(expected_score - actual_score) > 1e-4f) {
        return false;
      }
    }
  }
  return true;
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
    std::cerr << "Usage: pomai-search --command upsert|search|search_hybrid|delete|contract-record|contract-verify "
                 "[--host] [--port]\n";
    return 1;
  }
  if (cfg.command == "contract-record") {
    if (cfg.dataset_path.empty() || cfg.queries_path.empty() || cfg.out_dir.empty()) {
      std::cerr << "contract-record requires --dataset --queries --out\n";
      return 1;
    }
    if (!pomai_search::HandleContractRecord(cfg)) {
      std::cerr << "contract record failed\n";
      return 1;
    }
    return 0;
  }
  if (cfg.command == "contract-verify") {
    if (cfg.contract_dir.empty()) {
      std::cerr << "contract-verify requires --dir\n";
      return 1;
    }
    if (!pomai_search::HandleContractVerify(cfg)) {
      std::cerr << "contract verify failed\n";
      return 1;
    }
    return 0;
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
