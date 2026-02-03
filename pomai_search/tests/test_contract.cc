#include "pomai_search/net/json.h"
#include "pomai_search/search_engine.h"
#include "pomai_search/snapshot.h"
#include "tests/test_framework.h"

#include <fstream>
#include <sstream>
#include <string>

namespace pomai_search::test {
namespace {

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
  int contract_version = 0;
  if (GetIntField(object, "contract_version", &contract_version)) {
    cfg->contract_version = static_cast<uint32_t>(contract_version);
  }
  const JsonValue* seed_field = FindField(object, "global_seed");
  if (seed_field && seed_field->type == JsonValue::Type::kNumber) {
    cfg->global_seed = static_cast<uint64_t>(seed_field->number);
  }
  return true;
}

bool ParseRecords(const JsonValue& records, std::vector<SnapshotRecord>* out) {
  if (records.type != JsonValue::Type::kArray) {
    return false;
  }
  for (const auto& record : records.array) {
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
    out->push_back(std::move(item));
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

}  // namespace

POMAI_TEST(ContractQueries) {
  std::string dataset_json;
  std::string queries_json;
  std::string expected_json;
  EXPECT_TRUE(ReadFile("tests/contracts/dataset.json", &dataset_json));
  EXPECT_TRUE(ReadFile("tests/contracts/queries.json", &queries_json));
  EXPECT_TRUE(ReadFile("tests/contracts/expected.json", &expected_json));

  JsonValue dataset_root;
  std::string error;
  EXPECT_TRUE(ParseJson(dataset_json, &dataset_root, &error));
  const JsonValue* config_field = FindField(dataset_root, "config");
  const JsonValue* records_field = FindField(dataset_root, "records");
  EXPECT_TRUE(config_field != nullptr && records_field != nullptr);
  SearchEngineConfig cfg;
  EXPECT_TRUE(ParseConfig(*config_field, &cfg));
  std::vector<SnapshotRecord> records;
  EXPECT_TRUE(ParseRecords(*records_field, &records));

  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  for (const auto& record : records) {
    VectorView view{record.vector.data(), static_cast<int>(record.vector.size())};
    Status status = engine->Upsert(record.key, view, record.meta, std::nullopt, record.text);
    EXPECT_TRUE(status.ok());
  }

  JsonValue queries_root;
  EXPECT_TRUE(ParseJson(queries_json, &queries_root, &error));
  const JsonValue* queries_field = FindField(queries_root, "queries");
  EXPECT_TRUE(queries_field != nullptr && queries_field->type == JsonValue::Type::kArray);

  JsonValue expected_root;
  EXPECT_TRUE(ParseJson(expected_json, &expected_root, &error));
  const JsonValue* expected_field = FindField(expected_root, "queries");
  EXPECT_TRUE(expected_field != nullptr && expected_field->type == JsonValue::Type::kArray);
  EXPECT_EQ(queries_field->array.size(), expected_field->array.size());

  for (size_t i = 0; i < queries_field->array.size(); ++i) {
    const auto& query_obj = queries_field->array[i];
    const auto& expected_obj = expected_field->array[i];
    EXPECT_TRUE(query_obj.type == JsonValue::Type::kObject);
    EXPECT_TRUE(expected_obj.type == JsonValue::Type::kObject);

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

    StatusOr<SearchResponse> response_or;
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
    EXPECT_TRUE(response_or.ok());
    const auto& response = response_or.value();

    const JsonValue* results_field = FindField(expected_obj, "results");
    EXPECT_TRUE(results_field != nullptr && results_field->type == JsonValue::Type::kArray);
    EXPECT_EQ(response.results.size(), results_field->array.size());

    for (size_t r = 0; r < response.results.size(); ++r) {
      const auto& expected_result = results_field->array[r];
      EXPECT_TRUE(expected_result.type == JsonValue::Type::kObject);
      std::string key;
      EXPECT_TRUE(GetStringField(expected_result, "key", &key));
      const JsonValue* score_field = FindField(expected_result, "score");
      EXPECT_TRUE(score_field && score_field->type == JsonValue::Type::kNumber);
      EXPECT_EQ(response.results[r].key, key);
      EXPECT_NEAR(response.results[r].score, static_cast<float>(score_field->number), 1e-4f);
    }
  }

  return true;
}

}  // namespace pomai_search::test
