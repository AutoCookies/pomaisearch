#include "pomai_search/c_api.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/serialize/snapshot.h"
#include "pomai_search/logging.h"
#include "pomai_search/search_engine.h"
#include "pomai_search/types.h"

namespace pomai_search {
namespace {

struct LastErrorState {
  pomai_search_status_code_t code = POMAI_SEARCH_STATUS_OK;
  std::string message;
};

thread_local LastErrorState g_last_error;

void SetLastError(pomai_search_status_code_t code, const std::string& message) {
  g_last_error.code = code;
  g_last_error.message = message;
}

void ClearLastError() {
  SetLastError(POMAI_SEARCH_STATUS_OK, "");
}

pomai_search_status_code_t StatusCodeToC(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return POMAI_SEARCH_STATUS_OK;
    case StatusCode::kInvalidArgument:
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    case StatusCode::kNotFound:
      return POMAI_SEARCH_STATUS_NOT_FOUND;
    case StatusCode::kAlreadyExists:
      return POMAI_SEARCH_STATUS_ALREADY_EXISTS;
    case StatusCode::kResourceExhausted:
      return POMAI_SEARCH_STATUS_RESOURCE_EXHAUSTED;
    case StatusCode::kInternal:
      return POMAI_SEARCH_STATUS_INTERNAL;
    case StatusCode::kUnavailable:
      return POMAI_SEARCH_STATUS_UNAVAILABLE;
    default:
      return POMAI_SEARCH_STATUS_INTERNAL;
  }
}

pomai_search_status_code_t StatusToC(const Status& status) {
  if (status.ok()) {
    ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  }
  auto code = StatusCodeToC(status.code());
  SetLastError(code, status.message());
  return code;
}

template <typename Func>
pomai_search_status_code_t Guard(Func&& func) {
  try {
    return func();
  } catch (const std::bad_alloc& e) {
    SetLastError(POMAI_SEARCH_STATUS_OUT_OF_MEMORY, e.what());
    return POMAI_SEARCH_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& e) {
    SetLastError(POMAI_SEARCH_STATUS_INTERNAL, e.what());
    return POMAI_SEARCH_STATUS_INTERNAL;
  } catch (...) {
    SetLastError(POMAI_SEARCH_STATUS_INTERNAL, "unknown error");
    return POMAI_SEARCH_STATUS_INTERNAL;
  }
}

std::string_view StringViewFromBytes(const char* data, size_t len) {
  if (!data) {
    return {};
  }
  if (len == 0) {
    return std::string_view(data);
  }
  return std::string_view(data, len);
}

pomai_search_metric_t SimilarityToMetric(SearchEngineConfig::Similarity similarity) {
  switch (similarity) {
    case SearchEngineConfig::Similarity::Cosine:
      return POMAI_SEARCH_METRIC_COSINE;
    case SearchEngineConfig::Similarity::Dot:
    default:
      return POMAI_SEARCH_METRIC_DOT;
  }
}

SearchEngineConfig::Similarity MetricToSimilarity(pomai_search_metric_t metric) {
  switch (metric) {
    case POMAI_SEARCH_METRIC_COSINE:
      return SearchEngineConfig::Similarity::Cosine;
    case POMAI_SEARCH_METRIC_DOT:
    default:
      return SearchEngineConfig::Similarity::Dot;
  }
}

SearchEngineConfig::IndexType IndexTypeFromC(pomai_search_index_type_t type) {
  switch (type) {
    case POMAI_SEARCH_INDEX_HNSW:
      return SearchEngineConfig::IndexType::Hnsw;
    case POMAI_SEARCH_INDEX_IVF_FLAT:
      return SearchEngineConfig::IndexType::IvfFlat;
    case POMAI_SEARCH_INDEX_IVF_SQ8:
      return SearchEngineConfig::IndexType::IvfSq8;
    case POMAI_SEARCH_INDEX_FLAT:
    default:
      return SearchEngineConfig::IndexType::Flat;
  }
}

pomai_search_index_type_t IndexTypeToC(SearchEngineConfig::IndexType type) {
  switch (type) {
    case SearchEngineConfig::IndexType::Hnsw:
      return POMAI_SEARCH_INDEX_HNSW;
    case SearchEngineConfig::IndexType::IvfFlat:
      return POMAI_SEARCH_INDEX_IVF_FLAT;
    case SearchEngineConfig::IndexType::IvfSq8:
      return POMAI_SEARCH_INDEX_IVF_SQ8;
    case SearchEngineConfig::IndexType::Flat:
    default:
      return POMAI_SEARCH_INDEX_FLAT;
  }
}

Metadata MetadataFromC(const pomai_search_metadata_kv_t* metadata, size_t count) {
  Metadata result;
  if (!metadata) {
    return result;
  }
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const char* key = metadata[i].key;
    const char* value = metadata[i].value;
    if (!key || !value) {
      continue;
    }
    result.emplace(std::string(key), std::string(value));
  }
  return result;
}

Filter FilterFromC(const pomai_search_filter_t* filter) {
  Filter out;
  if (!filter) {
    return out;
  }
  if (filter->tag) {
    out.tag = filter->tag;
  }
  if (filter->source) {
    out.source = filter->source;
  }
  if (filter->lang) {
    out.lang = filter->lang;
  }
  return out;
}

int64_t RemainingTtlMs(const std::optional<std::chrono::system_clock::time_point>& expiry) {
  if (!expiry.has_value()) {
    return -1;
  }
  auto now = std::chrono::system_clock::now();
  auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(expiry.value() - now);
  return std::max<int64_t>(0, remaining.count());
}

struct EngineHandle {
  std::unique_ptr<SearchEngine> engine;
};

struct ResultsHandle {
  std::vector<ResultItem> items;
  std::vector<std::vector<pomai_search_metadata_kv_t>> metadata;
};

struct IterHandle {
  std::vector<SnapshotRecord> records;
  std::vector<std::vector<pomai_search_metadata_kv_t>> metadata;
  size_t index = 0;
};

}  // namespace
}  // namespace pomai_search

using pomai_search::EngineHandle;
using pomai_search::FilterFromC;
using pomai_search::Guard;
using pomai_search::IndexTypeFromC;
using pomai_search::IndexTypeToC;
using pomai_search::MetadataFromC;
using pomai_search::MetricToSimilarity;
using pomai_search::RemainingTtlMs;
using pomai_search::ResultsHandle;
using pomai_search::SetLastError;
using pomai_search::StatusToC;
using pomai_search::StringViewFromBytes;
using pomai_search::SimilarityToMetric;
using pomai_search::IterHandle;

extern "C" {

pomai_search_version_t pomai_search_c_version(void) {
  pomai_search_version_t version{};
  version.struct_size = sizeof(pomai_search_version_t);
  version.abi_version = POMAI_SEARCH_C_ABI_VERSION;
  version.major = POMAI_SEARCH_C_API_VERSION_MAJOR;
  version.minor = POMAI_SEARCH_C_API_VERSION_MINOR;
  version.patch = POMAI_SEARCH_C_API_VERSION_PATCH;
#ifdef POMAI_SEARCH_GIT_SHA
  version.git_sha = POMAI_SEARCH_GIT_SHA;
#else
  version.git_sha = "unknown";
#endif
  return version;
}

void pomai_search_engine_config_init(pomai_search_engine_config_t* config) {
  if (!config) {
    return;
  }
  pomai_search::SearchEngineConfig defaults;
  config->struct_size = sizeof(pomai_search_engine_config_t);
  config->dim = defaults.dim;
  config->num_shards = defaults.num_shards;
  config->metric = SimilarityToMetric(defaults.similarity);
  config->topk_default = defaults.topk_default;
  config->max_points_per_shard = defaults.max_points_per_shard;
  config->enable_avx2 = defaults.enable_avx2 ? 1 : 0;
  config->query_threads = defaults.query_threads;
  config->ingest_threads = defaults.ingest_threads;
  config->memory_alignment = defaults.memory_alignment;
  config->index_type = IndexTypeToC(defaults.index_type);
  config->hnsw_m = defaults.hnsw_m;
  config->hnsw_ef_construction = defaults.hnsw_ef_construction;
  config->hnsw_ef_search = defaults.hnsw_ef_search;
  config->hnsw_seed = defaults.hnsw_seed;
  config->ivf_nlist = defaults.ivf_nlist;
  config->ivf_nprobe = defaults.ivf_nprobe;
  config->global_seed = defaults.global_seed;
  config->contract_version = defaults.contract_version;
}

void pomai_search_query_options_init(pomai_search_query_options_t* options) {
  if (!options) {
    return;
  }
  options->struct_size = sizeof(pomai_search_query_options_t);
  options->topk = 0;
  pomai_search_filter_init(&options->filter);
}

void pomai_search_filter_init(pomai_search_filter_t* filter) {
  if (!filter) {
    return;
  }
  filter->struct_size = sizeof(pomai_search_filter_t);
  filter->tag = nullptr;
  filter->source = nullptr;
  filter->lang = nullptr;
}

void pomai_search_document_init(pomai_search_document_t* document) {
  if (!document) {
    return;
  }
  document->struct_size = sizeof(pomai_search_document_t);
  document->doc_id = nullptr;
  document->doc_id_len = 0;
  document->vector = nullptr;
  document->dim = 0;
  document->metadata = nullptr;
  document->metadata_count = 0;
  document->ttl_ms = -1;
  document->text = nullptr;
  document->text_len = 0;
}

void pomai_search_hybrid_query_init(pomai_search_hybrid_query_t* query) {
  if (!query) {
    return;
  }
  query->struct_size = sizeof(pomai_search_hybrid_query_t);
  query->text = nullptr;
  query->text_len = 0;
  query->vector = nullptr;
  query->dim = 0;
  query->alpha = 0.5f;
}

void pomai_search_iter_options_init(pomai_search_iter_options_t* options) {
  if (!options) {
    return;
  }
  options->struct_size = sizeof(pomai_search_iter_options_t);
  pomai_search_filter_init(&options->filter);
}

void pomai_search_doc_view_init(pomai_search_doc_view_t* view) {
  if (!view) {
    return;
  }
  view->struct_size = sizeof(pomai_search_doc_view_t);
  view->doc_id = nullptr;
  view->doc_id_len = 0;
  view->vector = nullptr;
  view->dim = 0;
  view->metadata = nullptr;
  view->metadata_count = 0;
  view->ttl_ms_remaining = -1;
  view->text = nullptr;
  view->text_len = 0;
}

void pomai_search_hit_init(pomai_search_hit_t* hit) {
  if (!hit) {
    return;
  }
  hit->struct_size = sizeof(pomai_search_hit_t);
  hit->doc_id = nullptr;
  hit->doc_id_len = 0;
  hit->score = 0.0f;
  hit->metadata = nullptr;
  hit->metadata_count = 0;
}

pomai_search_status_code_t pomai_search_engine_open(
    const pomai_search_engine_config_t* config, pomai_search_engine_t** out_engine) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!config || !out_engine) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "config and out_engine required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (config->struct_size < sizeof(pomai_search_engine_config_t)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "config struct_size too small");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    pomai_search::SearchEngineConfig cfg;
    cfg.dim = config->dim;
    cfg.num_shards = config->num_shards;
    cfg.similarity = MetricToSimilarity(config->metric);
    cfg.topk_default = config->topk_default;
    cfg.max_points_per_shard = config->max_points_per_shard;
    cfg.enable_avx2 = config->enable_avx2 != 0;
    cfg.query_threads = config->query_threads;
    cfg.ingest_threads = config->ingest_threads;
    cfg.memory_alignment = config->memory_alignment;
    cfg.index_type = IndexTypeFromC(config->index_type);
    cfg.hnsw_m = config->hnsw_m;
    cfg.hnsw_ef_construction = config->hnsw_ef_construction;
    cfg.hnsw_ef_search = config->hnsw_ef_search;
    cfg.hnsw_seed = config->hnsw_seed;
    cfg.ivf_nlist = config->ivf_nlist;
    cfg.ivf_nprobe = config->ivf_nprobe;
    cfg.global_seed = config->global_seed;
    cfg.contract_version = config->contract_version;

    auto engine_or = pomai_search::SearchEngine::Open(cfg);
    if (!engine_or.ok()) {
      return StatusToC(engine_or.status());
    }
    auto handle = std::make_unique<EngineHandle>();
    handle->engine = std::move(engine_or.value());
    *out_engine = reinterpret_cast<pomai_search_engine_t*>(handle.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_engine_close(pomai_search_engine_t* engine) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine) {
      pomai_search::ClearLastError();
      return POMAI_SEARCH_STATUS_OK;
    }
    auto handle = std::unique_ptr<EngineHandle>(reinterpret_cast<EngineHandle*>(engine));
    if (!handle->engine) {
      pomai_search::ClearLastError();
      return POMAI_SEARCH_STATUS_OK;
    }
    auto status = handle->engine->Close();
    return StatusToC(status);
  });
}

pomai_search_status_code_t pomai_search_engine_upsert(pomai_search_engine_t* engine,
                                                     const pomai_search_document_t* document) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !document) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine and document required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (document->struct_size < sizeof(pomai_search_document_t)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "document struct_size too small");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (!document->doc_id || !document->vector) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "doc_id and vector required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (document->dim == 0) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "dimension required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    if (handle->engine->Dim() != static_cast<int>(document->dim)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "dimension mismatch");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    std::string_view key = StringViewFromBytes(document->doc_id, document->doc_id_len);
    pomai_search::VectorView view{document->vector, static_cast<int>(document->dim)};
    pomai_search::Metadata meta = MetadataFromC(document->metadata, document->metadata_count);
    std::optional<std::chrono::milliseconds> ttl;
    if (document->ttl_ms >= 0) {
      ttl = std::chrono::milliseconds(document->ttl_ms);
    }
    std::optional<std::string> text;
    if (document->text) {
      text = std::string(StringViewFromBytes(document->text, document->text_len));
    }
    auto status = handle->engine->Upsert(key, view, std::move(meta), ttl, std::move(text));
    return StatusToC(status);
  });
}

pomai_search_status_code_t pomai_search_engine_delete(pomai_search_engine_t* engine,
                                                     const char* doc_id, size_t doc_id_len) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !doc_id) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine and doc_id required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    auto status = handle->engine->Delete(StringViewFromBytes(doc_id, doc_id_len));
    return StatusToC(status);
  });
}

pomai_search_status_code_t pomai_search_engine_search(
    pomai_search_engine_t* engine, const float* query, uint32_t dim,
    const pomai_search_query_options_t* options, pomai_search_results_t** out_results) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !query || !out_results) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine, query, out_results required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (dim == 0) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "dimension required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    if (handle->engine->Dim() != static_cast<int>(dim)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "dimension mismatch");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    pomai_search::SearchEngine::QueryOptions opt;
    if (options) {
      if (options->struct_size < sizeof(pomai_search_query_options_t)) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "options struct_size too small");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      opt.topk = options->topk;
      opt.filter = FilterFromC(&options->filter);
    }
    pomai_search::VectorView view{query, static_cast<int>(dim)};
    auto results_or = handle->engine->Search(view, opt);
    if (!results_or.ok()) {
      return StatusToC(results_or.status());
    }
    auto results = std::make_unique<ResultsHandle>();
    results->items = std::move(results_or.value());
    results->metadata.reserve(results->items.size());
    for (const auto& item : results->items) {
      std::vector<pomai_search_metadata_kv_t> kvs;
      kvs.reserve(item.meta.size());
      for (const auto& pair : item.meta) {
        kvs.push_back({pair.first.c_str(), pair.second.c_str()});
      }
      results->metadata.push_back(std::move(kvs));
    }
    *out_results = reinterpret_cast<pomai_search_results_t*>(results.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_engine_search_by_key(
    pomai_search_engine_t* engine, const char* doc_id, size_t doc_id_len,
    const pomai_search_query_options_t* options, pomai_search_results_t** out_results) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !doc_id || !out_results) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine, doc_id, out_results required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    pomai_search::SearchEngine::QueryOptions opt;
    if (options) {
      if (options->struct_size < sizeof(pomai_search_query_options_t)) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "options struct_size too small");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      opt.topk = options->topk;
      opt.filter = FilterFromC(&options->filter);
    }
    auto results_or = handle->engine->SearchByKey(StringViewFromBytes(doc_id, doc_id_len), opt);
    if (!results_or.ok()) {
      return StatusToC(results_or.status());
    }
    auto results = std::make_unique<ResultsHandle>();
    results->items = std::move(results_or.value());
    results->metadata.reserve(results->items.size());
    for (const auto& item : results->items) {
      std::vector<pomai_search_metadata_kv_t> kvs;
      kvs.reserve(item.meta.size());
      for (const auto& pair : item.meta) {
        kvs.push_back({pair.first.c_str(), pair.second.c_str()});
      }
      results->metadata.push_back(std::move(kvs));
    }
    *out_results = reinterpret_cast<pomai_search_results_t*>(results.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_engine_hybrid_search(
    pomai_search_engine_t* engine, const pomai_search_hybrid_query_t* query,
    const pomai_search_query_options_t* options, pomai_search_results_t** out_results) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !query || !out_results) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine, query, out_results required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (query->struct_size < sizeof(pomai_search_hybrid_query_t)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "query struct_size too small");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    pomai_search::SearchEngine::QueryOptions opt;
    if (options) {
      if (options->struct_size < sizeof(pomai_search_query_options_t)) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "options struct_size too small");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      opt.topk = options->topk;
      opt.filter = FilterFromC(&options->filter);
    }
    pomai_search::SearchEngine::HybridQuery hybrid{};
    if (query->text) {
      hybrid.text_query = std::string(StringViewFromBytes(query->text, query->text_len));
    }
    if (query->vector) {
      if (query->dim == 0) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "vector dim required");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      if (handle->engine->Dim() != static_cast<int>(query->dim)) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "dimension mismatch");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      hybrid.vector_query = std::vector<float>(query->vector, query->vector + query->dim);
    }
    hybrid.alpha = query->alpha;
    auto results_or = handle->engine->SearchHybrid(hybrid, opt);
    if (!results_or.ok()) {
      return StatusToC(results_or.status());
    }
    auto results = std::make_unique<ResultsHandle>();
    results->items = std::move(results_or.value());
    results->metadata.reserve(results->items.size());
    for (const auto& item : results->items) {
      std::vector<pomai_search_metadata_kv_t> kvs;
      kvs.reserve(item.meta.size());
      for (const auto& pair : item.meta) {
        kvs.push_back({pair.first.c_str(), pair.second.c_str()});
      }
      results->metadata.push_back(std::move(kvs));
    }
    *out_results = reinterpret_cast<pomai_search_results_t*>(results.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_results_count(const pomai_search_results_t* results,
                                                      uint32_t* out_count) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!results || !out_count) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "results and out_count required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<const ResultsHandle*>(results);
    *out_count = static_cast<uint32_t>(handle->items.size());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_results_get(const pomai_search_results_t* results,
                                                    uint32_t index, pomai_search_hit_t* out_hit) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!results || !out_hit) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "results and out_hit required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (out_hit->struct_size < sizeof(pomai_search_hit_t)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "hit struct_size too small");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<const ResultsHandle*>(results);
    if (index >= handle->items.size()) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "index out of range");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    const auto& item = handle->items[index];
    out_hit->doc_id = item.key.c_str();
    out_hit->doc_id_len = item.key.size();
    out_hit->score = item.score;
    if (index < handle->metadata.size()) {
      const auto& kvs = handle->metadata[index];
      out_hit->metadata = kvs.empty() ? nullptr : kvs.data();
      out_hit->metadata_count = kvs.size();
    } else {
      out_hit->metadata = nullptr;
      out_hit->metadata_count = 0;
    }
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

void pomai_search_results_free(pomai_search_results_t* results) {
  Guard([&]() -> pomai_search_status_code_t {
    delete reinterpret_cast<ResultsHandle*>(results);
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_engine_iter_begin(
    pomai_search_engine_t* engine, const pomai_search_iter_options_t* options,
    pomai_search_iter_t** out_iter) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !out_iter) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine and out_iter required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    pomai_search::Filter filter;
    if (options) {
      if (options->struct_size < sizeof(pomai_search_iter_options_t)) {
        SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "options struct_size too small");
        return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
      }
      filter = FilterFromC(&options->filter);
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    auto records_or = handle->engine->ExportRecords(filter);
    if (!records_or.ok()) {
      return StatusToC(records_or.status());
    }
    auto iter = std::make_unique<IterHandle>();
    iter->records = std::move(records_or.value());
    iter->metadata.reserve(iter->records.size());
    for (const auto& record : iter->records) {
      std::vector<pomai_search_metadata_kv_t> kvs;
      kvs.reserve(record.meta.size());
      for (const auto& pair : record.meta) {
        kvs.push_back({pair.first.c_str(), pair.second.c_str()});
      }
      iter->metadata.push_back(std::move(kvs));
    }
    *out_iter = reinterpret_cast<pomai_search_iter_t*>(iter.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_iter_next(pomai_search_iter_t* iter,
                                                  pomai_search_doc_view_t* out_doc,
                                                  uint8_t* out_has_value) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!iter || !out_doc || !out_has_value) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT,
                   "iter, out_doc, out_has_value required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    if (out_doc->struct_size < sizeof(pomai_search_doc_view_t)) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "doc_view struct_size too small");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<IterHandle*>(iter);
    if (handle->index >= handle->records.size()) {
      *out_has_value = 0;
      pomai_search::ClearLastError();
      return POMAI_SEARCH_STATUS_OK;
    }
    const auto& record = handle->records[handle->index];
    out_doc->doc_id = record.key.c_str();
    out_doc->doc_id_len = record.key.size();
    out_doc->vector = record.vector.data();
    out_doc->dim = static_cast<uint32_t>(record.vector.size());
    if (handle->index < handle->metadata.size()) {
      const auto& kvs = handle->metadata[handle->index];
      out_doc->metadata = kvs.empty() ? nullptr : kvs.data();
      out_doc->metadata_count = kvs.size();
    } else {
      out_doc->metadata = nullptr;
      out_doc->metadata_count = 0;
    }
    out_doc->ttl_ms_remaining = RemainingTtlMs(record.expiry);
    if (record.text) {
      out_doc->text = record.text->c_str();
      out_doc->text_len = record.text->size();
    } else {
      out_doc->text = nullptr;
      out_doc->text_len = 0;
    }
    ++handle->index;
    *out_has_value = 1;
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

void pomai_search_iter_close(pomai_search_iter_t* iter) {
  Guard([&]() -> pomai_search_status_code_t {
    delete reinterpret_cast<IterHandle*>(iter);
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

pomai_search_status_code_t pomai_search_engine_save(pomai_search_engine_t* engine,
                                                    const char* path, size_t path_len) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!engine || !path) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "engine and path required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto* handle = reinterpret_cast<EngineHandle*>(engine);
    auto status = pomai_search::SnapshotWriter::Write(*handle->engine,
                                                      StringViewFromBytes(path, path_len));
    return StatusToC(status);
  });
}

pomai_search_status_code_t pomai_search_engine_load(const char* path, size_t path_len,
                                                    pomai_search_engine_t** out_engine) {
  return Guard([&]() -> pomai_search_status_code_t {
    if (!path || !out_engine) {
      SetLastError(POMAI_SEARCH_STATUS_INVALID_ARGUMENT, "path and out_engine required");
      return POMAI_SEARCH_STATUS_INVALID_ARGUMENT;
    }
    auto engine_or = pomai_search::SnapshotReader::Read(StringViewFromBytes(path, path_len));
    if (!engine_or.ok()) {
      return StatusToC(engine_or.status());
    }
    auto handle = std::make_unique<EngineHandle>();
    handle->engine = std::move(engine_or.value());
    *out_engine = reinterpret_cast<pomai_search_engine_t*>(handle.release());
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

const char* pomai_search_last_error_message(void) {
  return pomai_search::g_last_error.message.c_str();
}

pomai_search_status_code_t pomai_search_last_error_code(void) {
  return pomai_search::g_last_error.code;
}

void pomai_search_set_log_callback(pomai_search_log_callback_t callback, void* user_ctx) {
  Guard([&]() -> pomai_search_status_code_t {
    pomai_search::Logger::Instance().set_callback(
        [callback, user_ctx](pomai_search::LogLevel level, const std::string& message) {
          if (!callback) {
            return;
          }
          callback(user_ctx, static_cast<int32_t>(level), message.c_str(), message.size());
        });
    pomai_search::ClearLastError();
    return POMAI_SEARCH_STATUS_OK;
  });
}

}  // extern "C"
