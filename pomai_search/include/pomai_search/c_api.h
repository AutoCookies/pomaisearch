#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "pomai_search/c_api_version.h"

#if defined(_WIN32) && defined(POMAI_SEARCH_C_API_SHARED)
#ifdef pomai_search_c_EXPORTS
#define POMAI_SEARCH_C_API __declspec(dllexport)
#else
#define POMAI_SEARCH_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(POMAI_SEARCH_C_API_SHARED)
#define POMAI_SEARCH_C_API __attribute__((visibility("default")))
#else
#define POMAI_SEARCH_C_API
#endif

typedef struct pomai_search_engine pomai_search_engine_t;
typedef struct pomai_search_results pomai_search_results_t;
typedef struct pomai_search_iter pomai_search_iter_t;

typedef enum pomai_search_status_code_t {
  POMAI_SEARCH_STATUS_OK = 0,
  POMAI_SEARCH_STATUS_INVALID_ARGUMENT = 1,
  POMAI_SEARCH_STATUS_NOT_FOUND = 2,
  POMAI_SEARCH_STATUS_ALREADY_EXISTS = 3,
  POMAI_SEARCH_STATUS_RESOURCE_EXHAUSTED = 4,
  POMAI_SEARCH_STATUS_INTERNAL = 5,
  POMAI_SEARCH_STATUS_UNAVAILABLE = 6,
  POMAI_SEARCH_STATUS_IO_ERROR = 7,
  POMAI_SEARCH_STATUS_OUT_OF_MEMORY = 8,
  POMAI_SEARCH_STATUS_UNSUPPORTED = 9,
  POMAI_SEARCH_STATUS_TIMEOUT = 10
} pomai_search_status_code_t;

typedef enum pomai_search_metric_t {
  POMAI_SEARCH_METRIC_DOT = 0,
  POMAI_SEARCH_METRIC_COSINE = 1
} pomai_search_metric_t;

typedef enum pomai_search_index_type_t {
  POMAI_SEARCH_INDEX_FLAT = 0,
  POMAI_SEARCH_INDEX_HNSW = 1,
  POMAI_SEARCH_INDEX_IVF_FLAT = 2,
  POMAI_SEARCH_INDEX_IVF_SQ8 = 3
} pomai_search_index_type_t;

typedef struct pomai_search_version_t {
  uint32_t struct_size;
  int32_t abi_version;
  int32_t major;
  int32_t minor;
  int32_t patch;
  const char* git_sha;
} pomai_search_version_t;

typedef struct pomai_search_engine_config_t {
  uint32_t struct_size;
  int32_t dim;
  int32_t num_shards;
  pomai_search_metric_t metric;
  int32_t topk_default;
  size_t max_points_per_shard;
  uint8_t enable_avx2;
  int32_t query_threads;
  int32_t ingest_threads;
  size_t memory_alignment;
  pomai_search_index_type_t index_type;
  int32_t hnsw_m;
  int32_t hnsw_ef_construction;
  int32_t hnsw_ef_search;
  uint32_t hnsw_seed;
  int32_t ivf_nlist;
  int32_t ivf_nprobe;
  uint64_t global_seed;
  uint32_t contract_version;
} pomai_search_engine_config_t;

typedef struct pomai_search_metadata_kv_t {
  const char* key;
  const char* value;
} pomai_search_metadata_kv_t;

typedef struct pomai_search_document_t {
  uint32_t struct_size;
  const char* doc_id;
  size_t doc_id_len;
  const float* vector;
  uint32_t dim;
  const pomai_search_metadata_kv_t* metadata;
  size_t metadata_count;
  int64_t ttl_ms;
  const char* text;
  size_t text_len;
} pomai_search_document_t;

typedef struct pomai_search_filter_t {
  uint32_t struct_size;
  const char* tag;
  const char* source;
  const char* lang;
} pomai_search_filter_t;

typedef struct pomai_search_query_options_t {
  uint32_t struct_size;
  int32_t topk;
  pomai_search_filter_t filter;
} pomai_search_query_options_t;

typedef struct pomai_search_hybrid_query_t {
  uint32_t struct_size;
  const char* text;
  size_t text_len;
  const float* vector;
  uint32_t dim;
  float alpha;
} pomai_search_hybrid_query_t;

typedef struct pomai_search_hit_t {
  uint32_t struct_size;
  const char* doc_id;
  size_t doc_id_len;
  float score;
  const pomai_search_metadata_kv_t* metadata;
  size_t metadata_count;
} pomai_search_hit_t;

typedef struct pomai_search_iter_options_t {
  uint32_t struct_size;
  pomai_search_filter_t filter;
} pomai_search_iter_options_t;

typedef struct pomai_search_doc_view_t {
  uint32_t struct_size;
  const char* doc_id;
  size_t doc_id_len;
  const float* vector;
  uint32_t dim;
  const pomai_search_metadata_kv_t* metadata;
  size_t metadata_count;
  int64_t ttl_ms_remaining;
  const char* text;
  size_t text_len;
} pomai_search_doc_view_t;

typedef void (*pomai_search_log_callback_t)(void* user_ctx, int32_t level,
                                            const char* message, size_t message_len);

POMAI_SEARCH_C_API pomai_search_version_t pomai_search_c_version(void);

POMAI_SEARCH_C_API void pomai_search_engine_config_init(pomai_search_engine_config_t* config);
POMAI_SEARCH_C_API void pomai_search_query_options_init(pomai_search_query_options_t* options);
POMAI_SEARCH_C_API void pomai_search_filter_init(pomai_search_filter_t* filter);
POMAI_SEARCH_C_API void pomai_search_document_init(pomai_search_document_t* document);
POMAI_SEARCH_C_API void pomai_search_hybrid_query_init(pomai_search_hybrid_query_t* query);
POMAI_SEARCH_C_API void pomai_search_iter_options_init(pomai_search_iter_options_t* options);
POMAI_SEARCH_C_API void pomai_search_doc_view_init(pomai_search_doc_view_t* view);
POMAI_SEARCH_C_API void pomai_search_hit_init(pomai_search_hit_t* hit);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_open(const pomai_search_engine_config_t* config,
                         pomai_search_engine_t** out_engine);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_close(pomai_search_engine_t* engine);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_upsert(pomai_search_engine_t* engine,
                           const pomai_search_document_t* document);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_delete(pomai_search_engine_t* engine, const char* doc_id, size_t doc_id_len);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_search(pomai_search_engine_t* engine, const float* query, uint32_t dim,
                           const pomai_search_query_options_t* options,
                           pomai_search_results_t** out_results);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_search_by_key(pomai_search_engine_t* engine, const char* doc_id,
                                  size_t doc_id_len, const pomai_search_query_options_t* options,
                                  pomai_search_results_t** out_results);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_hybrid_search(pomai_search_engine_t* engine,
                                  const pomai_search_hybrid_query_t* query,
                                  const pomai_search_query_options_t* options,
                                  pomai_search_results_t** out_results);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_results_count(const pomai_search_results_t* results, uint32_t* out_count);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_results_get(const pomai_search_results_t* results, uint32_t index,
                         pomai_search_hit_t* out_hit);
POMAI_SEARCH_C_API void pomai_search_results_free(pomai_search_results_t* results);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_iter_begin(pomai_search_engine_t* engine,
                               const pomai_search_iter_options_t* options,
                               pomai_search_iter_t** out_iter);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_iter_next(pomai_search_iter_t* iter, pomai_search_doc_view_t* out_doc,
                       uint8_t* out_has_value);
POMAI_SEARCH_C_API void pomai_search_iter_close(pomai_search_iter_t* iter);

POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_save(pomai_search_engine_t* engine, const char* path, size_t path_len);
POMAI_SEARCH_C_API pomai_search_status_code_t
pomai_search_engine_load(const char* path, size_t path_len, pomai_search_engine_t** out_engine);

POMAI_SEARCH_C_API const char* pomai_search_last_error_message(void);
POMAI_SEARCH_C_API pomai_search_status_code_t pomai_search_last_error_code(void);

POMAI_SEARCH_C_API void pomai_search_set_log_callback(pomai_search_log_callback_t callback,
                                                      void* user_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif
