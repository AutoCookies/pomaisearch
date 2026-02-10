#include "pomai_search/c_api.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  pomai_search_engine_t* engine;
  const float* query;
  uint32_t dim;
} search_thread_args_t;

static void expect_ok(pomai_search_status_code_t code, const char* context) {
  if (code != POMAI_SEARCH_STATUS_OK) {
    fprintf(stderr, "%s failed: %d (%s)\n", context, code,
            pomai_search_last_error_message());
  }
  assert(code == POMAI_SEARCH_STATUS_OK);
}

static void* search_thread(void* arg) {
  search_thread_args_t* args = (search_thread_args_t*)arg;
  for (int i = 0; i < 20; ++i) {
    pomai_search_results_t* results = NULL;
    pomai_search_status_code_t code =
        pomai_search_engine_search(args->engine, args->query, args->dim, NULL, &results);
    expect_ok(code, "search_thread");
    pomai_search_results_free(results);
  }
  return NULL;
}

static void test_open_close(void) {
  pomai_search_engine_config_t cfg;
  pomai_search_engine_config_init(&cfg);
  cfg.dim = 4;
  cfg.num_shards = 1;

  pomai_search_engine_t* engine = NULL;
  expect_ok(pomai_search_engine_open(&cfg, &engine), "engine_open");
  expect_ok(pomai_search_engine_close(engine), "engine_close");
}

static pomai_search_engine_t* build_engine(void) {
  pomai_search_engine_config_t cfg;
  pomai_search_engine_config_init(&cfg);
  cfg.dim = 4;
  cfg.num_shards = 1;
  pomai_search_engine_t* engine = NULL;
  expect_ok(pomai_search_engine_open(&cfg, &engine), "engine_open");
  return engine;
}

static void upsert_doc(pomai_search_engine_t* engine, const char* id, const float* vec,
                       const char* tag, const char* text) {
  pomai_search_metadata_kv_t meta[1];
  meta[0].key = "tag";
  meta[0].value = tag;
  pomai_search_document_t doc;
  pomai_search_document_init(&doc);
  doc.doc_id = id;
  doc.doc_id_len = strlen(id);
  doc.vector = vec;
  doc.dim = 4;
  doc.metadata = meta;
  doc.metadata_count = 1;
  doc.text = text;
  doc.text_len = text ? strlen(text) : 0;
  expect_ok(pomai_search_engine_upsert(engine, &doc), "engine_upsert");
}

static void test_upsert_search_delete(void) {
  pomai_search_engine_t* engine = build_engine();
  float vec1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  upsert_doc(engine, "doc1", vec1, "alpha", "hello world");
  upsert_doc(engine, "doc2", vec2, "beta", "quick brown fox");

  pomai_search_results_t* results = NULL;
  expect_ok(pomai_search_engine_search(engine, vec1, 4, NULL, &results), "engine_search");

  uint32_t count = 0;
  expect_ok(pomai_search_results_count(results, &count), "results_count");
  assert(count > 0);
  pomai_search_hit_t hit;
  pomai_search_hit_init(&hit);
  expect_ok(pomai_search_results_get(results, 0, &hit), "results_get");
  assert(hit.doc_id != NULL);
  assert(strncmp(hit.doc_id, "doc1", hit.doc_id_len) == 0);
  pomai_search_results_free(results);

  expect_ok(pomai_search_engine_delete(engine, "doc1", strlen("doc1")), "engine_delete");
  results = NULL;
  pomai_search_status_code_t code =
      pomai_search_engine_search_by_key(engine, "doc1", strlen("doc1"), NULL, &results);
  if (code != POMAI_SEARCH_STATUS_NOT_FOUND) {
    fprintf(stderr, "expected not found, got %d (%s)\n", code,
            pomai_search_last_error_message());
    assert(code == POMAI_SEARCH_STATUS_NOT_FOUND);
  }

  expect_ok(pomai_search_engine_close(engine), "engine_close");
}

static void test_iteration_and_save_load(void) {
  pomai_search_engine_t* engine = build_engine();
  float vec1[4] = {1.0f, 0.1f, 0.0f, 0.0f};
  float vec2[4] = {0.1f, 1.0f, 0.0f, 0.0f};
  upsert_doc(engine, "doc1", vec1, "alpha", "hello world");
  upsert_doc(engine, "doc2", vec2, "beta", "quick brown fox");

  pomai_search_iter_t* iter = NULL;
  expect_ok(pomai_search_engine_iter_begin(engine, NULL, &iter), "iter_begin");
  uint8_t has_value = 0;
  uint32_t seen = 0;
  pomai_search_doc_view_t view;
  pomai_search_doc_view_init(&view);
  do {
    expect_ok(pomai_search_iter_next(iter, &view, &has_value), "iter_next");
    if (has_value) {
      ++seen;
      assert(view.doc_id != NULL);
      assert(view.vector != NULL);
      assert(view.dim == 4);
    }
  } while (has_value);
  assert(seen == 2);
  pomai_search_iter_close(iter);

  const char* snapshot_path = "c_api_test.snapshot";
  expect_ok(pomai_search_engine_save(engine, snapshot_path, strlen(snapshot_path)), "save");
  expect_ok(pomai_search_engine_close(engine), "engine_close");

  pomai_search_engine_t* loaded = NULL;
  expect_ok(pomai_search_engine_load(snapshot_path, strlen(snapshot_path), &loaded), "load");
  pomai_search_results_t* results = NULL;
  expect_ok(pomai_search_engine_search(loaded, vec2, 4, NULL, &results), "search_loaded");
  pomai_search_results_free(results);
  expect_ok(pomai_search_engine_close(loaded), "engine_close_loaded");
}

static void test_hybrid_and_concurrency(void) {
  pomai_search_engine_t* engine = build_engine();
  float vec1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  upsert_doc(engine, "doc1", vec1, "alpha", "hello world");
  upsert_doc(engine, "doc2", vec2, "beta", "quick brown fox");

  pomai_search_hybrid_query_t hybrid;
  pomai_search_hybrid_query_init(&hybrid);
  hybrid.text = "hello";
  hybrid.text_len = strlen("hello");
  hybrid.vector = vec1;
  hybrid.dim = 4;
  hybrid.alpha = 0.7f;
  pomai_search_results_t* results = NULL;
  expect_ok(pomai_search_engine_hybrid_search(engine, &hybrid, NULL, &results), "hybrid_search");
  pomai_search_results_free(results);

  pthread_t threads[4];
  search_thread_args_t args;
  args.engine = engine;
  args.query = vec1;
  args.dim = 4;
  for (int i = 0; i < 4; ++i) {
    int rc = pthread_create(&threads[i], NULL, search_thread, &args);
    if (rc != 0) {
      fprintf(stderr, "pthread_create failed: %d\n", rc);
      assert(rc == 0);
    }
  }
  for (int i = 0; i < 4; ++i) {
    int rc = pthread_join(threads[i], NULL);
    if (rc != 0) {
      fprintf(stderr, "pthread_join failed: %d\n", rc);
      assert(rc == 0);
    }
  }

  expect_ok(pomai_search_engine_close(engine), "engine_close");
}

static void test_invalid_inputs(void) {
  pomai_search_status_code_t code =
      pomai_search_engine_open(NULL, NULL);
  if (code != POMAI_SEARCH_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "expected invalid argument, got %d (%s)\n", code,
            pomai_search_last_error_message());
    assert(code == POMAI_SEARCH_STATUS_INVALID_ARGUMENT);
  }
}

int main(void) {
  test_open_close();
  test_upsert_search_delete();
  test_iteration_and_save_load();
  test_hybrid_and_concurrency();
  test_invalid_inputs();
  printf("C API tests passed.\n");
  return 0;
}
