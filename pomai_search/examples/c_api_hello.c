#include "pomai_search/c_api.h"

#include <stdio.h>
#include <string.h>

static void check(pomai_search_status_code_t code, const char* context) {
  if (code != POMAI_SEARCH_STATUS_OK) {
    fprintf(stderr, "%s failed: %d (%s)\n", context, code,
            pomai_search_last_error_message());
  }
}

int main(void) {
  pomai_search_engine_config_t cfg;
  pomai_search_engine_config_init(&cfg);
  cfg.dim = 4;
  cfg.num_shards = 1;

  pomai_search_engine_t* engine = NULL;
  check(pomai_search_engine_open(&cfg, &engine), "engine_open");

  float vec[4] = {1.0f, 0.5f, 0.0f, 0.0f};
  pomai_search_document_t doc;
  pomai_search_document_init(&doc);
  doc.doc_id = "doc1";
  doc.doc_id_len = strlen(doc.doc_id);
  doc.vector = vec;
  doc.dim = 4;
  check(pomai_search_engine_upsert(engine, &doc), "engine_upsert");

  pomai_search_results_t* results = NULL;
  check(pomai_search_engine_search(engine, vec, 4, NULL, &results), "engine_search");

  uint32_t count = 0;
  check(pomai_search_results_count(results, &count), "results_count");
  for (uint32_t i = 0; i < count; ++i) {
    pomai_search_hit_t hit;
    pomai_search_hit_init(&hit);
    check(pomai_search_results_get(results, i, &hit), "results_get");
    printf("hit: %.*s score=%.4f\n", (int)hit.doc_id_len, hit.doc_id, hit.score);
  }
  pomai_search_results_free(results);

  check(pomai_search_engine_close(engine), "engine_close");
  return 0;
}
