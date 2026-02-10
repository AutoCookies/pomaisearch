#include "pomai_search/c_api.h"

#include <stdio.h>
#include <string.h>

static void json_escape(const char* input, size_t len, FILE* out) {
  for (size_t i = 0; i < len; ++i) {
    unsigned char c = (unsigned char)input[i];
    switch (c) {
      case '\"':
        fputs("\\\"", out);
        break;
      case '\\':
        fputs("\\\\", out);
        break;
      case '\n':
        fputs("\\n", out);
        break;
      case '\r':
        fputs("\\r", out);
        break;
      case '\t':
        fputs("\\t", out);
        break;
      default:
        fputc(c, out);
        break;
    }
  }
}

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

  float vec1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  pomai_search_metadata_kv_t meta1[] = {{"tag", "alpha"}};
  pomai_search_document_t doc;
  pomai_search_document_init(&doc);
  doc.doc_id = "doc1";
  doc.doc_id_len = strlen(doc.doc_id);
  doc.vector = vec1;
  doc.dim = 4;
  doc.metadata = meta1;
  doc.metadata_count = 1;
  doc.text = "hello world";
  doc.text_len = strlen(doc.text);
  check(pomai_search_engine_upsert(engine, &doc), "engine_upsert");

  pomai_search_iter_t* iter = NULL;
  check(pomai_search_engine_iter_begin(engine, NULL, &iter), "iter_begin");

  pomai_search_doc_view_t view;
  pomai_search_doc_view_init(&view);
  uint8_t has_value = 0;
  do {
    check(pomai_search_iter_next(iter, &view, &has_value), "iter_next");
    if (!has_value) {
      break;
    }
    fputs("{\"doc_id\":\"", stdout);
    json_escape(view.doc_id, view.doc_id_len, stdout);
    fputs("\",\"dim\":", stdout);
    fprintf(stdout, "%u", view.dim);
    fputs(",\"metadata\":{", stdout);
    for (size_t i = 0; i < view.metadata_count; ++i) {
      if (i > 0) {
        fputc(',', stdout);
      }
      fputc('\"', stdout);
      json_escape(view.metadata[i].key, strlen(view.metadata[i].key), stdout);
      fputs("\":\"", stdout);
      json_escape(view.metadata[i].value, strlen(view.metadata[i].value), stdout);
      fputc('\"', stdout);
    }
    fputs("}", stdout);
    if (view.text) {
      fputs(",\"text\":\"", stdout);
      json_escape(view.text, view.text_len, stdout);
      fputc('\"', stdout);
    }
    fputs("}\n", stdout);
  } while (has_value);

  pomai_search_iter_close(iter);
  check(pomai_search_engine_close(engine), "engine_close");
  return 0;
}
