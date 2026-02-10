# Pomai Search C ABI

This document describes the stable C ABI wrapper for Pomai Search Engine. The C ABI is designed to
be safe and easy to bind from Rust, Go, Python, Node, and other languages without exposing C++
types.

## Versioning & ABI Stability

* ABI version: `POMAI_SEARCH_C_ABI_VERSION` (changes only on breaking ABI updates).
* API version: `POMAI_SEARCH_C_API_VERSION_{MAJOR,MINOR,PATCH}` for non-breaking feature changes.
* Every public struct begins with `struct_size` to support forward-compatible extension.

Call `pomai_search_c_version()` to retrieve the current versions and git SHA.

## Ownership & Lifetime Rules

* The caller **must not** free any memory returned by the C API directly.
* Any allocation crossing the boundary has a matching free function:
  * `pomai_search_results_free()` for search results.
  * `pomai_search_iter_close()` for iterators.
  * `pomai_search_engine_close()` for engines.
* `pomai_search_hit_t` and `pomai_search_doc_view_t` contain pointers that are valid **only**
  while the owning results/iterator object is alive and **until the next call** that mutates the
  same object (e.g. the next `pomai_search_iter_next()`).

## Error Model

* All functions return `pomai_search_status_code_t`.
* Extended details are available via:
  * `pomai_search_last_error_code()`
  * `pomai_search_last_error_message()` (thread-local, valid until the next C API call on the same
    thread).

## Thread Safety

* `pomai_search_engine_t` operations are thread-safe for concurrent searches and upserts.
* `pomai_search_results_t` is immutable once created and can be read concurrently.
* `pomai_search_iter_t` is **not** thread-safe; use it from a single thread at a time.
* Error retrieval is thread-local, so concurrent calls do not race on error state.
* `pomai_search_document_t.ttl_ms` uses milliseconds; set to `-1` for no expiration.
* `pomai_search_engine_iter_begin()` materializes a snapshot of records for consistent iteration,
  which may increase memory usage on large datasets.

## Logging

Use `pomai_search_set_log_callback()` to intercept library logs. When unset, Pomai Search logs to
stderr.

## Build & Link

```bash
cmake -S . -B build -DPOMAI_SEARCH_BUILD_C_API=ON
cmake --build build
```

Link against `pomai_search_c` and include `pomai_search/c_api.h`.

### Minimal CMake snippet

```cmake
find_package(pomai_search REQUIRED)
target_link_libraries(my_app PRIVATE pomai_search_c)
target_include_directories(my_app PRIVATE /path/to/pomai_search/include)
```

## Examples

* `examples/c_api_hello.c` — minimal vector insert/search.
* `examples/c_api_iter_export.c` — iterate and export JSONL.

## Notes

* The C ABI does not change search semantics; it forwards to the C++ SearchEngine.
* Snapshot save/load is supported via `pomai_search_engine_save()` and
  `pomai_search_engine_load()` using the native snapshot format.
