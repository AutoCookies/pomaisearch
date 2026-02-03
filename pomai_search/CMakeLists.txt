cmake_minimum_required(VERSION 3.16)
project(pomai_search LANGUAGES CXX)

enable_testing()

option(POMAI_SEARCH_ENABLE_AVX2 "Enable AVX2 kernels" ON)
option(POMAI_SEARCH_ENABLE_ASAN "Enable ASAN" OFF)
option(POMAI_SEARCH_ENABLE_TSAN "Enable TSAN" OFF)
option(POMAI_SEARCH_ENABLE_UBSAN "Enable UBSAN" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(pomai_search_core
  src/status.cc
  src/logging.cc
  src/thread_pool.cc
  src/vector_arena.cc
  src/simd/scalar_kernels.cc
  src/simd/avx2_kernels.cc
  src/simd/cpu_features.cc
  src/index/flat_index.cc
  src/index/hnsw_index.cc
  src/search_engine.cc
  src/net/http_server.cc
  src/net/json.cc
  src/observability/metrics.cc
)

target_include_directories(pomai_search_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(pomai_search_core PRIVATE $<$<BOOL:${POMAI_SEARCH_ENABLE_AVX2}>:POMAI_SEARCH_ENABLE_AVX2=1>)

target_compile_options(pomai_search_core PRIVATE -O3 -Wall -Wextra -Wpedantic -Werror)

set_source_files_properties(src/simd/avx2_kernels.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

add_executable(pomai-searchd
  src/server/search_server.cc
)

target_link_libraries(pomai-searchd PRIVATE pomai_search_core)

target_compile_options(pomai-searchd PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(pomai-search
  src/cli/search_cli.cc
)

target_link_libraries(pomai-search PRIVATE pomai_search_core)

target_compile_options(pomai-search PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(pomai_search_tests
  tests/test_main.cc
  tests/test_status.cc
  tests/test_thread_pool.cc
  tests/test_vector_arena.cc
  tests/test_simd.cc
  tests/test_shard_or_index_flat.cc
  tests/test_engine_flat.cc
  tests/test_engine_hnsw.cc
  tests/test_hybrid.cc
  tests/test_hash.cc
  tests/test_determinism.cc
  tests/test_concurrency.cc
)

target_link_libraries(pomai_search_tests PRIVATE pomai_search_core)

target_include_directories(pomai_search_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(pomai_search_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_test(NAME pomai_search_tests COMMAND pomai_search_tests)

add_executable(pomai_search_bench_flat
  bench/bench_flat.cc
)

target_link_libraries(pomai_search_bench_flat PRIVATE pomai_search_core)

target_compile_options(pomai_search_bench_flat PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(pomai_search_bench_hnsw
  bench/bench_hnsw.cc
)

target_link_libraries(pomai_search_bench_hnsw PRIVATE pomai_search_core)

target_compile_options(pomai_search_bench_hnsw PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(pomai_search_bench_recall
  bench/bench_recall.cc
)

target_link_libraries(pomai_search_bench_recall PRIVATE pomai_search_core)

target_compile_options(pomai_search_bench_recall PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(POMAI_SEARCH_ENABLE_ASAN)
  target_compile_options(pomai_search_core PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(pomai_search_core PRIVATE -fsanitize=address)
endif()

if(POMAI_SEARCH_ENABLE_TSAN)
  target_compile_options(pomai_search_core PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(pomai_search_core PRIVATE -fsanitize=thread)
endif()

if(POMAI_SEARCH_ENABLE_UBSAN)
  target_compile_options(pomai_search_core PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
  target_link_options(pomai_search_core PRIVATE -fsanitize=undefined)
endif()
