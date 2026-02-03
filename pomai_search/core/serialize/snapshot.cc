#include "core/serialize/snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

#include "pomai_search/hash.h"
#include "pomai_search/search_engine.h"
#include "src/search_engine_impl.h"

namespace pomai_search {
namespace {

constexpr uint32_t kSnapshotVersion = 2; // V2: Native Binary
constexpr char kMagic[] = "POMAI_SNAP";

template <typename T>
bool WriteValue(std::FILE* out, const T& value) {
  return std::fwrite(&value, sizeof(T), 1, out) == 1;
}

template <typename T>
bool ReadValue(std::FILE* in, T* value) {
  return std::fread(value, sizeof(T), 1, in) == 1;
}

uint64_t SnapshotSeed(const SearchEngineConfig& cfg) {
  uint64_t seed = StableHash64Combine(StableHash64("snapshot"), std::to_string(cfg.contract_version));
  seed = StableHash64Combine(seed, std::to_string(cfg.global_seed));
  seed = StableHash64Combine(seed, std::to_string(cfg.dim));
  return seed;
}

}  // namespace

Status SnapshotWriter::Write(const SearchEngine& engine, std::string_view path) {
  if (!engine.impl_) return Status(StatusCode::kInternal, "engine not initialized");
  
  std::filesystem::path target(path);
  std::filesystem::path tmp = target;
  tmp += ".tmp";
  std::FILE* out = std::fopen(tmp.c_str(), "wb");
  if (!out) return Status(StatusCode::kInternal, "failed to open snapshot");

  auto close_out = [&]() { if (out) { std::fclose(out); out = nullptr; } };
  
  // Header
  if (std::fwrite(kMagic, 1, sizeof(kMagic), out) != sizeof(kMagic)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  if (!WriteValue(out, kSnapshotVersion)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  
  // Config
  const auto& cfg = engine.impl_->cfg;
  uint64_t seed = SnapshotSeed(cfg);
  if (!WriteValue(out, seed)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  
  if (!WriteValue(out, cfg.dim) || !WriteValue(out, cfg.num_shards) ||
      !WriteValue(out, cfg.topk_default) || !WriteValue(out, cfg.max_points_per_shard) ||
      !WriteValue(out, cfg.enable_avx2) || !WriteValue(out, cfg.query_threads) ||
      !WriteValue(out, cfg.ingest_threads) || !WriteValue(out, cfg.memory_alignment) ||
      !WriteValue(out, cfg.hnsw_m) || !WriteValue(out, cfg.hnsw_ef_construction) ||
      !WriteValue(out, cfg.hnsw_ef_search) || !WriteValue(out, cfg.hnsw_seed) ||
      !WriteValue(out, cfg.global_seed) || !WriteValue(out, cfg.contract_version)) {
      close_out(); return Status(StatusCode::kInternal, "write failed");
  }
  
  int similarity = static_cast<int>(cfg.similarity);
  int index_type = static_cast<int>(cfg.index_type);
  if (!WriteValue(out, similarity) || !WriteValue(out, index_type)) {
      close_out(); return Status(StatusCode::kInternal, "write failed");
  }
  
  // Shards
  for (const auto& shard : engine.impl_->shards) {
      Status s = shard->Save(out);
      if (!s.ok()) { close_out(); return s; }
  }
  
  std::fflush(out);
  int fd = ::fileno(out);
  if (fd >= 0) ::fsync(fd);
  close_out();
  
  std::error_code ec;
  std::filesystem::rename(tmp, target, ec);
  if (ec) return Status(StatusCode::kInternal, "failed to finalize snapshot");
  
  return Status::Ok();
}

StatusOr<std::unique_ptr<SearchEngine>> SnapshotReader::Read(std::string_view path) {
  std::FILE* in = std::fopen(std::string(path).c_str(), "rb");
  if (!in) return Status(StatusCode::kNotFound, "snapshot not found");
  
  auto close_in = [&]() { if (in) { std::fclose(in); in = nullptr; } };
  
  char magic[sizeof(kMagic)] = {};
  if (std::fread(magic, 1, sizeof(magic), in) != sizeof(magic) ||
      std::string_view(magic, sizeof(magic)) != std::string_view(kMagic, sizeof(kMagic))) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot header");
  }
  
  uint32_t version = 0;
  if (!ReadValue(in, &version) || version != kSnapshotVersion) {
      close_in(); return Status(StatusCode::kInvalidArgument, "unsupported snapshot version");
  }
  
  uint64_t seed = 0;
  if (!ReadValue(in, &seed)) { close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot"); }
  
  SearchEngineConfig cfg;
  if (!ReadValue(in, &cfg.dim) || !ReadValue(in, &cfg.num_shards) ||
      !ReadValue(in, &cfg.topk_default) || !ReadValue(in, &cfg.max_points_per_shard) ||
      !ReadValue(in, &cfg.enable_avx2) || !ReadValue(in, &cfg.query_threads) ||
      !ReadValue(in, &cfg.ingest_threads) || !ReadValue(in, &cfg.memory_alignment) ||
      !ReadValue(in, &cfg.hnsw_m) || !ReadValue(in, &cfg.hnsw_ef_construction) ||
      !ReadValue(in, &cfg.hnsw_ef_search) || !ReadValue(in, &cfg.hnsw_seed) ||
      !ReadValue(in, &cfg.global_seed) || !ReadValue(in, &cfg.contract_version)) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  
  int similarity = 0;
  int index_type = 0;
  if (!ReadValue(in, &similarity) || !ReadValue(in, &index_type)) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  cfg.similarity = static_cast<SearchEngineConfig::Similarity>(similarity);
  cfg.index_type = static_cast<SearchEngineConfig::IndexType>(index_type);
  
  if (SnapshotSeed(cfg) != seed) { close_in(); return Status(StatusCode::kInvalidArgument, "snapshot seed mismatch"); }
  
  auto engine_or = SearchEngine::Open(cfg);
  if (!engine_or.ok()) { close_in(); return engine_or.status(); }
  auto engine = std::move(engine_or.value());
  
  // Load Shards
  for (const auto& shard : engine->impl_->shards) {
      Status s = shard->Load(in);
      if (!s.ok()) { close_in(); return s; }
  }
  
  close_in();
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

}  // namespace pomai_search
