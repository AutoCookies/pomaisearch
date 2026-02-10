#include "core/serialize/snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "pomai_search/hash.h"
#include "pomai_search/logging.h"
#include "pomai_search/search_engine.h"
#include "src/search_engine_impl.h"

namespace pomai_search {
namespace {

constexpr uint32_t kSnapshotVersion = 3; // V3: CRC32 + payload size
constexpr char kMagic[] = "POMAI_SNAP";

constexpr size_t kHeaderSize = sizeof(kMagic) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);

uint32_t Crc32Update(uint32_t crc, const unsigned char* data, size_t len) {
  static uint32_t table[256];
  static bool initialized = false;
  if (!initialized) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int j = 0; j < 8; ++j) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    initialized = true;
  }
  crc = crc ^ 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

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
  POMAI_LOG_INFO(std::string("{\"event\":\"snapshot_write_start\",\"path\":\"") +
                 std::string(path) + "\"}");
  
  std::filesystem::path target(path);
  std::filesystem::path tmp = target;
  tmp += ".tmp";
  std::FILE* out = std::fopen(tmp.c_str(), "w+b");
  if (!out) return Status(StatusCode::kInternal, "failed to open snapshot");

  auto close_out = [&]() { if (out) { std::fclose(out); out = nullptr; } };
  
  // Header
  if (std::fwrite(kMagic, 1, sizeof(kMagic), out) != sizeof(kMagic)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  if (!WriteValue(out, kSnapshotVersion)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  uint64_t payload_size = 0;
  uint32_t payload_crc = 0;
  if (!WriteValue(out, payload_size)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  if (!WriteValue(out, payload_crc)) { close_out(); return Status(StatusCode::kInternal, "write failed"); }
  
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

  long end_pos = std::ftell(out);
  if (end_pos < 0) { close_out(); return Status(StatusCode::kInternal, "snapshot size failed"); }
  if (static_cast<size_t>(end_pos) < kHeaderSize) {
    close_out();
    return Status(StatusCode::kInternal, "snapshot too small");
  }
  payload_size = static_cast<uint64_t>(end_pos - static_cast<long>(kHeaderSize));

  if (std::fseek(out, static_cast<long>(kHeaderSize), SEEK_SET) != 0) {
    close_out();
    return Status(StatusCode::kInternal, "snapshot seek failed");
  }
  std::vector<unsigned char> buffer(64 * 1024);
  uint64_t remaining = payload_size;
  payload_crc = 0;
  while (remaining > 0) {
    size_t to_read = static_cast<size_t>(std::min<uint64_t>(buffer.size(), remaining));
    size_t read = std::fread(buffer.data(), 1, to_read, out);
    if (read != to_read) {
      close_out();
      return Status(StatusCode::kInternal, "snapshot checksum read failed");
    }
    payload_crc = Crc32Update(payload_crc, buffer.data(), read);
    remaining -= read;
  }
  if (std::fseek(out, static_cast<long>(sizeof(kMagic) + sizeof(uint32_t)), SEEK_SET) != 0) {
    close_out();
    return Status(StatusCode::kInternal, "snapshot seek failed");
  }
  if (!WriteValue(out, payload_size) || !WriteValue(out, payload_crc)) {
    close_out();
    return Status(StatusCode::kInternal, "snapshot header write failed");
  }
  std::fflush(out);
  if (fd >= 0) ::fsync(fd);
  close_out();
  
  std::error_code ec;
  std::filesystem::rename(tmp, target, ec);
  if (ec) return Status(StatusCode::kInternal, "failed to finalize snapshot");
  std::filesystem::path dir_path = target.parent_path();
  if (dir_path.empty()) {
    dir_path = ".";
  }
  int dir_fd = ::open(dir_path.c_str(), O_RDONLY | O_DIRECTORY);
  if (dir_fd >= 0) {
    ::fsync(dir_fd);
    ::close(dir_fd);
  }
  POMAI_LOG_INFO(std::string("{\"event\":\"snapshot_write_complete\",\"path\":\"") +
                 std::string(path) + "\",\"bytes\":" + std::to_string(payload_size) +
                 ",\"crc32\":" + std::to_string(payload_crc) + "}");
  return Status::Ok();
}

StatusOr<std::unique_ptr<SearchEngine>> SnapshotReader::Read(std::string_view path) {
  POMAI_LOG_INFO(std::string("{\"event\":\"snapshot_read_start\",\"path\":\"") +
                 std::string(path) + "\"}");
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

  uint64_t payload_size = 0;
  uint32_t payload_crc = 0;
  if (!ReadValue(in, &payload_size) || !ReadValue(in, &payload_crc)) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot header");
  }

  if (std::fseek(in, 0, SEEK_END) != 0) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot size");
  }
  long end_pos = std::ftell(in);
  if (end_pos < 0 || static_cast<uint64_t>(end_pos) < kHeaderSize) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot size");
  }
  uint64_t actual_payload = static_cast<uint64_t>(end_pos - static_cast<long>(kHeaderSize));
  if (actual_payload != payload_size) {
      close_in(); return Status(StatusCode::kInvalidArgument, "snapshot size mismatch");
  }
  if (std::fseek(in, static_cast<long>(kHeaderSize), SEEK_SET) != 0) {
      close_in(); return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  std::vector<unsigned char> buffer(64 * 1024);
  uint64_t remaining = payload_size;
  uint32_t computed_crc = 0;
  while (remaining > 0) {
      size_t to_read = static_cast<size_t>(std::min<uint64_t>(buffer.size(), remaining));
      size_t read = std::fread(buffer.data(), 1, to_read, in);
      if (read != to_read) {
          close_in(); return Status(StatusCode::kInvalidArgument, "snapshot checksum mismatch");
      }
      computed_crc = Crc32Update(computed_crc, buffer.data(), read);
      remaining -= read;
  }
  if (computed_crc != payload_crc) {
      close_in(); return Status(StatusCode::kInvalidArgument, "snapshot checksum mismatch");
  }
  if (std::fseek(in, static_cast<long>(kHeaderSize), SEEK_SET) != 0) {
      close_in(); return Status(StatusCode::kInvalidArgument, "snapshot seek failed");
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
  POMAI_LOG_INFO(std::string("{\"event\":\"snapshot_read_complete\",\"path\":\"") +
                 std::string(path) + "\",\"bytes\":" + std::to_string(payload_size) +
                 ",\"crc32\":" + std::to_string(payload_crc) + "}");
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

}  // namespace pomai_search
