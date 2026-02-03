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

constexpr uint32_t kSnapshotVersion = 1;
constexpr char kMagic[] = "POMAI_SNAP";

template <typename T>
bool WriteValue(std::FILE* out, const T& value) {
  return std::fwrite(&value, sizeof(T), 1, out) == 1;
}

bool WriteString(std::FILE* out, std::string_view value) {
  uint32_t size = static_cast<uint32_t>(value.size());
  if (!WriteValue(out, size)) {
    return false;
  }
  return std::fwrite(value.data(), 1, size, out) == size;
}

template <typename T>
bool ReadValue(std::FILE* in, T* value) {
  return std::fread(value, sizeof(T), 1, in) == 1;
}

bool ReadString(std::FILE* in, std::string* value) {
  uint32_t size = 0;
  if (!ReadValue(in, &size)) {
    return false;
  }
  std::string buffer(size, '\0');
  if (size > 0 && std::fread(buffer.data(), 1, size, in) != size) {
    return false;
  }
  *value = std::move(buffer);
  return true;
}

uint64_t SnapshotSeed(const SearchEngineConfig& cfg) {
  uint64_t seed = StableHash64Combine(StableHash64("snapshot"), std::to_string(cfg.contract_version));
  seed = StableHash64Combine(seed, std::to_string(cfg.global_seed));
  seed = StableHash64Combine(seed, std::to_string(cfg.dim));
  return seed;
}

}  // namespace

Status SnapshotWriter::Write(const SearchEngine& engine, std::string_view path) {
  std::vector<SnapshotRecord> records;
  SearchEngineConfig cfg;
  {
    if (!engine.impl_) {
      return Status(StatusCode::kInternal, "engine not initialized");
    }
    cfg = engine.impl_->cfg;
    for (const auto& shard : engine.impl_->shards) {
      shard->ExportRecords(&records);
    }
  }
  std::filesystem::path target(path);
  std::filesystem::path tmp = target;
  tmp += ".tmp";
  std::FILE* out = std::fopen(tmp.c_str(), "wb");
  if (!out) {
    return Status(StatusCode::kInternal, "failed to open snapshot");
  }
  auto close_out = [&]() {
    if (out) {
      std::fclose(out);
      out = nullptr;
    }
  };
  if (std::fwrite(kMagic, 1, sizeof(kMagic), out) != sizeof(kMagic)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  if (!WriteValue(out, kSnapshotVersion)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  uint64_t seed = SnapshotSeed(cfg);
  if (!WriteValue(out, seed)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  if (!WriteValue(out, cfg.dim) || !WriteValue(out, cfg.num_shards) ||
      !WriteValue(out, cfg.topk_default) || !WriteValue(out, cfg.max_points_per_shard) ||
      !WriteValue(out, cfg.enable_avx2) || !WriteValue(out, cfg.query_threads) ||
      !WriteValue(out, cfg.ingest_threads) || !WriteValue(out, cfg.memory_alignment) ||
      !WriteValue(out, cfg.hnsw_m) || !WriteValue(out, cfg.hnsw_ef_construction) ||
      !WriteValue(out, cfg.hnsw_ef_search) || !WriteValue(out, cfg.hnsw_seed) ||
      !WriteValue(out, cfg.global_seed) || !WriteValue(out, cfg.contract_version)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  int similarity = static_cast<int>(cfg.similarity);
  int index_type = static_cast<int>(cfg.index_type);
  if (!WriteValue(out, similarity) || !WriteValue(out, index_type)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  uint32_t record_count = static_cast<uint32_t>(records.size());
  if (!WriteValue(out, record_count)) {
    close_out();
    return Status(StatusCode::kInternal, "failed to write snapshot");
  }
  for (const auto& record : records) {
    if (!WriteString(out, record.key)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    uint32_t dim = static_cast<uint32_t>(record.vector.size());
    if (!WriteValue(out, dim)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    if (dim > 0 && std::fwrite(record.vector.data(), sizeof(float), dim, out) != dim) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    uint32_t meta_count = static_cast<uint32_t>(record.meta.size());
    if (!WriteValue(out, meta_count)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    std::vector<std::string> keys;
    keys.reserve(record.meta.size());
    for (const auto& pair : record.meta) {
      keys.push_back(pair.first);
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
      if (!WriteString(out, key)) {
        close_out();
        return Status(StatusCode::kInternal, "failed to write snapshot record");
      }
      auto it = record.meta.find(key);
      if (it == record.meta.end()) {
        close_out();
        return Status(StatusCode::kInternal, "failed to write snapshot record");
      }
      if (!WriteString(out, it->second)) {
        close_out();
        return Status(StatusCode::kInternal, "failed to write snapshot record");
      }
    }
    uint8_t has_text = record.text.has_value() ? 1 : 0;
    if (!WriteValue(out, has_text)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    if (record.text && !WriteString(out, *record.text)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
    int64_t expiry_ms = 0;
    if (record.expiry) {
      expiry_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      record.expiry->time_since_epoch())
                      .count();
      // Ensure we don't accidentally write 0 for a valid expiry if it happened to be exactly epoch (unlikely but safe)
      if (expiry_ms == 0) expiry_ms = 1;
    }
    if (!WriteValue(out, expiry_ms)) {
      close_out();
      return Status(StatusCode::kInternal, "failed to write snapshot record");
    }
  }
  std::fflush(out);
  int fd = ::fileno(out);
  if (fd >= 0) {
    ::fsync(fd);
  }
  close_out();
  std::error_code ec;
  std::filesystem::rename(tmp, target, ec);
  if (ec) {
    return Status(StatusCode::kInternal, "failed to finalize snapshot");
  }
  return Status::Ok();
}

StatusOr<std::unique_ptr<SearchEngine>> SnapshotReader::Read(std::string_view path) {
  std::FILE* in = std::fopen(std::string(path).c_str(), "rb");
  if (!in) {
    return Status(StatusCode::kNotFound, "snapshot not found");
  }
  auto close_in = [&]() {
    if (in) {
      std::fclose(in);
      in = nullptr;
    }
  };
  char magic[sizeof(kMagic)] = {};
  if (std::fread(magic, 1, sizeof(magic), in) != sizeof(magic) ||
      std::string_view(magic, sizeof(magic)) != std::string_view(kMagic, sizeof(kMagic))) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "invalid snapshot header");
  }
  uint32_t version = 0;
  if (!ReadValue(in, &version) || version != kSnapshotVersion) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "unsupported snapshot version");
  }
  uint64_t seed = 0;
  if (!ReadValue(in, &seed)) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  SearchEngineConfig cfg;
  if (!ReadValue(in, &cfg.dim) || !ReadValue(in, &cfg.num_shards) ||
      !ReadValue(in, &cfg.topk_default) || !ReadValue(in, &cfg.max_points_per_shard) ||
      !ReadValue(in, &cfg.enable_avx2) || !ReadValue(in, &cfg.query_threads) ||
      !ReadValue(in, &cfg.ingest_threads) || !ReadValue(in, &cfg.memory_alignment) ||
      !ReadValue(in, &cfg.hnsw_m) || !ReadValue(in, &cfg.hnsw_ef_construction) ||
      !ReadValue(in, &cfg.hnsw_ef_search) || !ReadValue(in, &cfg.hnsw_seed) ||
      !ReadValue(in, &cfg.global_seed) || !ReadValue(in, &cfg.contract_version)) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  int similarity = 0;
  int index_type = 0;
  if (!ReadValue(in, &similarity) || !ReadValue(in, &index_type)) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  cfg.similarity = static_cast<SearchEngineConfig::Similarity>(similarity);
  cfg.index_type = static_cast<SearchEngineConfig::IndexType>(index_type);
  if (SnapshotSeed(cfg) != seed) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "snapshot seed mismatch");
  }
  uint32_t record_count = 0;
  if (!ReadValue(in, &record_count)) {
    close_in();
    return Status(StatusCode::kInvalidArgument, "invalid snapshot");
  }
  auto engine_or = SearchEngine::Open(cfg);
  if (!engine_or.ok()) {
    close_in();
    return engine_or.status();
  }
  auto engine = std::move(engine_or.value());
  for (uint32_t i = 0; i < record_count; ++i) {
    SnapshotRecord record;
    if (!ReadString(in, &record.key)) {
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
    }
    uint32_t dim = 0;
    if (!ReadValue(in, &dim)) {
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
    }
    record.vector.resize(dim);
    if (dim > 0 &&
        std::fread(record.vector.data(), sizeof(float), dim, in) != dim) {
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
    }
    uint32_t meta_count = 0;
    if (!ReadValue(in, &meta_count)) {
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
    }
    for (uint32_t m = 0; m < meta_count; ++m) {
      std::string key;
      std::string value;
      if (!ReadString(in, &key) || !ReadString(in, &value)) {
        close_in();
        return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
      }
      record.meta.emplace(std::move(key), std::move(value));
    }
    uint8_t has_text = 0;
    if (!ReadValue(in, &has_text)) {
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
    }
    if (has_text) {
      std::string text;
      if (!ReadString(in, &text)) {
        close_in();
        return Status(StatusCode::kInvalidArgument, "invalid snapshot record");
      }
      record.text = std::move(text);
    }
    int64_t expiry_ms = 0;
    if (!ReadValue(in, &expiry_ms)) {
      // For backward compatibility (if version < X), we might need to handle missing expiry.
      // But here we are changing format and assuming consistent versioning/upgrade or clean slate.
      // The task says previous snapshots are incompatible.
      close_in();
      return Status(StatusCode::kInvalidArgument, "invalid snapshot record (missing expiry)");
    }
    if (expiry_ms != 0) {
      record.expiry = std::chrono::system_clock::time_point(std::chrono::milliseconds(expiry_ms));
    }
    VectorView view{record.vector.data(), static_cast<int>(record.vector.size())};
    size_t shard_index = StableHash64(record.key) % engine->impl_->shards.size();
    Status status = engine->impl_->shards[shard_index]->Upsert(
        record.key, view, std::move(record.meta), record.expiry, record.text);
    if (!status.ok()) {
      close_in();
      return status;
    }
  }
  close_in();
  return StatusOr<std::unique_ptr<SearchEngine>>(std::move(engine));
}

}  // namespace pomai_search
