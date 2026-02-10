#include "core/index/ivf_sq8_index.h"

#include <algorithm>
#include <chrono>
#include <queue>
#include <unordered_set>

namespace pomai_search {

IvfSq8Index::IvfSq8Index(const VectorStore* store, int dim, Config config)
    : store_(store),
      dim_(dim),
      config_(config),
      kmeans_({config.nlist, 20, 39, 42, config.similarity}, dim),
      quantizer_(dim) {
  if (config_.min_train_size == 0) {
    config_.min_train_size = static_cast<size_t>(config_.nlist * 39);
    if (config_.min_train_size < 100) config_.min_train_size = 100;
  }
}

Status IvfSq8Index::Upsert(uint32_t id, size_t offset, float /*norm*/) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto& doc = docs_[id];
  doc.offset = offset;
  doc.deleted = false;
  ++doc.gen;

  if (trained_) {
    auto store_guard = store_->AcquireRead();
    const float* vec = store_->Get(offset, store_guard);
    Status s = AddToInvertedList(id, doc.gen, vec);
    if (!s.ok()) {
      return s;
    }
    int centroid = kmeans_.AssignOne(vec);
    if (centroid >= 0 && centroid < static_cast<int>(list_metrics_.size()) &&
        ListNeedsCompaction(static_cast<size_t>(centroid))) {
      CompactList(static_cast<size_t>(centroid));
    }
  } else {
    buffer_ids_.push_back(id);
    if (buffer_ids_.size() >= config_.min_train_size) {
      return TrainImpl(lock);
    }
  }
  return Status::Ok();
}

Status IvfSq8Index::Delete(uint32_t id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = docs_.find(id);
  if (it == docs_.end()) return Status(StatusCode::kNotFound, "not found");
  it->second.deleted = true;
  ++it->second.gen;
  return Status::Ok();
}

Status IvfSq8Index::Train() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (trained_) return Status::Ok();
  return TrainImpl(lock);
}

Status IvfSq8Index::TrainImpl(const std::unique_lock<std::shared_mutex>& /*lock*/) {
  std::vector<uint32_t> training_offsets;
  training_offsets.reserve(buffer_ids_.size());
  for (uint32_t id : buffer_ids_) {
    auto it = docs_.find(id);
    if (it != docs_.end() && !it->second.deleted) {
      training_offsets.push_back(static_cast<uint32_t>(it->second.offset));
    }
  }

  if (training_offsets.empty()) return Status::Ok();

  Status s = kmeans_.Train(*store_, training_offsets);
  if (!s.ok()) return s;
  quantizer_.Train(*store_, training_offsets);

  lists_.assign(config_.nlist, {});
  codes_.assign(config_.nlist, {});
  list_metrics_.assign(config_.nlist, {});

  auto store_guard = store_->AcquireRead();
  for (uint32_t id : buffer_ids_) {
    auto it = docs_.find(id);
    if (it != docs_.end() && !it->second.deleted) {
      const float* vec = store_->Get(it->second.offset, store_guard);
      AddToInvertedList(id, it->second.gen, vec);
    }
  }
  buffer_ids_.clear();
  trained_ = true;
  return Status::Ok();
}

Status IvfSq8Index::AddToInvertedList(uint32_t id, uint32_t gen, const float* vec) {
  int centroid = kmeans_.AssignOne(vec);
  if (centroid < 0 || centroid >= static_cast<int>(lists_.size())) {
    return Status(StatusCode::kInternal, "KMeans error");
  }

  lists_[centroid].push_back(PostingEntry{id, gen});
  size_t old_size = codes_[centroid].size();
  codes_[centroid].resize(old_size + static_cast<size_t>(dim_));
  quantizer_.Encode(vec, codes_[centroid].data() + old_size);

  return Status::Ok();
}

bool IvfSq8Index::EntryIsLive(const PostingEntry& entry) const {
  auto it = docs_.find(entry.id);
  if (it == docs_.end()) return false;
  return !it->second.deleted && it->second.gen == entry.gen;
}

bool IvfSq8Index::ListNeedsCompaction(size_t list_idx) const {
  if (list_idx >= lists_.size()) {
    return false;
  }
  const auto& list = lists_[list_idx];
  const auto& metric = list_metrics_[list_idx];
  if (list.empty()) {
    return false;
  }
  float stale_ratio = metric.stale_checked > 0
                          ? static_cast<float>(metric.stale_entries) /
                                static_cast<float>(metric.stale_checked)
                          : 0.0f;
  size_t bytes = list.size() * sizeof(PostingEntry) + codes_[list_idx].size();
  return stale_ratio > config_.stale_ratio_threshold || list.size() > config_.max_list_size ||
         bytes > config_.max_list_bytes;
}

void IvfSq8Index::CompactList(size_t list_idx) {
  if (list_idx >= lists_.size()) {
    return;
  }
  auto start = std::chrono::steady_clock::now();
  auto& list = lists_[list_idx];
  auto& code_blob = codes_[list_idx];
  std::vector<PostingEntry> compacted;
  std::vector<uint8_t> compacted_codes;
  compacted.reserve(list.size());
  compacted_codes.reserve(code_blob.size());

  for (size_t i = 0; i < list.size(); ++i) {
    if (!EntryIsLive(list[i])) {
      continue;
    }
    compacted.push_back(list[i]);
    size_t code_start = i * static_cast<size_t>(dim_);
    compacted_codes.insert(compacted_codes.end(), code_blob.begin() + code_start,
                           code_blob.begin() + code_start + static_cast<size_t>(dim_));
  }

  list.swap(compacted);
  code_blob.swap(compacted_codes);
  list_metrics_[list_idx] = {};
  ++compaction_runs_;
  compaction_time_ms_ +=
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count());
}

void IvfSq8Index::Compact() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  for (size_t i = 0; i < lists_.size(); ++i) {
    if (ListNeedsCompaction(i)) {
      CompactList(i);
    }
  }
}

StatusOr<std::vector<Candidate>> IvfSq8Index::Search(VectorView q, int topk,
                                                     const Filter& /*f*/) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto store_guard = store_->AcquireRead();

  if (!trained_) {
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                        std::greater<>> pq;
    DotFunc dot = GetDotFunc(CpuSupportsAvx2());
    for (uint32_t id : buffer_ids_) {
      auto it = docs_.find(id);
      if (it == docs_.end() || it->second.deleted) continue;
      const float* vec = store_->Get(it->second.offset, store_guard);
      float score = dot(q.data, vec, dim_);
      if (pq.size() < static_cast<size_t>(topk)) pq.push({score, id});
      else if (score > pq.top().first) {
        pq.pop();
        pq.push({score, id});
      }
    }
    std::vector<Candidate> res;
    while (!pq.empty()) {
      res.push_back({pq.top().second, pq.top().first});
      pq.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
  }

  int nprobe = std::min(config_.nprobe, config_.nlist);
  std::vector<std::pair<float, int>> centroid_scores;
  centroid_scores.reserve(config_.nlist);
  DotFunc dot = GetDotFunc(CpuSupportsAvx2());
  const float* centroids = kmeans_.centroids().data();
  for (int c = 0; c < config_.nlist; ++c) {
    float score = dot(q.data, centroids + c * dim_, dim_);
    centroid_scores.push_back({score, c});
  }
  std::partial_sort(centroid_scores.begin(), centroid_scores.begin() + nprobe,
                    centroid_scores.end(), std::greater<>());

  int refine_k = static_cast<int>(topk * config_.refine_factor);
  std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                      std::greater<>> approx_pq;

  std::vector<float> dists_buffer;
  for (int i = 0; i < nprobe; ++i) {
    size_t c = static_cast<size_t>(centroid_scores[i].second);
    const auto& entries = lists_[c];
    const auto& codes = codes_[c];
    if (entries.empty()) continue;

    dists_buffer.resize(entries.size());
    quantizer_.ComputeDotProducts(q.data, codes.data(), static_cast<int>(entries.size()),
                                  dists_buffer.data());

    auto& metric = const_cast<ListMetrics&>(list_metrics_[c]);
    for (size_t j = 0; j < entries.size(); ++j) {
      ++metric.stale_checked;
      if (!EntryIsLive(entries[j])) {
        ++metric.stale_entries;
        continue;
      }
      float score = dists_buffer[j];
      if (approx_pq.size() < static_cast<size_t>(refine_k)) {
        approx_pq.push({score, entries[j].id});
      } else if (score > approx_pq.top().first) {
        approx_pq.pop();
        approx_pq.push({score, entries[j].id});
      }
    }
  }

  std::vector<uint32_t> refine_ids;
  while (!approx_pq.empty()) {
    refine_ids.push_back(approx_pq.top().second);
    approx_pq.pop();
  }

  std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                      std::greater<>> exact_pq;
  for (uint32_t id : refine_ids) {
    auto it = docs_.find(id);
    if (it == docs_.end() || it->second.deleted) continue;
    const float* vec = store_->Get(it->second.offset, store_guard);
    float score = dot(q.data, vec, dim_);
    if (exact_pq.size() < static_cast<size_t>(topk)) {
      exact_pq.push({score, id});
    } else if (score > exact_pq.top().first) {
      exact_pq.pop();
      exact_pq.push({score, id});
    }
  }

  std::vector<Candidate> result;
  while (!exact_pq.empty()) {
    result.push_back({exact_pq.top().second, exact_pq.top().first});
    exact_pq.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

IndexStats IvfSq8Index::GetStats() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  IndexStats stats;
  for (const auto& pair : docs_) {
    if (!pair.second.deleted) {
      ++stats.num_points;
    }
  }
  size_t total_entries = 0;
  for (const auto& list : lists_) {
    total_entries += list.size();
  }
  stats.num_deleted = total_entries >= stats.num_points ? total_entries - stats.num_points : 0;
  return stats;
}

Status IvfSq8Index::Save(std::FILE* out) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  if (!fwrite(&config_.nlist, sizeof(config_.nlist), 1, out)) return Status(StatusCode::kInternal, "write failed");
  if (!fwrite(&trained_, sizeof(trained_), 1, out)) return Status(StatusCode::kInternal, "write failed");

  uint64_t buf_sz = buffer_ids_.size();
  if (!fwrite(&buf_sz, sizeof(buf_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
  if (buf_sz > 0 && fwrite(buffer_ids_.data(), sizeof(uint32_t), buf_sz, out) != buf_sz) return Status(StatusCode::kInternal, "write failed");

  uint64_t map_sz = docs_.size();
  if (!fwrite(&map_sz, sizeof(map_sz), 1, out)) return Status(StatusCode::kInternal, "write failed");
  for (const auto& pair : docs_) {
    if (!fwrite(&pair.first, sizeof(pair.first), 1, out)) return Status(StatusCode::kInternal, "write failed");
    if (!fwrite(&pair.second, sizeof(DocState), 1, out)) return Status(StatusCode::kInternal, "write failed");
  }

  if (trained_) {
    Status s = kmeans_.Save(out);
    if (!s.ok()) return s;

    s = quantizer_.Save(out);
    if (!s.ok()) return s;

    uint64_t num_lists = lists_.size();
    if (!fwrite(&num_lists, sizeof(num_lists), 1, out)) return Status(StatusCode::kInternal, "write failed");
    for (const auto& lst : lists_) {
      uint64_t lsz = lst.size();
      if (!fwrite(&lsz, sizeof(lsz), 1, out)) return Status(StatusCode::kInternal, "write failed");
      if (lsz > 0 && fwrite(lst.data(), sizeof(PostingEntry), lsz, out) != lsz) return Status(StatusCode::kInternal, "write failed");
    }

    uint64_t num_codes = codes_.size();
    if (!fwrite(&num_codes, sizeof(num_codes), 1, out)) return Status(StatusCode::kInternal, "write failed");
    for (const auto& code : codes_) {
      uint64_t csz = code.size();
      if (!fwrite(&csz, sizeof(csz), 1, out)) return Status(StatusCode::kInternal, "write failed");
      if (csz > 0 && fwrite(code.data(), sizeof(uint8_t), csz, out) != csz) return Status(StatusCode::kInternal, "write failed");
    }
  }
  return Status::Ok();
}

Status IvfSq8Index::Load(std::FILE* in) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  int nlist = 0;
  if (!fread(&nlist, sizeof(nlist), 1, in)) return Status(StatusCode::kInternal, "read failed");
  if (nlist != config_.nlist) return Status(StatusCode::kInternal, "nlist mismatch");

  if (!fread(&trained_, sizeof(trained_), 1, in)) return Status(StatusCode::kInternal, "read failed");

  uint64_t buf_sz = 0;
  if (!fread(&buf_sz, sizeof(buf_sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
  buffer_ids_.resize(buf_sz);
  if (buf_sz > 0 && fread(buffer_ids_.data(), sizeof(uint32_t), buf_sz, in) != buf_sz) return Status(StatusCode::kInternal, "read failed");

  uint64_t map_sz = 0;
  if (!fread(&map_sz, sizeof(map_sz), 1, in)) return Status(StatusCode::kInternal, "read failed");
  docs_.clear();
  docs_.reserve(map_sz);
  for (size_t i = 0; i < map_sz; ++i) {
    uint32_t k;
    DocState v;
    if (!fread(&k, sizeof(k), 1, in)) return Status(StatusCode::kInternal, "read failed");
    if (!fread(&v, sizeof(DocState), 1, in)) return Status(StatusCode::kInternal, "read failed");
    docs_[k] = v;
  }

  if (trained_) {
    Status s = kmeans_.Load(in);
    if (!s.ok()) return s;

    s = quantizer_.Load(in);
    if (!s.ok()) return s;

    uint64_t num_lists = 0;
    if (!fread(&num_lists, sizeof(num_lists), 1, in)) return Status(StatusCode::kInternal, "read failed");
    lists_.resize(num_lists);
    list_metrics_.assign(num_lists, {});
    for (size_t i = 0; i < num_lists; ++i) {
      uint64_t lsz = 0;
      if (!fread(&lsz, sizeof(lsz), 1, in)) return Status(StatusCode::kInternal, "read failed");
      lists_[i].resize(lsz);
      if (lsz > 0 && fread(lists_[i].data(), sizeof(PostingEntry), lsz, in) != lsz) return Status(StatusCode::kInternal, "read failed");
    }

    uint64_t num_codes = 0;
    if (!fread(&num_codes, sizeof(num_codes), 1, in)) return Status(StatusCode::kInternal, "read failed");
    codes_.resize(num_codes);
    for (size_t i = 0; i < num_codes; ++i) {
      uint64_t csz = 0;
      if (!fread(&csz, sizeof(csz), 1, in)) return Status(StatusCode::kInternal, "read failed");
      codes_[i].resize(csz);
      if (csz > 0 && fread(codes_[i].data(), sizeof(uint8_t), csz, in) != csz) return Status(StatusCode::kInternal, "read failed");
    }
  }

  return Status::Ok();
}

}  // namespace pomai_search
