# Performance Tuning Guide

Optimize Pomai Search for your specific workload.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Index Selection](#index-selection)
- [Parameter Tuning](#parameter-tuning)
- [Hardware Optimization](#hardware-optimization)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Quick Wins

### 1. Enable AVX2

```cpp
SearchEngineConfig cfg;
cfg.enable_avx2 = true;  // 4-8x faster dot products
```

**Requirement:** CPU with AVX2 support (Intel Haswell+, AMD Excavator+)

### 2. Use Multiple Shards

```cpp
cfg.num_shards = std::thread::hardware_concurrency();  // Use all cores
```

**Rule of thumb:** 1-2 shards per CPU core for query-heavy workloads

### 3. Pre-normalize Vectors

For cosine similarity, normalize vectors before insertion:

```cpp
void normalize(float* vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; ++i) vec[i] /= norm;
}

// Then use Dot similarity (equivalent to Cosine for normalized vectors)
cfg.similarity = SearchEngineConfig::Similarity::Dot;
```

**Speedup:** ~2x (avoids normalization during search)

### 4. Use Snapshots

```cpp
// Save once
SnapshotWriter::Write(*engine, "index.pomai");

// Load instantly (no re-indexing)
auto engine = SnapshotReader::Read("index.pomai").value();
```

**Benefit:** O(N) load vs O(N log N) rebuild

---

## Index Selection

### Decision Tree

```
Start
  │
  ├─ Need 100% recall? ──Yes──> Flat
  │                      
  No
  │
  ├─ Dataset size?
  │   ├─ <100K vectors ──> HNSW (best recall/speed)
  │   ├─ 100K-10M vectors ──> IVF-Flat or HNSW
  │   └─ >10M vectors ──> IVF-SQ8 (memory efficient)
  │
  ├─ Memory constrained? ──Yes──> IVF-SQ8
  │                        
  No
  │
  └─ Need >99% recall? ──Yes──> HNSW
                         No──> IVF-Flat
```

### Index Comparison

| Metric | Flat | HNSW | IVF-Flat | IVF-SQ8 |
|--------|------|------|----------|---------|
| **Recall** | 100% | 99.5% | 92% | 96% |
| **QPS (1M)** | 50 | 5,000 | 2,000 | 3,500 |
| **Memory** | 1x | 1.4x | 1.1x | 0.35x |
| **Build Time** | Fast | Slow | Medium | Medium |
| **Best For** | Exact | High recall | Large scale | Memory limited |

---

## Parameter Tuning

### HNSW Parameters

#### M (Graph Connectivity)

**Effect:** Higher M = better recall, more memory

```cpp
cfg.hnsw_m = 16;  // Default, good balance
cfg.hnsw_m = 32;  // High recall (99.8%+)
cfg.hnsw_m = 8;   // Low memory
```

**Memory impact:** ~12 bytes per edge per vector

**Recommendation:**
- M=8: Memory constrained
- M=16: Default (recommended)
- M=32: Maximum recall
- M=64: Overkill (diminishing returns)

#### ef_construction (Build Quality)

**Effect:** Higher = better graph quality, slower build

```cpp
cfg.hnsw_ef_construction = 200;  // Default
cfg.hnsw_ef_construction = 400;  // High quality
cfg.hnsw_ef_construction = 100;  // Fast build
```

**Build time impact:** Linear (2x ef = 2x build time)

**Recommendation:**
- 100: Fast prototyping
- 200: Production default
- 400: Maximum quality
- 800+: Rarely needed

#### ef_search (Query Quality)

**Effect:** Higher = better recall, slower queries

```cpp
cfg.hnsw_ef_search = 50;   // Default
cfg.hnsw_ef_search = 100;  // High recall
cfg.hnsw_ef_search = 20;   // Fast queries
```

**Query time impact:** Linear

**Recommendation:**
- 20: Fast queries (95% recall)
- 50: Balanced (99% recall)
- 100: High recall (99.5% recall)
- 200+: Diminishing returns

#### Tuning Example

**Goal: 99.8% recall, acceptable latency**

```cpp
cfg.hnsw_m = 32;
cfg.hnsw_ef_construction = 400;
cfg.hnsw_ef_search = 100;
```

**Goal: Fast queries, 98% recall acceptable**

```cpp
cfg.hnsw_m = 16;
cfg.hnsw_ef_construction = 200;
cfg.hnsw_ef_search = 30;
```

---

### IVF Parameters

#### nlist (Number of Clusters)

**Effect:** Higher = faster queries, lower recall

```cpp
cfg.ivf_nlist = 100;   // Default
cfg.ivf_nlist = 1000;  // Large dataset
cfg.ivf_nlist = 50;    // Small dataset
```

**Rule of thumb:** `nlist = sqrt(N)` where N = number of vectors

**Examples:**
- 10K vectors: nlist = 100
- 100K vectors: nlist = 316
- 1M vectors: nlist = 1000
- 10M vectors: nlist = 3162

#### nprobe (Clusters to Search)

**Effect:** Higher = better recall, slower queries

```cpp
cfg.ivf_nprobe = 10;   // Default
cfg.ivf_nprobe = 20;   // Better recall
cfg.ivf_nprobe = 5;    // Faster queries
```

**Recall/Speed Tradeoff:**

| nprobe | Recall | Relative Speed |
|--------|--------|----------------|
| 1 | 60% | 10x |
| 5 | 85% | 2x |
| 10 | 92% | 1x |
| 20 | 96% | 0.5x |
| 50 | 98% | 0.2x |

**Recommendation:** Start with nprobe = nlist / 10

#### Tuning Example

**1M vectors, balanced:**
```cpp
cfg.ivf_nlist = 1000;
cfg.ivf_nprobe = 10;
```

**10M vectors, high recall:**
```cpp
cfg.ivf_nlist = 3000;
cfg.ivf_nprobe = 30;
```

---

## Hardware Optimization

### CPU

**Recommendations:**
- **Cores:** More cores = better parallelism
- **Cache:** Larger L3 cache helps with graph traversal
- **SIMD:** AVX2 or AVX-512 for 4-8x speedup

**Optimal CPU:**
- Intel Xeon (Cascade Lake or newer)
- AMD EPYC (Rome or newer)
- Apple M1/M2 (excellent SIMD)

### Memory

**Requirements:**

| Dataset Size | Flat | HNSW | IVF-SQ8 |
|-------------|------|------|---------|
| 1M × 128-dim | 512 MB | 720 MB | 180 MB |
| 10M × 128-dim | 5.1 GB | 7.2 GB | 1.8 GB |
| 100M × 128-dim | 51 GB | 72 GB | 18 GB |

**Recommendations:**
- Use IVF-SQ8 for large datasets
- Consider distributed deployment for >100M vectors
- Use memory-mapped files for datasets larger than RAM (future feature)

### Storage

**Snapshot I/O:**
- **SSD:** 10x faster load times vs HDD
- **NVMe:** 2-3x faster than SATA SSD
- **Network storage:** Acceptable for infrequent loads

**Benchmark (1M vectors):**
- HDD: 5s load time
- SATA SSD: 0.5s load time
- NVMe SSD: 0.2s load time

---

## Monitoring

### Metrics

```cpp
std::string metrics = engine->MetricsJson();
// Parse and monitor:
// - query_latency_p50, p95, p99
// - queries_per_second
// - upserts_per_second
```

### Key Metrics to Track

1. **Query Latency (p95, p99)**
   - Target: <10ms for HNSW, <50ms for IVF
   - Alert if p99 > 100ms

2. **Throughput (QPS)**
   - Baseline: Measure at startup
   - Alert if drops >20%

3. **Memory Usage**
   - Track RSS (resident set size)
   - Alert if approaching limit

4. **Recall (offline)**
   - Periodically test against ground truth
   - Alert if recall drops >2%

### Example Monitoring Setup

```cpp
// Periodic metrics collection
void monitor() {
    auto stats = engine->GetStats();
    auto metrics = engine->MetricsJson();
    
    // Log to monitoring system
    log("num_points", stats.num_points);
    log("num_deleted", stats.num_deleted);
    log("metrics", metrics);
    
    // Alert if needed
    if (stats.num_deleted > stats.num_points * 0.3) {
        alert("High deletion rate, consider compaction");
    }
}
```

---

## Troubleshooting

### High Query Latency

**Symptoms:** p99 latency >100ms

**Diagnosis:**
1. Check CPU usage: `top` or `htop`
2. Check memory: `free -h`
3. Profile queries: Use `SearchWithExplain()`

**Solutions:**
- Reduce `ef_search` or `nprobe`
- Add more shards
- Upgrade CPU
- Use IVF instead of HNSW for large datasets

### Low Recall

**Symptoms:** Search results don't match expectations

**Diagnosis:**
1. Measure recall against ground truth
2. Check index parameters

**Solutions:**
- Increase `ef_search` (HNSW)
- Increase `nprobe` (IVF)
- Use higher `M` (HNSW)
- Switch to Flat index for exact search

### High Memory Usage

**Symptoms:** OOM errors, swapping

**Diagnosis:**
1. Check `GetStats().num_points`
2. Calculate expected memory
3. Profile with `valgrind --tool=massif`

**Solutions:**
- Switch to IVF-SQ8 (4x reduction)
- Reduce `hnsw_m`
- Delete unused documents
- Add more RAM
- Use distributed deployment

### Slow Startup

**Symptoms:** Engine takes >10s to open

**Diagnosis:**
1. Check if using snapshots
2. Measure snapshot load time

**Solutions:**
- Use snapshots instead of rebuilding
- Use faster storage (SSD/NVMe)
- Reduce dataset size
- Use distributed deployment

### Inconsistent Results

**Symptoms:** Same query returns different results

**Diagnosis:**
1. Check if vectors are being updated
2. Verify `global_seed` is set

**Solutions:**
- Set `cfg.global_seed` to fixed value
- Use `SearchWithExplain()` to debug
- Check for concurrent updates

---

## Benchmarking

### Running Benchmarks

```bash
cd build
./benchmark --dataset sift1m --index hnsw --topk 10
```

### Interpreting Results

**Key metrics:**
- **Build time:** Time to index all vectors
- **QPS:** Queries per second (higher = better)
- **Recall@k:** Fraction of true top-k found (higher = better)
- **Memory:** RSS in MB (lower = better)

**Example output:**
```
Index: HNSW
Build time: 120.5s
QPS: 4,823
Recall@10: 0.995
Memory: 720 MB
```

### Comparing Configurations

```bash
# Test different ef_search values
for ef in 20 50 100 200; do
    ./benchmark --index hnsw --ef_search $ef
done
```

---

## Best Practices Summary

1. ✅ **Use AVX2** for 4-8x speedup
2. ✅ **Choose right index** for your dataset size
3. ✅ **Tune parameters** based on recall/speed requirements
4. ✅ **Use snapshots** for fast restarts
5. ✅ **Monitor metrics** continuously
6. ✅ **Benchmark** before deploying
7. ✅ **Pre-normalize** vectors for cosine similarity
8. ✅ **Use multiple shards** for parallelism
9. ✅ **Set capacity limits** to prevent OOM
10. ✅ **Profile** before optimizing

---

## Performance Checklist

Before going to production:

- [ ] Benchmarked on representative dataset
- [ ] Measured recall on test set (>95%)
- [ ] Profiled query latency (p99 <100ms)
- [ ] Tested snapshot save/load
- [ ] Configured monitoring
- [ ] Set up alerts
- [ ] Documented configuration
- [ ] Tested failure scenarios
- [ ] Verified memory limits
- [ ] Enabled AVX2

---

## Getting Help

If you're still experiencing performance issues:

1. **Measure first:** Use profiling tools
2. **Share metrics:** Include QPS, latency, recall
3. **Describe workload:** Dataset size, query pattern
4. **Try defaults:** Start with recommended parameters
5. **Ask community:** Discord, GitHub Issues
