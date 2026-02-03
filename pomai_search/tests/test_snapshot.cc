#include "pomai_search/search_engine.h"
#include "core/serialize/snapshot.h"
#include "tests/test_framework.h"

#include <thread>
#include <chrono>

namespace pomai_search::test {

POMAI_TEST(SnapshotRoundTrip) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  cfg.global_seed = 55;
  cfg.contract_version = 3;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  float a[2] = {1.0f, 0.0f};
  float b[2] = {0.0f, 1.0f};
  engine->Upsert("doc-a", VectorView{a, 2}, {{"tag", "alpha"}}, std::nullopt, "alpha");
  engine->Upsert("doc-b", VectorView{b, 2}, {{"tag", "beta"}}, std::nullopt, "beta");
  float q[2] = {1.0f, 0.0f};
  auto before = engine->Search(VectorView{q, 2});
  EXPECT_TRUE(before.ok());

  Status snapshot_status = SnapshotWriter::Write(*engine, "tests/contracts/tmp_snapshot.bin");
  EXPECT_TRUE(snapshot_status.ok());
  auto restored_or = SnapshotReader::Read("tests/contracts/tmp_snapshot.bin");
  EXPECT_TRUE(restored_or.ok());
  auto restored = std::move(restored_or.value());
  auto after = restored->Search(VectorView{q, 2});
  EXPECT_TRUE(after.ok());
  EXPECT_EQ(before.value().size(), after.value().size());
  for (size_t i = 0; i < before.value().size(); ++i) {
    EXPECT_EQ(before.value()[i].key, after.value()[i].key);
    EXPECT_NEAR(before.value()[i].score, after.value()[i].score, 1e-5f);
  }
  return true;
}

POMAI_TEST(SnapshotTTL) {
  SearchEngineConfig cfg;
  cfg.dim = 2;
  cfg.num_shards = 1;
  auto engine_or = SearchEngine::Open(cfg);
  EXPECT_TRUE(engine_or.ok());
  auto engine = std::move(engine_or.value());
  
  float a[2] = {1.0f, 0.0f};
  // Long TTL (1 hour)
  engine->Upsert("doc-long", VectorView{a, 2}, {}, std::chrono::hours(1), std::nullopt);
  // Short TTL (100ms)
  engine->Upsert("doc-short", VectorView{a, 2}, {}, std::chrono::milliseconds(100), std::nullopt);

  Status snapshot_status = SnapshotWriter::Write(*engine, "tests/contracts/ttl_snapshot.bin");
  EXPECT_TRUE(snapshot_status.ok());
  
  auto restored_or = SnapshotReader::Read("tests/contracts/ttl_snapshot.bin");
  EXPECT_TRUE(restored_or.ok());
  auto restored = std::move(restored_or.value());
  
  // doc-long should exist
  auto res_long = restored->Search(VectorView{a, 2});
  EXPECT_TRUE(res_long.ok());
  bool found_long = false;
  for(const auto& r : res_long.value()) if(r.key == "doc-long") found_long = true;
  EXPECT_TRUE(found_long);
  
  // Wait for doc-short to expire
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // doc-short should be gone
  auto res_short = restored->Search(VectorView{a, 2});
  EXPECT_TRUE(res_short.ok());
  bool found_short = false;
  for(const auto& r : res_short.value()) if(r.key == "doc-short") found_short = true;
  EXPECT_TRUE(!found_short);
  
  return true;
}

}  // namespace pomai_search::test
