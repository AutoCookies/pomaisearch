#include "pomai_search/search_engine.h"
#include "pomai_search/snapshot.h"
#include "tests/test_framework.h"

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

}  // namespace pomai_search::test
