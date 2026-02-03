#include "pomai_search/status.h"
#include "tests/test_framework.h"

namespace pomai_search::test {

POMAI_TEST(StatusOrOkPath) {
  StatusOr<int> ok_value(42);
  EXPECT_TRUE(ok_value.ok());
  EXPECT_EQ(ok_value.value(), 42);
  EXPECT_EQ(ok_value.ValueOrDie(), 42);
  EXPECT_EQ(ok_value.value_or(7), 42);
  return true;
}

POMAI_TEST(StatusOrErrorPath) {
  StatusOr<int> error(Status(StatusCode::kInvalidArgument, "bad"));
  EXPECT_TRUE(!error.ok());
  EXPECT_EQ(error.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(error.status().message(), "bad");
  EXPECT_EQ(error.value_or(9), 9);
  return true;
}

POMAI_TEST(StatusOrRejectsOkStatus) {
  StatusOr<int> error(Status::Ok());
  EXPECT_TRUE(!error.ok());
  EXPECT_EQ(error.status().code(), StatusCode::kInternal);
  return true;
}

POMAI_TEST(StatusToString) {
  Status status(StatusCode::kNotFound, "missing");
  EXPECT_TRUE(status.ToString().find("missing") != std::string::npos);
  return true;
}

}  // namespace pomai_search::test
