#include "pomai_search/status.h"
#include "test_framework.h"

using pomai_search::Status;
using pomai_search::StatusCode;
using pomai_search::StatusOr;

POMAI_TEST(TestStatusBasics) {
  Status ok;
  EXPECT_TRUE(ok.ok());
  Status err(StatusCode::kInvalidArgument, "bad");
  EXPECT_TRUE(!err.ok());
  EXPECT_TRUE(err.ToString().find("InvalidArgument") != std::string::npos);
  return true;
}

POMAI_TEST(TestStatusOrError) {
  Status err(StatusCode::kNotFound, "missing");
  StatusOr<int> result(err);
  EXPECT_TRUE(!result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
  EXPECT_EQ(result.value_or(7), 7);
  return true;
}

POMAI_TEST(TestStatusOrOkValue) {
  StatusOr<int> result(42);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value(), 42);
  EXPECT_EQ(result.value_or(7), 42);
  return true;
}

POMAI_TEST(TestStatusOrOkStatusIsError) {
  StatusOr<int> result(Status::Ok());
  EXPECT_TRUE(!result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInternal);
  return true;
}
