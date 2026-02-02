#pragma once

#include <string>
#include <utility>

namespace pomai_search {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kResourceExhausted,
  kInternal,
  kUnavailable,
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  explicit Status(StatusCode code, std::string message = {})
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  std::string ToString() const;

 private:
  StatusCode code_;
  std::string message_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(const Status& status) : status_(status) {}
  StatusOr(Status&& status) : status_(std::move(status)) {}
  StatusOr(const T& value) : status_(Status::Ok()), value_(value), has_value_(true) {}
  StatusOr(T&& value) : status_(Status::Ok()), value_(std::move(value)), has_value_(true) {}

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  const T& value() const & { return value_; }
  T& value() & { return value_; }
  T&& value() && { return std::move(value_); }

 private:
  Status status_{};
  T value_{};
  bool has_value_ = false;
};

}  // namespace pomai_search
