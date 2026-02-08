#pragma once

#include <cassert>
#include <cstdlib>
#include <optional>
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
  StatusOr(const Status& status) : status_(status) { EnsureNotOk(); }
  StatusOr(Status&& status) : status_(std::move(status)) { EnsureNotOk(); }
  StatusOr(const T& value) : status_(Status::Ok()), value_(value) {}
  StatusOr(T&& value) : status_(Status::Ok()), value_(std::move(value)) {}

  StatusOr(const StatusOr& other) = default;
  StatusOr(StatusOr&& other) noexcept = default;
  StatusOr& operator=(const StatusOr& other) = default;
  StatusOr& operator=(StatusOr&& other) noexcept = default;

  bool ok() const { return status_.ok(); }
  explicit operator bool() const { return ok(); }
  const Status& status() const { return status_; }

  const T& value() const & {
    CheckOk();
    return *value_;
  }
  T& value() & {
    CheckOk();
    return *value_;
  }
  T&& value() && {
    CheckOk();
    return std::move(*value_);
  }

  const T& ValueOrDie() const & { return value(); }
  T& ValueOrDie() & { return value(); }
  T&& ValueOrDie() && { return std::move(*this).value(); }

  T value_or(T default_value) const {
    return ok() ? *value_ : std::move(default_value);
  }

 private:
  void EnsureNotOk() {
    if (status_.ok()) {
      status_ = Status(StatusCode::kInternal, "StatusOr constructed with OK status");
    }
  }

  void CheckOk() const {
    if (!ok()) {
#ifndef NDEBUG
      assert(false && "Accessed value of non-OK StatusOr");
#endif
      std::abort();
    }
  }

  Status status_{};
  std::optional<T> value_{};
};

}  // namespace pomai_search
