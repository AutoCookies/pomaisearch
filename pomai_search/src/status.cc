#include "pomai_search/status.h"

namespace pomai_search {

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  std::string code;
  switch (code_) {
    case StatusCode::kOk:
      code = "OK";
      break;
    case StatusCode::kInvalidArgument:
      code = "InvalidArgument";
      break;
    case StatusCode::kNotFound:
      code = "NotFound";
      break;
    case StatusCode::kAlreadyExists:
      code = "AlreadyExists";
      break;
    case StatusCode::kResourceExhausted:
      code = "ResourceExhausted";
      break;
    case StatusCode::kInternal:
      code = "Internal";
      break;
    case StatusCode::kUnavailable:
      code = "Unavailable";
      break;
  }
  if (message_.empty()) {
    return code;
  }
  return code + ": " + message_;
}

}  // namespace pomai_search
