#include "pomai_search/status.h"

#include <string>

namespace pomai_search {

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  return "Status(" + std::to_string(static_cast<int>(code_)) + "): " + message_;
}

}  // namespace pomai_search
