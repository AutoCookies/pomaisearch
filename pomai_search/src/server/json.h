#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai_search/status.h"

namespace pomai_search {

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::unordered_map<std::string, JsonValue> object_value;

  static JsonValue MakeNull();
  static JsonValue MakeBool(bool value);
  static JsonValue MakeNumber(double value);
  static JsonValue MakeString(std::string value);
  static JsonValue MakeArray(std::vector<JsonValue> value);
  static JsonValue MakeObject(std::unordered_map<std::string, JsonValue> value);
};

StatusOr<JsonValue> ParseJson(std::string_view input);

}  // namespace pomai_search
