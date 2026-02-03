#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pomai_search {

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  double number = 0.0;
  bool boolean = false;
  std::string string;
  std::vector<JsonValue> array;
  std::unordered_map<std::string, JsonValue> object;
};

std::string JsonEscape(std::string_view input);

bool ParseJson(std::string_view input, JsonValue* out, std::string* error);

const JsonValue* FindField(const JsonValue& object, const std::string& name);

bool GetStringField(const JsonValue& object, const std::string& name, std::string* out);
bool GetIntField(const JsonValue& object, const std::string& name, int* out);
bool GetBoolField(const JsonValue& object, const std::string& name, bool* out);
bool GetFloatArrayField(const JsonValue& object, const std::string& name, std::vector<float>* out);
bool GetStringMapField(const JsonValue& object, const std::string& name,
                       std::unordered_map<std::string, std::string>* out);

}  // namespace pomai_search
