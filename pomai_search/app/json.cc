#include "app/json.h"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace pomai_search {

std::string JsonEscape(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::uppercase
              << static_cast<int>(static_cast<unsigned char>(c));
          out += oss.str();
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}

namespace {

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  bool Parse(JsonValue* out, std::string* error) {
    error_ = error;
    SkipWhitespace();
    if (!ParseValue(out)) {
      return false;
    }
    SkipWhitespace();
    if (pos_ != input_.size()) {
      return Fail("Unexpected trailing characters");
    }
    return true;
  }

 private:
  bool ParseValue(JsonValue* out) {
    if (pos_ >= input_.size()) {
      return Fail("Unexpected end of input");
    }
    char c = input_[pos_];
    if (c == '"') {
      out->type = JsonValue::Type::kString;
      return ParseString(&out->string);
    }
    if (c == '{') {
      return ParseObject(out);
    }
    if (c == '[') {
      return ParseArray(out);
    }
    if (c == 't' || c == 'f') {
      return ParseBool(out);
    }
    if (c == 'n') {
      return ParseNull(out);
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      out->type = JsonValue::Type::kNumber;
      return ParseNumber(&out->number);
    }
    return Fail("Unexpected character");
  }

  bool ParseObject(JsonValue* out) {
    out->type = JsonValue::Type::kObject;
    out->object.clear();
    ++pos_;
    SkipWhitespace();
    if (pos_ < input_.size() && input_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (pos_ < input_.size()) {
      SkipWhitespace();
      std::string key;
      if (!ParseString(&key)) {
        return false;
      }
      SkipWhitespace();
      if (!Consume(':')) {
        return Fail("Expected ':' in object");
      }
      SkipWhitespace();
      JsonValue value;
      if (!ParseValue(&value)) {
        return false;
      }
      out->object.emplace(std::move(key), std::move(value));
      SkipWhitespace();
      if (Consume('}')) {
        return true;
      }
      if (!Consume(',')) {
        return Fail("Expected ',' in object");
      }
    }
    return Fail("Unterminated object");
  }

  bool ParseArray(JsonValue* out) {
    out->type = JsonValue::Type::kArray;
    out->array.clear();
    ++pos_;
    SkipWhitespace();
    if (pos_ < input_.size() && input_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (pos_ < input_.size()) {
      SkipWhitespace();
      JsonValue value;
      if (!ParseValue(&value)) {
        return false;
      }
      out->array.push_back(std::move(value));
      SkipWhitespace();
      if (Consume(']')) {
        return true;
      }
      if (!Consume(',')) {
        return Fail("Expected ',' in array");
      }
    }
    return Fail("Unterminated array");
  }

  bool ParseString(std::string* out) {
    if (!Consume('"')) {
      return Fail("Expected '\"' to start string");
    }
    std::string result;
    while (pos_ < input_.size()) {
      char c = input_[pos_++];
      if (c == '"') {
        *out = std::move(result);
        return true;
      }
      if (c == '\\') {
        if (pos_ >= input_.size()) {
          return Fail("Unterminated escape");
        }
        char esc = input_[pos_++];
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            result += esc;
            break;
          case 'b':
            result += '\b';
            break;
          case 'f':
            result += '\f';
            break;
          case 'n':
            result += '\n';
            break;
          case 'r':
            result += '\r';
            break;
          case 't':
            result += '\t';
            break;
          case 'u':
            if (pos_ + 3 >= input_.size()) {
              return Fail("Invalid unicode escape");
            }
            pos_ += 4;
            result += '?';
            break;
          default:
            return Fail("Invalid escape");
        }
      } else {
        result += c;
      }
    }
    return Fail("Unterminated string");
  }

  bool ParseNumber(double* out) {
    size_t start = pos_;
    if (input_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
    if (pos_ < input_.size() && input_[pos_] == '.') {
      ++pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
      }
    }
    std::string num = std::string(input_.substr(start, pos_ - start));
    char* end = nullptr;
    *out = std::strtod(num.c_str(), &end);
    if (end == num.c_str()) {
      return Fail("Invalid number");
    }
    return true;
  }

  bool ParseBool(JsonValue* out) {
    if (input_.substr(pos_, 4) == "true") {
      pos_ += 4;
      out->type = JsonValue::Type::kBool;
      out->boolean = true;
      return true;
    }
    if (input_.substr(pos_, 5) == "false") {
      pos_ += 5;
      out->type = JsonValue::Type::kBool;
      out->boolean = false;
      return true;
    }
    return Fail("Invalid bool");
  }

  bool ParseNull(JsonValue* out) {
    if (input_.substr(pos_, 4) != "null") {
      return Fail("Invalid null");
    }
    pos_ += 4;
    out->type = JsonValue::Type::kNull;
    return true;
  }

  void SkipWhitespace() {
    while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
  }

  bool Consume(char c) {
    if (pos_ < input_.size() && input_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  bool Fail(const std::string& message) {
    if (error_) {
      *error_ = message;
    }
    return false;
  }

  std::string_view input_;
  size_t pos_ = 0;
  std::string* error_ = nullptr;
};

}  // namespace

bool ParseJson(std::string_view input, JsonValue* out, std::string* error) {
  JsonParser parser(input);
  return parser.Parse(out, error);
}

const JsonValue* FindField(const JsonValue& object, const std::string& name) {
  auto it = object.object.find(name);
  if (it == object.object.end()) {
    return nullptr;
  }
  return &it->second;
}

bool GetStringField(const JsonValue& object, const std::string& name, std::string* out) {
  const JsonValue* field = FindField(object, name);
  if (!field || field->type != JsonValue::Type::kString) {
    return false;
  }
  *out = field->string;
  return true;
}

bool GetIntField(const JsonValue& object, const std::string& name, int* out) {
  const JsonValue* field = FindField(object, name);
  if (!field || field->type != JsonValue::Type::kNumber) {
    return false;
  }
  *out = static_cast<int>(field->number);
  return true;
}

bool GetBoolField(const JsonValue& object, const std::string& name, bool* out) {
  const JsonValue* field = FindField(object, name);
  if (!field || field->type != JsonValue::Type::kBool) {
    return false;
  }
  *out = field->boolean;
  return true;
}

bool GetFloatArrayField(const JsonValue& object, const std::string& name, std::vector<float>* out) {
  const JsonValue* field = FindField(object, name);
  if (!field || field->type != JsonValue::Type::kArray) {
    return false;
  }
  std::vector<float> values;
  values.reserve(field->array.size());
  for (const auto& item : field->array) {
    if (item.type != JsonValue::Type::kNumber) {
      return false;
    }
    values.push_back(static_cast<float>(item.number));
  }
  *out = std::move(values);
  return true;
}

bool GetStringMapField(const JsonValue& object, const std::string& name,
                       std::unordered_map<std::string, std::string>* out) {
  const JsonValue* field = FindField(object, name);
  if (!field || field->type != JsonValue::Type::kObject) {
    return false;
  }
  std::unordered_map<std::string, std::string> map;
  for (const auto& pair : field->object) {
    if (pair.second.type != JsonValue::Type::kString) {
      return false;
    }
    map.emplace(pair.first, pair.second.string);
  }
  *out = std::move(map);
  return true;
}

}  // namespace pomai_search
