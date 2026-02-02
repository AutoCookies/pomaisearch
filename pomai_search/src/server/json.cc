#include "json.h"

#include <cctype>

namespace pomai_search {

JsonValue JsonValue::MakeNull() { return JsonValue{}; }
JsonValue JsonValue::MakeBool(bool value) {
  JsonValue v;
  v.type = Type::Bool;
  v.bool_value = value;
  return v;
}
JsonValue JsonValue::MakeNumber(double value) {
  JsonValue v;
  v.type = Type::Number;
  v.number_value = value;
  return v;
}
JsonValue JsonValue::MakeString(std::string value) {
  JsonValue v;
  v.type = Type::String;
  v.string_value = std::move(value);
  return v;
}
JsonValue JsonValue::MakeArray(std::vector<JsonValue> value) {
  JsonValue v;
  v.type = Type::Array;
  v.array_value = std::move(value);
  return v;
}
JsonValue JsonValue::MakeObject(std::unordered_map<std::string, JsonValue> value) {
  JsonValue v;
  v.type = Type::Object;
  v.object_value = std::move(value);
  return v;
}

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  StatusOr<JsonValue> Parse() {
    SkipWhitespace();
    auto value = ParseValue();
    if (!value.ok()) {
      return value;
    }
    SkipWhitespace();
    if (pos_ != input_.size()) {
      return Status(StatusCode::kInvalidArgument, "trailing input");
    }
    return value;
  }

 private:
  void SkipWhitespace() {
    while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
  }

  StatusOr<JsonValue> ParseValue() {
    if (pos_ >= input_.size()) {
      return Status(StatusCode::kInvalidArgument, "unexpected end");
    }
    char c = input_[pos_];
    if (c == 'n') {
      return ParseLiteral("null", JsonValue::MakeNull());
    }
    if (c == 't') {
      return ParseLiteral("true", JsonValue::MakeBool(true));
    }
    if (c == 'f') {
      return ParseLiteral("false", JsonValue::MakeBool(false));
    }
    if (c == '"') {
      return ParseString();
    }
    if (c == '[') {
      return ParseArray();
    }
    if (c == '{') {
      return ParseObject();
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return ParseNumber();
    }
    return Status(StatusCode::kInvalidArgument, "invalid token");
  }

  StatusOr<JsonValue> ParseLiteral(std::string_view literal, JsonValue value) {
    if (input_.substr(pos_, literal.size()) != literal) {
      return Status(StatusCode::kInvalidArgument, "invalid literal");
    }
    pos_ += literal.size();
    return value;
  }

  StatusOr<JsonValue> ParseString() {
    if (input_[pos_] != '"') {
      return Status(StatusCode::kInvalidArgument, "expected string");
    }
    ++pos_;
    std::string out;
    while (pos_ < input_.size()) {
      char c = input_[pos_++];
      if (c == '"') {
        return JsonValue::MakeString(out);
      }
      if (c == '\\') {
        if (pos_ >= input_.size()) {
          return Status(StatusCode::kInvalidArgument, "invalid escape");
        }
        char esc = input_[pos_++];
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            out.push_back(esc);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            return Status(StatusCode::kInvalidArgument, "unsupported escape");
        }
        continue;
      }
      out.push_back(c);
    }
    return Status(StatusCode::kInvalidArgument, "unterminated string");
  }

  StatusOr<JsonValue> ParseNumber() {
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
    std::string number(input_.substr(start, pos_ - start));
    if (number.empty() || number == "-") {
      return Status(StatusCode::kInvalidArgument, "invalid number");
    }
    char* end = nullptr;
    double value = std::strtod(number.c_str(), &end);
    if (end == number.c_str()) {
      return Status(StatusCode::kInvalidArgument, "invalid number");
    }
    return JsonValue::MakeNumber(value);
  }

  StatusOr<JsonValue> ParseArray() {
    if (input_[pos_] != '[') {
      return Status(StatusCode::kInvalidArgument, "expected array");
    }
    ++pos_;
    SkipWhitespace();
    std::vector<JsonValue> values;
    if (pos_ < input_.size() && input_[pos_] == ']') {
      ++pos_;
      return JsonValue::MakeArray(values);
    }
    while (true) {
      SkipWhitespace();
      auto value = ParseValue();
      if (!value.ok()) {
        return value;
      }
      values.push_back(value.value());
      SkipWhitespace();
      if (pos_ >= input_.size()) {
        return Status(StatusCode::kInvalidArgument, "unterminated array");
      }
      char c = input_[pos_++];
      if (c == ']') {
        break;
      }
      if (c != ',') {
        return Status(StatusCode::kInvalidArgument, "expected comma");
      }
    }
    return JsonValue::MakeArray(values);
  }

  StatusOr<JsonValue> ParseObject() {
    if (input_[pos_] != '{') {
      return Status(StatusCode::kInvalidArgument, "expected object");
    }
    ++pos_;
    SkipWhitespace();
    std::unordered_map<std::string, JsonValue> values;
    if (pos_ < input_.size() && input_[pos_] == '}') {
      ++pos_;
      return JsonValue::MakeObject(values);
    }
    while (true) {
      SkipWhitespace();
      auto key = ParseString();
      if (!key.ok()) {
        return key;
      }
      SkipWhitespace();
      if (pos_ >= input_.size() || input_[pos_] != ':') {
        return Status(StatusCode::kInvalidArgument, "expected colon");
      }
      ++pos_;
      SkipWhitespace();
      auto value = ParseValue();
      if (!value.ok()) {
        return value;
      }
      values.emplace(key.value().string_value, value.value());
      SkipWhitespace();
      if (pos_ >= input_.size()) {
        return Status(StatusCode::kInvalidArgument, "unterminated object");
      }
      char c = input_[pos_++];
      if (c == '}') {
        break;
      }
      if (c != ',') {
        return Status(StatusCode::kInvalidArgument, "expected comma");
      }
    }
    return JsonValue::MakeObject(values);
  }

  std::string_view input_;
  size_t pos_ = 0;
};

StatusOr<JsonValue> ParseJson(std::string_view input) {
  JsonParser parser(input);
  return parser.Parse();
}

}  // namespace pomai_search
