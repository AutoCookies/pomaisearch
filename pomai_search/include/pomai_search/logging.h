#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace pomai_search {

enum class LogLevel {
  kDebug = 0,
  kInfo,
  kWarn,
  kError,
};

class Logger {
 public:
  static Logger& Instance();

  void set_level(LogLevel level) { level_.store(level); }
  LogLevel level() const { return level_.load(); }

  void Log(LogLevel level, const std::string& message);
  void set_callback(std::function<void(LogLevel, const std::string&)> callback);

 private:
  Logger() = default;
  std::atomic<LogLevel> level_{LogLevel::kInfo};
  std::mutex callback_mutex_;
  std::function<void(LogLevel, const std::string&)> callback_;
};

}  // namespace pomai_search

#define POMAI_LOG_DEBUG(msg) ::pomai_search::Logger::Instance().Log(::pomai_search::LogLevel::kDebug, msg)
#define POMAI_LOG_INFO(msg) ::pomai_search::Logger::Instance().Log(::pomai_search::LogLevel::kInfo, msg)
#define POMAI_LOG_WARN(msg) ::pomai_search::Logger::Instance().Log(::pomai_search::LogLevel::kWarn, msg)
#define POMAI_LOG_ERROR(msg) ::pomai_search::Logger::Instance().Log(::pomai_search::LogLevel::kError, msg)
