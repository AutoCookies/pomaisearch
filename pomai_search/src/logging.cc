#include "pomai_search/logging.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace pomai_search {

Logger& Logger::Instance() {
  static Logger instance;
  return instance;
}

void Logger::set_callback(std::function<void(LogLevel, const std::string&)> callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  callback_ = std::move(callback);
}

void Logger::Log(LogLevel level, const std::string& message) {
  if (level < level_.load()) {
    return;
  }
  std::function<void(LogLevel, const std::string&)> callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = callback_;
  }
  if (callback) {
    callback(level, message);
    return;
  }
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &now_time);
#else
  localtime_r(&now_time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  const char* level_str = "INFO";
  switch (level) {
    case LogLevel::kDebug:
      level_str = "DEBUG";
      break;
    case LogLevel::kInfo:
      level_str = "INFO";
      break;
    case LogLevel::kWarn:
      level_str = "WARN";
      break;
    case LogLevel::kError:
      level_str = "ERROR";
      break;
  }
  std::cerr << "[" << oss.str() << "] " << level_str << " " << message << "\n";
}

}  // namespace pomai_search
