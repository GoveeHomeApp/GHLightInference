#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <cstdarg>
#include <memory>

#ifdef __ANDROID__
#include <android/log.h>
#elif defined(__APPLE__)
#include <os/log.h>
#endif

class Logger {
public:
    enum class LogLevel {
        VERBOSE,
        DBUG,
        INFO,
        WARNING,
        ERROR
    };

    static void log(LogLevel level, const char *tag, const char *format, ...) {
        va_list args;
        va_start(args, format);

        char buffer[1024];
        vsnprintf(buffer, sizeof(buffer), format, args);

        va_end(args);

        std::string message(buffer);

#ifdef __ANDROID__
        android_LogPriority priority;
            switch (level) {
                case LogLevel::VERBOSE: priority = ANDROID_LOG_VERBOSE; break;
                case LogLevel::DBUG: priority = ANDROID_LOG_DEBUG; break;
                case LogLevel::INFO: priority = ANDROID_LOG_INFO; break;
                case LogLevel::WARNING: priority = ANDROID_LOG_WARN; break;
                case LogLevel::ERROR: priority = ANDROID_LOG_ERROR; break;
            }
            __android_log_write(priority, tag, message.c_str());
#elif defined(__APPLE__)
        os_log_type_t type;
            switch (level) {
                case LogLevel::VERBOSE: type = OS_LOG_TYPE_DEBUG; break;
                case LogLevel::DBUG: type = OS_LOG_TYPE_DEBUG; break;
                case LogLevel::INFO: type = OS_LOG_TYPE_INFO; break;
                case LogLevel::WARNING: type = OS_LOG_TYPE_DEFAULT; break;
                case LogLevel::ERROR: type = OS_LOG_TYPE_ERROR; break;
            }
            os_log_with_type(OS_LOG_DEFAULT, type, "%{public}s: %{public}s", tag, message.c_str());
#else
        std::cout << "[" << getLevelString(level) << "] " << tag << ": " << message << std::endl;
#endif
    }

private:
    static const char *getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::VERBOSE:
                return "VERBOSE";
            case LogLevel::DBUG:
                return "DEBUG";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }
};

#define LOGV(TAG, ...) Logger::log(Logger::LogLevel::VERBOSE, TAG, __VA_ARGS__)
#define LOGD(TAG, ...) Logger::log(Logger::LogLevel::DBUG, TAG, __VA_ARGS__)
#define LOGI(TAG, ...) Logger::log(Logger::LogLevel::INFO, TAG, __VA_ARGS__)
#define LOGW(TAG, ...) Logger::log(Logger::LogLevel::WARNING, TAG, __VA_ARGS__)
#define LOGE(TAG, ...) Logger::log(Logger::LogLevel::ERROR, TAG, __VA_ARGS__)

#endif // LOGGER_HPP
