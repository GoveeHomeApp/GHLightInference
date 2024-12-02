//
// Created by linpeng on 2024/7/3.
//
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
        L_VERBOSE,
        L_DEBUG,
        L_INFO,
        L_WARNING,
        L_ERROR
    };

    static void log(LogLevel level, const char *tag, const char *format, ...) {
//        if (true)return;
        va_list args;
        va_start(args, format);

        char buffer[1024];
        vsnprintf(buffer, sizeof(buffer), format, args);

        va_end(args);

        std::string message(buffer);

#ifdef __ANDROID__
        android_LogPriority priority;
            switch (level) {
                case LogLevel::L_VERBOSE:
                    priority = ANDROID_LOG_VERBOSE;
                    break;
                case LogLevel::L_DEBUG:
                    priority = ANDROID_LOG_DEBUG;
                    break;
                case LogLevel::L_INFO:
                    priority = ANDROID_LOG_INFO;
                    break;
                case LogLevel::L_WARNING:
                    priority = ANDROID_LOG_WARN;
                    break;
                case LogLevel::L_ERROR:
                    priority = ANDROID_LOG_ERROR;
                    break;
            }
            __android_log_write(priority, tag, message.c_str());
#elif defined(__APPLE__)
        os_log_type_t type;
            switch (level) {
                case LogLevel::L_VERBOSE: type = OS_LOG_TYPE_DEBUG; break;
                case LogLevel::L_DEBUG: type = OS_LOG_TYPE_DEBUG; break;
                case LogLevel::L_INFO: type = OS_LOG_TYPE_INFO; break;
                case LogLevel::L_WARNING: type = OS_LOG_TYPE_DEFAULT; break;
                case LogLevel::L_ERROR: type = OS_LOG_TYPE_ERROR; break;
            }
            os_log_with_type(OS_LOG_DEFAULT, type, "%{public}s: %{public}s", tag, message.c_str());
#else
        std::cout << "[" << getLevelString(level) << "] " << tag << ": " << message << std::endl;
#endif
    }

private:

    static const char *getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::L_VERBOSE:
                return "VERBOSE";
            case LogLevel::L_DEBUG:
                return "DEBUG";
            case LogLevel::L_INFO:
                return "INFO";
            case LogLevel::L_WARNING:
                return "WARNING";
            case LogLevel::L_ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }
};


#define LOGV(TAG, ...) Logger::log(Logger::LogLevel::L_VERBOSE, TAG, __VA_ARGS__)
#define LOGD(TAG, ...) Logger::log(Logger::LogLevel::L_DEBUG, TAG, __VA_ARGS__)
#define LOGI(TAG, ...) Logger::log(Logger::LogLevel::L_INFO, TAG, __VA_ARGS__)
#define LOGW(TAG, ...) Logger::log(Logger::LogLevel::L_WARNING, TAG, __VA_ARGS__)
#define LOGE(TAG, ...) Logger::log(Logger::LogLevel::L_ERROR, TAG, __VA_ARGS__)

#endif // LOGGER_HPP
