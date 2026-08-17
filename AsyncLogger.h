#pragma once

#include <windows.h>
#include <fmt/format.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>

//
// AsyncLogger
// Dedicated logging subsystem for the real-time inference pipeline.
//
// Design goals (hard real-time producers):
//  - The hot path (Log call) NEVER blocks on I/O and NEVER allocates on the
//    heap: the message is formatted into a stack buffer, then memcpy'd into a
//    pre-allocated fixed-size slot of a bounded ring buffer under a mutex held
//    for a few tens of nanoseconds.
//  - If the queue is full the message is DROPPED (and counted), it never
//    stalls an inference thread: losing a log line is acceptable, missing a
//    frame deadline is not.
//  - A single background consumer thread (lowest priority, pinned to core 0,
//    away from the inference cores) drains the queue periodically and performs
//    the actual console I/O.
//
// Usage:
//   AsyncLogger::Instance().Start();          // once, at process startup
//   Log::Info("Worker {} ready", id);         // from any thread
//   Log::Error("failure: {}", what);
//   AsyncLogger::Instance().Shutdown();       // once, at exit (drains queue)
//

enum class LogLevel : uint8_t { Info = 0, Warning = 1, Error = 2 };

class AsyncLogger {
public:
    // Max characters per message (longer messages are truncated, not dropped)
    static constexpr size_t kMaxMessageChars = 480;
    // Ring capacity. Power of two, bounded: total footprint ~2 MB, allocated once.
    static constexpr size_t kQueueCapacity = 4096;
    // The consumer flushes at least this often even without a notify
    static constexpr DWORD kFlushIntervalMs = 200;
    // Producers notify the consumer only when the queue crosses this fill level
    // (or on Error), so the common case is one flush per interval, not per message
    static constexpr size_t kNotifyThreshold = kQueueCapacity / 4;

    static AsyncLogger& Instance();

    void Start();
    void Shutdown();

    template <typename... Args>
    void Log(LogLevel level, fmt::format_string<Args...> fmtStr, Args&&... args)
    {
        // Format on the caller's stack: fmt::format_to_n never allocates and
        // truncates safely at kMaxMessageChars
        char local[kMaxMessageChars];
        const auto out = fmt::format_to_n(local, kMaxMessageChars, fmtStr, std::forward<Args>(args)...);
        // (std::min) parenthesized: immune to the windows.h min/max macros
        const size_t written = (std::min)(static_cast<size_t>(out.size), kMaxMessageChars);
        Enqueue(level, local, static_cast<uint16_t>(written));
    }

    AsyncLogger(const AsyncLogger&) = delete;
    AsyncLogger& operator=(const AsyncLogger&) = delete;

private:
    AsyncLogger() = default;
    ~AsyncLogger();

    struct Slot {
        SYSTEMTIME ts;
        LogLevel   level;
        uint16_t   length;
        char       text[kMaxMessageChars];
    };

    void Enqueue(LogLevel level, const char* msg, uint16_t length);
    void ConsumerLoop();
    // Drains and prints every queued slot. Returns the number of slots printed.
    size_t DrainQueue();

    // Bounded ring buffer, allocated once with the singleton (no heap churn)
    Slot ring_[kQueueCapacity];
    size_t head_ = 0;   // next write position
    size_t tail_ = 0;   // next read position
    size_t count_ = 0;  // slots currently queued

    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread consumer_;
    std::atomic<bool> running_{ false };
    std::atomic<uint64_t> dropped_{ 0 };
};

// Convenience free functions: every subsystem logs through these instead of
// printing to the console directly
namespace Log {

template <typename... Args>
inline void Info(fmt::format_string<Args...> fmtStr, Args&&... args)
{
    AsyncLogger::Instance().Log(LogLevel::Info, fmtStr, std::forward<Args>(args)...);
}

template <typename... Args>
inline void Warning(fmt::format_string<Args...> fmtStr, Args&&... args)
{
    AsyncLogger::Instance().Log(LogLevel::Warning, fmtStr, std::forward<Args>(args)...);
}

template <typename... Args>
inline void Error(fmt::format_string<Args...> fmtStr, Args&&... args)
{
    AsyncLogger::Instance().Log(LogLevel::Error, fmtStr, std::forward<Args>(args)...);
}

} // namespace Log
