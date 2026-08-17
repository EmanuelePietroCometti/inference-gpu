#include "AsyncLogger.h"
#include "RealTimeConfig.h"

#include <fmt/core.h>
#include <fmt/color.h>
#include <chrono>
#include <cstdio>
#include <cstring>

AsyncLogger& AsyncLogger::Instance()
{
    static AsyncLogger instance;
    return instance;
}

AsyncLogger::~AsyncLogger()
{
    Shutdown();
}

// fmt emette i colori come sequenze ANSI (\x1b[38;2;...m): il conhost classico
// non le interpreta finche' non si attiva ENABLE_VIRTUAL_TERMINAL_PROCESSING,
// e le stampa letteralmente come "<-[38;2;...m". Se l'attivazione fallisce
// (console troppo vecchia) i colori vengono semplicemente disabilitati.
static bool s_ansiColorSupported = false;

static void EnableVTProcessing()
{
    s_ansiColorSupported = true;
    for (DWORD stdHandle : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h = GetStdHandle(stdHandle);
        DWORD mode = 0;
        if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) {
            continue;   // stream redirezionato su file/pipe: non e' una console
        }
        if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            s_ansiColorSupported = false;
        }
    }
}

void AsyncLogger::Start()
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return; // already started
    }
    EnableVTProcessing();
    consumer_ = std::thread(&AsyncLogger::ConsumerLoop, this);
}

void AsyncLogger::Shutdown()
{
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return; // never started or already stopped
    }
    cv_.notify_one();
    if (consumer_.joinable()) {
        consumer_.join();
    }
    // Final drain: nothing queued after Shutdown returns is lost
    DrainQueue();
    std::fflush(stdout);
}

void AsyncLogger::Enqueue(LogLevel level, const char* msg, uint16_t length)
{
    // Timestamp taken at enqueue time so the printed time reflects the event,
    // not the flush. GetLocalTime reads shared kernel memory: no syscall cost.
    SYSTEMTIME ts;
    GetLocalTime(&ts);

    bool shouldNotify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == kQueueCapacity) {
            // Queue full: drop instead of blocking the real-time producer
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Slot& slot = ring_[head_];
        slot.ts = ts;
        slot.level = level;
        slot.length = length;
        std::memcpy(slot.text, msg, length);

        head_ = (head_ + 1) % kQueueCapacity;
        ++count_;

        // Wake the consumer early only when the queue is filling up or the
        // message is an error; routine traffic waits for the periodic flush
        shouldNotify = (count_ >= kNotifyThreshold) || (level == LogLevel::Error);
    }
    if (shouldNotify) {
        cv_.notify_one();
    }
}

size_t AsyncLogger::DrainQueue()
{
    size_t printed = 0;
    for (;;) {
        Slot local;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (count_ == 0) break;
            // Copy ONE slot out under the lock (~half a microsecond), then
            // print with the lock released so producers are never stalled by I/O
            local = ring_[tail_];
            tail_ = (tail_ + 1) % kQueueCapacity;
            --count_;
        }

        const fmt::string_view text(local.text, local.length);
        switch (local.level) {
        case LogLevel::Error:
            fmt::print(stderr, s_ansiColorSupported
                ? fg(fmt::color::red) | fmt::emphasis::bold : fmt::text_style{},
                "{:02}:{:02}:{:02}.{:03} [ERR ] {}\n",
                local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
            break;
        case LogLevel::Warning:
            fmt::print(s_ansiColorSupported
                ? fg(fmt::color::yellow) : fmt::text_style{},
                "{:02}:{:02}:{:02}.{:03} [WARN] {}\n",
                local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
            break;
        default:
            fmt::print("{:02}:{:02}:{:02}.{:03} [INFO] {}\n",
                local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
            break;
        }
        ++printed;
    }

    const uint64_t dropped = dropped_.exchange(0, std::memory_order_relaxed);
    if (dropped > 0) {
        fmt::print(stderr, s_ansiColorSupported
            ? fg(fmt::color::red) | fmt::emphasis::bold : fmt::text_style{},
            "[LOGGER] {} messages dropped (queue full)\n", dropped);
    }
    return printed;
}

void AsyncLogger::ConsumerLoop()
{
    // Lowest priority, pinned to core 0: console I/O must never steal CPU time
    // from the TIME_CRITICAL inference threads on the dedicated cores
    RT::ConfigureBackgroundThread();

    while (running_.load(std::memory_order_relaxed)) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(kFlushIntervalMs),
                [this] { return count_ >= kNotifyThreshold || !running_.load(std::memory_order_relaxed); });
        }
        DrainQueue();
    }
    // Shutdown drains once more after the join, nothing is lost here
}
