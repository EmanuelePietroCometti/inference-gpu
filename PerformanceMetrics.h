#pragma once

#include <fmt/core.h>
#include <array>
#include <atomic>
#include <mutex>
#include <cstddef>
#include "AsyncLogger.h"

//
// PerformanceMetrics
// Rolling-average accumulator for the four pipeline stages.
//
// Each field gets its own small FIXED-size ring buffer (capacity =
// rolling window) and its own mutex, so stages never block each other and
// there is never a reallocation after startup. The periodic summary line is
// switched from a direct, blocking fmt::print (console I/O) to Log::Info(),
// which is what AsyncLogger exists for -- see its own header: "the hot path
// (Log call) NEVER blocks on I/O".
//
struct PerformanceMetrics {
    static constexpr int kWindow = 10;

    struct RollingField {
        std::array<double, kWindow> buf{};
        size_t next = 0;
        size_t count = 0; // caps at kWindow
        std::mutex m;

        void add(double v) {
            std::lock_guard<std::mutex> l(m);
            buf[next] = v;
            next = (next + 1) % kWindow;
            if (count < kWindow) ++count;
        }
        double average() {
            std::lock_guard<std::mutex> l(m);
            double sum = 0.0;
            for (size_t i = 0; i < count; ++i) sum += buf[i];
            return count ? sum / count : 0.0;
        }
    };

    RollingField preprocessing, batchPrep, gpu, h2d, run, d2h, postprocessing;
    std::atomic<uint64_t> completedBatches_{ 0 };
    int batch_size = 17;

    void addPreprocessingTime(double t) { preprocessing.add(t); }
    void addBatchPrepTime(double t) { batchPrep.add(t); }
    void addGpuTime(double t) { gpu.add(t); }
    void addH2DTime(double t) { h2d.add(t); }
    void addRunTime(double t) { run.add(t); }
    void addD2HTime(double t) { d2h.add(t); }
    void addPostprocessingTime(double t) { postprocessing.add(t); }

    // Call ONCE per completed batch (unchanged call site: postprocessingWorker(),
    // right after addPostprocessingTime()).
    void printRollingAverage(int window = kWindow) {
        const uint64_t n = ++completedBatches_;
        if (window <= 0 || n % (uint64_t)window != 0) return;

        // Log::Info formats onto the CALLER's stack and enqueues onto
        // AsyncLogger's bounded ring in a few tens of nanoseconds -- unlike
        // fmt::print, it never touches the console synchronously and never
        // blocks this (hot-path) thread.
        Log::Info("[MONITOR] Batch {}-{} | per-batch(ms) CPU:{:.2f} DMA:{:.2f} | "
            "GPUwall:{:.2f} = H2D:{:.3f}+Run:{:.3f}+D2H:{:.3f} | Out:{:.2f}",
            n - (uint64_t)window + 1, n,
            preprocessing.average(), batchPrep.average(), gpu.average(),
            h2d.average(), run.average(), d2h.average(), postprocessing.average());
    }

    void clear() { completedBatches_ = 0; } // ring buffers self-overwrite, nothing to free
};