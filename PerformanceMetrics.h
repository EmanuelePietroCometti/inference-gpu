#pragma once

#include <fmt/core.h>
#include <vector>
#include <mutex>
#include <numeric>
#include <cstddef>

//
// PerformanceMetrics
// Thread-safe accumulator for the four pipeline stages, preserved verbatim
// from the original inference-gpu monolith. Times are per-batch; per-patch
// figures are derived by dividing by the batch size at report time.
//
struct PerformanceMetrics {
    std::vector<double> preprocessing_times;   // CPU resize + normalize (per batch)
    std::vector<double> batch_prep_times;      // pinned host staging (per batch)
    std::vector<double> gpu_times;             // H2D + Run + D2H (per batch)
    std::vector<double> postprocessing_times;  // overlay build (per batch)
    std::mutex mtx;
    int batch_size = 17;

    void addPreprocessingTime(double t) { std::lock_guard<std::mutex> l(mtx); preprocessing_times.push_back(t); }
    void addBatchPrepTime(double t)     { std::lock_guard<std::mutex> l(mtx); batch_prep_times.push_back(t); }
    void addGpuTime(double t)           { std::lock_guard<std::mutex> l(mtx); gpu_times.push_back(t); }
    void addPostprocessingTime(double t){ std::lock_guard<std::mutex> l(mtx); postprocessing_times.push_back(t); }

    void printRollingAverage(int window = 10) {
        std::lock_guard<std::mutex> l(mtx);

        size_t p_size = postprocessing_times.size();
        if (p_size == 0 || p_size % window != 0) return;

        auto calc_last_n_avg = [](const std::vector<double>& vec, int n) -> double {
            if ((int)vec.size() < n) return 0.0;
            double sum = std::accumulate(vec.end() - n, vec.end(), 0.0);
            return sum / n;
        };

        double avg_prep  = calc_last_n_avg(preprocessing_times, window) / batch_size;
        double avg_batch = calc_last_n_avg(batch_prep_times, window) / batch_size;
        double avg_gpu   = calc_last_n_avg(gpu_times, window) / batch_size;
        double avg_post  = calc_last_n_avg(postprocessing_times, window) / batch_size;

        fmt::print("[MONITOR] Batch {}-{} | Avg patch latency -> CPU: {:.2f}ms | DMA: {:.2f}ms | GPU: {:.2f}ms | Out: {:.2f}ms\n",
            p_size - window + 1, p_size, avg_prep, avg_batch, avg_gpu, avg_post);
    }

    void clear() {
        std::lock_guard<std::mutex> l(mtx);
        preprocessing_times.clear();
        batch_prep_times.clear();
        gpu_times.clear();
        postprocessing_times.clear();
    }
};
