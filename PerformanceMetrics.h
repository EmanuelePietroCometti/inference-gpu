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
    std::vector<double> gpu_times;             // wall-time del blocco GPU, incl. sync/contesa (per batch)
    std::vector<double> h2d_times;             // device: span H2D (per batch, solo path GPU)
    std::vector<double> run_times;             // device: span Run (per batch, solo path GPU)
    std::vector<double> d2h_times;             // device: span D2H (per batch, solo path GPU)
    std::vector<double> postprocessing_times;  // overlay build (per batch)
    std::mutex mtx;
    int batch_size = 17;

    void addPreprocessingTime(double t) { std::lock_guard<std::mutex> l(mtx); preprocessing_times.push_back(t); }
    void addBatchPrepTime(double t)     { std::lock_guard<std::mutex> l(mtx); batch_prep_times.push_back(t); }
    void addGpuTime(double t)           { std::lock_guard<std::mutex> l(mtx); gpu_times.push_back(t); }
    void addH2DTime(double t) { std::lock_guard<std::mutex> l(mtx); h2d_times.push_back(t); }
    void addRunTime(double t) { std::lock_guard<std::mutex> l(mtx); run_times.push_back(t); }
    void addD2HTime(double t) { std::lock_guard<std::mutex> l(mtx); d2h_times.push_back(t); }
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

        double avg_prep  = calc_last_n_avg(preprocessing_times, window);
        double avg_batch = calc_last_n_avg(batch_prep_times, window);
        double avg_gpu   = calc_last_n_avg(gpu_times, window);
        double avg_h2d = calc_last_n_avg(h2d_times, window);
        double avg_run = calc_last_n_avg(run_times, window);
        double avg_d2h = calc_last_n_avg(d2h_times, window);
        double avg_post  = calc_last_n_avg(postprocessing_times, window);

        fmt::print("[MONITOR] Batch {}-{} | per-batch(ms) CPU:{:.2f} DMA:{:.2f} | GPUwall:{:.2f} = H2D:{:.3f}+Run:{:.3f}+D2H:{:.3f} | Out:{:.2f}\n",
            p_size - window + 1, p_size, avg_prep, avg_batch, avg_gpu, avg_h2d, avg_run, avg_d2h, avg_post);
    }

    void clear() {
        std::lock_guard<std::mutex> l(mtx);
        preprocessing_times.clear();
        batch_prep_times.clear();
        gpu_times.clear();
        h2d_times.clear();
        run_times.clear();
        d2h_times.clear();
        postprocessing_times.clear();
    }
};
