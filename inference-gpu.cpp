#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include <functional>
#include <unordered_map>

namespace fs = std::filesystem;

const int BATCH_SIZE = 17;
const int PATCH_SIZE = 512;
const int Y_OFFSET = 224;
const int X_STRIDE = 480;
const int OUT_WIDTH = 1536;
const int OUT_HEIGHT = 512;
const int RESIZE_WIDTH = 256;
const int RESIZE_HEIGHT = 256;
const int CROP_WIDTH = 256;
const int CROP_HEIGHT = 256;
const size_t BYTES_PER_OUT_PATCH = OUT_WIDTH * OUT_HEIGHT * 3;

using ResultCallback = std::function<void(unsigned char* out_buffer, int frame_id)>;

struct PerformanceMetrics {
    std::vector<double> preprocessing_times;
    std::vector<double> batch_prep_times;
    std::vector<double> gpu_times;
    std::vector<double> postprocessing_times;
    std::mutex mtx;

    void addPreprocessingTime(double t) { std::lock_guard<std::mutex> l(mtx); preprocessing_times.push_back(t); }
    void addBatchPrepTime(double t) { std::lock_guard<std::mutex> l(mtx); batch_prep_times.push_back(t); }
    void addGpuTime(double t) { std::lock_guard<std::mutex> l(mtx); gpu_times.push_back(t); }
    void addPostprocessingTime(double t) { std::lock_guard<std::mutex> l(mtx); postprocessing_times.push_back(t); }

    void printRollingAverage(int window = 10) {
        std::lock_guard<std::mutex> l(mtx);

        size_t p_size = postprocessing_times.size();
        if (p_size == 0 || p_size % window != 0) return;

        auto calc_last_n_avg = [](const std::vector<double>& vec, int n) -> double {
            if (vec.size() < n) return 0.0;
            double sum = std::accumulate(vec.end() - n, vec.end(), 0.0);
            return sum / n;
        };

        double avg_prep = calc_last_n_avg(preprocessing_times, window) / BATCH_SIZE;
        double avg_batch = calc_last_n_avg(batch_prep_times, window) / BATCH_SIZE;
        double avg_gpu = calc_last_n_avg(gpu_times, window) / BATCH_SIZE;
        double avg_post = calc_last_n_avg(postprocessing_times, window) / BATCH_SIZE;

        fmt::print("[MONITOR] Frame {}-{} | Avg patch latency -> CPU: {:.2f}ms | DMA: {:.2f}ms | GPU: {:.2f}ms | Out: {:.2f}ms\n",
            p_size - window + 1, p_size, avg_prep, avg_batch, avg_gpu, avg_post);
    }
};

struct RawImageTask {
    std::vector<cv::Mat> patches;
    int frame_id;
};

struct BatchData {
    std::shared_ptr<float> pinned_blob;
    std::vector<cv::Mat> original_patches;
    int frame_id;
};

struct InferenceResult {
    std::shared_ptr<BatchData> batch_info;
    std::vector<float> scores;
    std::vector<float> maps;
    int map_h, map_w;
};

struct CudaPinnedDeleter {
    void operator()(float* ptr) const {
        if (ptr) cudaFreeHost(ptr);
    }
};

template<typename T>
class BoundedThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    size_t max_size_;
    bool stopped_ = false;

public:
    explicit BoundedThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [this]() { return queue_.size() < max_size_ || stopped_; });
        if (stopped_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_ || stopped_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
	}

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [this]() { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }
};

class AsyncAnomalyDetector {
private:
    std::atomic<bool> is_running_{ true };
	std::atomic<int64_t> dropped_frames_{ 0 };
    ResultCallback on_result_ready_;
    PerformanceMetrics& metrics_;

    BoundedThreadSafeQueue<RawImageTask> q_raw_{ 20 };
    BoundedThreadSafeQueue<std::shared_ptr<BatchData>> q_prep_{ 5 };
    BoundedThreadSafeQueue<std::shared_ptr<InferenceResult>> q_inf_{ 10 };

    std::thread t_prep_, t_inf_, t_post_;

    Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "AsyncInference" };
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> in_names_, out_names_;
    std::vector<int64_t> in_shape_;

    void preprocessingWorker() {
        while (is_running_) {
            try {
                RawImageTask task;
                if (!q_raw_.pop(task)) break;

                auto batch = std::make_shared<BatchData>();
                batch->frame_id = task.frame_id;
                batch->original_patches = task.patches;

                std::vector<cv::Mat> processed(BATCH_SIZE);

                auto start_prep = std::chrono::high_resolution_clock::now();
                const cv::Size target_size(RESIZE_WIDTH, RESIZE_HEIGHT);
                const cv::Rect center_crop_roi((RESIZE_WIDTH - CROP_WIDTH) / 2, (RESIZE_HEIGHT - CROP_HEIGHT) / 2, CROP_WIDTH, CROP_HEIGHT);
                cv::parallel_for_(cv::Range(0, BATCH_SIZE), [&](const cv::Range& r) {
                    for (int i = r.start; i < r.end; i++) {
                        cv::Mat resized, rgb;
                        cv::resize(task.patches[i], resized, target_size);
                        cv::Mat cropped = resized(center_crop_roi);

                        cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
                        rgb.convertTo(processed[i], CV_32F, 1.0 / 255.0);
                    }
                });
                auto end_prep = std::chrono::high_resolution_clock::now();
                metrics_.addPreprocessingTime(std::chrono::duration<double, std::milli>(end_prep - start_prep).count());

                auto start_batch = std::chrono::high_resolution_clock::now();
                cv::Mat blob = cv::dnn::blobFromImages(processed, 1.0, cv::Size(CROP_WIDTH, CROP_HEIGHT), cv::Scalar(), false, false, CV_32F);
                if (!blob.isContinuous()) blob = blob.clone();

                size_t bytes = BATCH_SIZE * 3 * CROP_HEIGHT * CROP_WIDTH * sizeof(float);
                float* pinned_ptr = nullptr;

                cudaError_t status = cudaMallocHost((void**)&pinned_ptr, bytes);
                if (status != cudaSuccess) {
                    throw std::runtime_error(fmt::format("cudaMallocHost failed! CUDA error: {}", (int)status));
                }

                batch->pinned_blob.reset(pinned_ptr, CudaPinnedDeleter());
                std::memcpy(batch->pinned_blob.get(), blob.ptr<float>(), bytes);

                auto end_batch = std::chrono::high_resolution_clock::now();
                metrics_.addBatchPrepTime(std::chrono::duration<double, std::milli>(end_batch - start_batch).count());

                q_prep_.push(batch);
            }
            catch (const std::exception& e) {
                fmt::print(stderr, "[EXCEPTION THREAD 1 - Preprocessing] {}\n", e.what());
            }
        }
    }

    void inferenceWorker() {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> current_input_shape = in_shape_;
        current_input_shape[0] = BATCH_SIZE;

        while (is_running_) {
            try {
                std::shared_ptr<BatchData> batch;
                if (!q_prep_.pop(batch)) break;

                Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                    mem_info,
                    batch->pinned_blob.get(),
                    BATCH_SIZE * 3 * CROP_HEIGHT * CROP_WIDTH,
                    current_input_shape.data(),
                    current_input_shape.size()
                );

                auto start_gpu = std::chrono::high_resolution_clock::now();
                auto outputs = session_->Run(Ort::RunOptions{ nullptr }, in_names_.data(), &in_tensor, 1, out_names_.data(), out_names_.size());
                auto end_gpu = std::chrono::high_resolution_clock::now();
                metrics_.addGpuTime(std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count());

                auto res = std::make_shared<InferenceResult>();
                res->batch_info = batch;

                for (size_t i = 0; i < outputs.size(); ++i) {
                    std::string node_name = out_names_[i];
                    auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();

                    if (node_name.find("score") != std::string::npos) {
                        float* raw_scores = outputs[i].GetTensorMutableData<float>();
                        res->scores.assign(raw_scores, raw_scores + BATCH_SIZE);
                    }
                    else if (node_name.find("map") != std::string::npos) {
                        float* raw_maps = outputs[i].GetTensorMutableData<float>();
                        res->map_h = shape[shape.size() - 2];
                        res->map_w = shape[shape.size() - 1];
                        size_t total_elements = outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
                        res->maps.assign(raw_maps, raw_maps + total_elements);
                    }
                }
                q_inf_.push(res);
            }
            catch (const Ort::Exception& e) {
                fmt::print(stderr, "[EXCEPTION THREAD 2 - TensorRT] Model error: {}\n", e.what());
            }
            catch (const std::exception& e) {
                fmt::print(stderr, "[EXCEPTION THREAD 2 - Inference] {}\n", e.what());
            }
        }
    }

    void postprocessingWorker() {
        while (is_running_) {
            try {
                std::shared_ptr<InferenceResult> res;
                if (!q_inf_.pop(res)) break;

                if (res->maps.empty() || res->scores.empty()) {
                    throw std::runtime_error("Impossible to find 'score' or 'map' outputs in the ONNX model.");
                }

                auto start_post = std::chrono::high_resolution_clock::now();

                unsigned char* out_buffer = new unsigned char[BATCH_SIZE * BYTES_PER_OUT_PATCH];

                const cv::Size patch_size(PATCH_SIZE, PATCH_SIZE);
                const cv::Scalar color_red(0, 0, 255);
                const cv::Scalar color_black(0, 0, 0);
                const cv::Scalar color_white(255, 255, 255);

                cv::parallel_for_(cv::Range(0, BATCH_SIZE), [&](const cv::Range& r) {

                    cv::Mat norm, color, bin;
                    std::vector<std::vector<cv::Point>> cnts;

                    for (int j = r.start; j < r.end; j++) {
                        cv::Mat orig = res->batch_info->original_patches[j];
                        float score = res->scores[j];

                        cv::Mat dest(OUT_HEIGHT, OUT_WIDTH, CV_8UC3, out_buffer + (j * BYTES_PER_OUT_PATCH));

                        cv::Mat dest_orig = dest(cv::Rect(0, 0, PATCH_SIZE, PATCH_SIZE));
                        cv::Mat dest_overlay = dest(cv::Rect(PATCH_SIZE, 0, PATCH_SIZE, PATCH_SIZE));
                        cv::Mat dest_seg = dest(cv::Rect(PATCH_SIZE * 2, 0, PATCH_SIZE, PATCH_SIZE));

                        orig.copyTo(dest_orig);
                        orig.copyTo(dest_seg);

                        cv::Mat map(res->map_h, res->map_w, CV_32F, res->maps.data() + (j * res->map_h * res->map_w));

                        cv::normalize(map, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                        cv::applyColorMap(norm, color, cv::COLORMAP_JET);
                        cv::resize(color, color, patch_size, 0, 0, cv::INTER_NEAREST);

                        cv::addWeighted(orig, 0.5, color, 0.5, 0, dest_overlay);

                        cv::threshold(norm, bin, 150, 255, cv::THRESH_BINARY);
                        cv::resize(bin, bin, patch_size, 0, 0, cv::INTER_NEAREST);

                        cnts.clear();
                        cv::findContours(bin, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        cv::drawContours(dest_seg, cnts, -1, color_red, 2);

                        std::string score_text = fmt::format("Anomaly score: {:.4f}", score);
                        cv::putText(dest_overlay, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, color_black, 3, cv::LINE_AA);
                        cv::putText(dest_overlay, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, color_white, 1, cv::LINE_AA);

                    }
                });

                auto end_post = std::chrono::high_resolution_clock::now();
                metrics_.addPostprocessingTime(std::chrono::duration<double, std::milli>(end_post - start_post).count());
                metrics_.printRollingAverage(10);

                if (on_result_ready_) on_result_ready_(out_buffer, res->batch_info->frame_id);
            }
            catch (const std::exception& e) {
                fmt::print(stderr, "[EXCEPTION THREAD 3 - Postprocessing] {}\n", e.what());
            }
        }
    }

public:
    AsyncAnomalyDetector(const std::string& model_path, const std::string& trt_cache_dir, ResultCallback callback, PerformanceMetrics& metrics)
        : on_result_ready_(callback), metrics_(metrics) {

        cv::setNumThreads(std::max(1, cv::getNumberOfCPUs() / 2));

        if (!fs::exists(trt_cache_dir)) fs::create_directories(trt_cache_dir);

        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_fp16_enable = 1;
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = trt_cache_dir.c_str();

        session_options_.AppendExecutionProvider_TensorRT(trt_options);

        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::wstring model_path_ws(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, model_path_ws.c_str(), session_options_);

        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
        input_names_str_.push_back(input_name_ptr.get());
        in_names_.push_back(input_names_str_.back().c_str());
        in_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        size_t num_output_nodes = session_->GetOutputCount();

        output_names_str_.reserve(num_output_nodes);
        out_names_.reserve(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names_str_.push_back(session_->GetOutputNameAllocated(i, allocator_).get());
        }

        for (const auto& name_str : output_names_str_) {
            out_names_.push_back(name_str.c_str());
        }

        t_prep_ = std::thread(&AsyncAnomalyDetector::preprocessingWorker, this);
        t_inf_ = std::thread(&AsyncAnomalyDetector::inferenceWorker, this);
        t_post_ = std::thread(&AsyncAnomalyDetector::postprocessingWorker, this);

        fmt::print("Async AI Core Initialized Successfully.\n");
    }

    ~AsyncAnomalyDetector() {
        is_running_ = false;
        q_raw_.stop(); q_prep_.stop(); q_inf_.stop();
        if (t_prep_.joinable()) t_prep_.join();
        if (t_inf_.joinable()) t_inf_.join();
        if (t_post_.joinable()) t_post_.join();
    }

    int64_t getDroppedFramesCount() const {
        return dropped_frames_.load();
    }

    bool pushFrame(const unsigned char* buffer, int frame_id) {
        cv::Mat full(960, 8192, CV_8UC3, (void*)buffer);
        RawImageTask task;
        task.frame_id = frame_id;

        for (int i = 0; i < BATCH_SIZE; i++) {
            task.patches.push_back(full(cv::Rect(i * X_STRIDE, Y_OFFSET, PATCH_SIZE, PATCH_SIZE)).clone());
        }
        if(!q_raw_.try_push(task)) {
            dropped_frames_++;
			fmt::print(stderr, "[WARNING]: Dropped frame {} due to full queue! Total dropped: {}\n", frame_id, dropped_frames_.load());
            return false;
        }
        return true;
    }
};


// ==============================================================================
// LOCAL TESTING SIMULATOR WITH METRICS TRACKING
// ==============================================================================
std::atomic<int> completed_frames{ 0 };

void machineProcessedDataReceiver(unsigned char* out_buffer, int frame_id) {
   
    if (frame_id == 0) {
        
        cv::Mat debug_img(BATCH_SIZE * OUT_HEIGHT, OUT_WIDTH, CV_8UC3, out_buffer);

        std::string filename = fmt::format("debug_output_frame_{}.bmp", frame_id);
        cv::imwrite(filename, debug_img);

        fmt::print("\n[DEBUG] Saved visual verification image to: {}\n", filename);
    }
    
    delete[] out_buffer;
    completed_frames++;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fmt::print(stderr, "Usage: {} <trt_cache_dir> <onnx_model_path> <test_image_path>\n", argv[0]);
        return -1;
    }

    std::string trt_cache_dir = argv[1];
    std::string model_path = argv[2];
    std::string test_image_path = argv[3];

    PerformanceMetrics metrics;

    try {
        fmt::print("STARTING LOCAL SIMULATION\n");

        AsyncAnomalyDetector detector(model_path, trt_cache_dir, machineProcessedDataReceiver, metrics);

        fmt::print("Loading test image: {}\n", test_image_path);
        cv::Mat source_image = cv::imread(test_image_path, cv::IMREAD_COLOR);

        if (source_image.empty()) {
            throw std::runtime_error("Failed to load test image.");
        }

        if (source_image.cols != 8192 || source_image.rows != 960) {
            cv::resize(source_image, source_image, cv::Size(8192, 960));
        }

        if (!source_image.isContinuous()) {
            source_image = source_image.clone();
        }

        unsigned char* fake_camera_buffer = source_image.data;

        const int TOTAL_TEST_FRAMES = 1000;
        const int CAMERA_FPS = 35;
        const int DELAY_MS = 1000 / CAMERA_FPS;

        fmt::print("Starting stress-test: {} frames at {} FPS...\n", TOTAL_TEST_FRAMES, CAMERA_FPS);
        auto start_pipeline = std::chrono::high_resolution_clock::now();
        int dropped_frames_count = 0;
        for (int frame_id = 0; frame_id < TOTAL_TEST_FRAMES; frame_id++) {
            if (!detector.pushFrame(fake_camera_buffer, frame_id)) {
                dropped_frames_count++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_MS));
        }

        while ((completed_frames + dropped_frames_count) < TOTAL_TEST_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto end_pipeline = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(end_pipeline - start_pipeline).count();

        auto calc_avg = [](const std::vector<double>& times) -> double {
            if (times.empty()) return 0.0;
            return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        };

        double avg_preprocessing_per_image = calc_avg(metrics.preprocessing_times) / BATCH_SIZE;
        double avg_batch_prep_per_image = calc_avg(metrics.batch_prep_times) / BATCH_SIZE;
        double avg_gpu_batch = calc_avg(metrics.gpu_times);
        double avg_gpu_per_image = avg_gpu_batch / BATCH_SIZE;
        double avg_postprocessing_per_image = calc_avg(metrics.postprocessing_times) / BATCH_SIZE;

        int total_images = metrics.preprocessing_times.size() * BATCH_SIZE;

        fmt::print("\n========================================\n");
        fmt::print("        DETAILED PERFORMANCE REPORT      \n");
        fmt::print("========================================\n");
        fmt::print("Batch size used: {}\n", BATCH_SIZE);
        fmt::print("Estimated total patches processed: {}\n", total_images);
        fmt::print("----------------------------------------\n");
        fmt::print("Average times per SINGLE patch (512x512):\n");
        fmt::print("  - Read & preprocess (CPU): {:.3f} ms\n", avg_preprocessing_per_image);
        fmt::print("  - Batch prep (Pinned memory): {:.3f} ms\n", avg_batch_prep_per_image);
        fmt::print("  - GPU inference          : {:.3f} ms  (Total Batch: {:.3f} ms)\n", avg_gpu_per_image, avg_gpu_batch);
        fmt::print("  - Postprocess to RAM     : {:.3f} ms\n", avg_postprocessing_per_image);
        fmt::print("----------------------------------------\n");

        double t1_cost = avg_preprocessing_per_image + avg_batch_prep_per_image;
        double t2_cost = avg_gpu_per_image;
        double t3_cost = avg_postprocessing_per_image;
        double bottleneck = std::max({ t1_cost, t2_cost, t3_cost });

        fmt::print("Total real-world test duration: {:.2f} ms\n", total_time_ms);
        fmt::print("Theoretical max throughput    : {:.2f} FPS (Limited by slowest thread: {:.3f} ms/patch)\n", 1000.0 / bottleneck, bottleneck);
        
        double actual_ms_image = total_time_ms / total_images;
        double actual_fps = 1000.0 / actual_ms_image;
        fmt::print("Actual machine throughput     : {:.2f} FPS\n", actual_fps);
        fmt::print("========================================\n");

    }
    catch (const std::exception& e) {
        fmt::print(stderr, "Fatal Error: {}\n", e.what());
        return -1;
    }

    return 0;
}