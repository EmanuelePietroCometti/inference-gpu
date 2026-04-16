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


namespace fs = std::filesystem;

struct PerformanceMetrics {
    std::vector<double> preprocessing_times;
    std::vector<double> batch_prep_times;

    std::vector<double> gpu_times;

    std::vector<double> postprocessing_times;
};

struct BatchData {
    cv::Mat blob;
    std::vector<cv::Mat> original_images;
    std::vector<std::string> filenames;
    int valid_images;
};

struct InferenceResult {
    std::shared_ptr<BatchData> batch_info;
    std::vector<float> pred_scores;
    std::vector<float> anomaly_maps;
    int map_h;
    int map_w;
    std::vector<uint8_t> pred_masks;
    int mask_h;
    int mask_w;
    bool has_masks = false;
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

// PIPELINE CONFIGURATION
const int BATCH_SIZE = 17;
const int RESIZE_WIDTH = 256;
const int RESIZE_HEIGHT = 256;
const int CROP_WIDTH = 256;
const int CROP_HEIGHT = 256;

std::atomic<bool> g_is_running{ true };

cv::Mat preprocessImage(const cv::Mat& src_image) {
    cv::Mat resized_image;
    cv::resize(src_image, resized_image, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), 0, 0, cv::INTER_LINEAR);

    int top = (RESIZE_HEIGHT - CROP_HEIGHT) / 2;
    int left = (RESIZE_WIDTH - CROP_WIDTH) / 2;
    cv::Rect roi(left, top, CROP_WIDTH, CROP_HEIGHT);
    cv::Mat cropped_image = resized_image(roi).clone();

    cv::Mat rgb_float;
    cv::cvtColor(cropped_image, rgb_float, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb_float.convertTo(float_img, CV_32F, 1.0 / 255.0);

    return float_img;
}

// Thread 1: preprocessing
void preprocessingThread(
    const std::string& input_dir, 
    BoundedThreadSafeQueue<std::shared_ptr<BatchData>>& queue_out,
    PerformanceMetrics& metrics
) {
    std::vector<std::string> batch_filepaths;
    std::vector<std::string> batch_filenames;
    batch_filepaths.reserve(BATCH_SIZE);
    batch_filenames.reserve(BATCH_SIZE);

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!g_is_running) break;

        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".bmp" && ext != ".BMP") continue;

            batch_filepaths.push_back(entry.path().string());
            batch_filenames.push_back(entry.path().filename().string());

            if (batch_filepaths.size() == BATCH_SIZE) {
                auto current_batch = std::make_shared<BatchData>();

                current_batch->original_images.resize(BATCH_SIZE);
                std::vector<cv::Mat> temp_preprocessed_images(BATCH_SIZE);

                current_batch->filenames = batch_filenames;
                current_batch->valid_images = BATCH_SIZE;

                auto start_prep = std::chrono::high_resolution_clock::now();

                cv::parallel_for_(cv::Range(0, BATCH_SIZE), [&](const cv::Range& range) {
                    for (int i = range.start; i < range.end; i++) {
                        cv::Mat image = cv::imread(batch_filepaths[i], cv::IMREAD_COLOR);

                        if (image.empty()) {
                            fmt::print(stderr, "Warning: Corrupted image replaced with dummy: {}\n", batch_filepaths[i]);
                            image = cv::Mat::zeros(cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), CV_8UC3);
                        }

                        current_batch->original_images[i] = image.clone();
                        temp_preprocessed_images[i] = preprocessImage(image);
                    }
                    });

                auto end_prep = std::chrono::high_resolution_clock::now();
                metrics.preprocessing_times.push_back(std::chrono::duration<double, std::milli>(end_prep - start_prep).count());

                auto start_batch = std::chrono::high_resolution_clock::now();
                current_batch->blob = cv::dnn::blobFromImages(
                    temp_preprocessed_images, 1.0, cv::Size(CROP_WIDTH, CROP_HEIGHT), cv::Scalar(), false, false, CV_32F
                );

                if (!current_batch->blob.isContinuous()) {
                    current_batch->blob = current_batch->blob.clone();
                }
                auto end_batch = std::chrono::high_resolution_clock::now();
                metrics.batch_prep_times.push_back(std::chrono::duration<double, std::milli>(end_batch - start_batch).count());

                if (!queue_out.push(current_batch)) break;

                batch_filepaths.clear();
                batch_filenames.clear();
            }
        }
    }

    if (!batch_filepaths.empty()) {
        int remaining_size = batch_filepaths.size();
        auto current_batch = std::make_shared<BatchData>();

        current_batch->original_images.resize(remaining_size);
        current_batch->filenames = batch_filenames;
        current_batch->valid_images = remaining_size;

        std::vector<cv::Mat> temp_preprocessed_images(remaining_size);

        auto start_prep = std::chrono::high_resolution_clock::now();
        cv::parallel_for_(cv::Range(0, remaining_size), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++) {
                cv::Mat image = cv::imread(batch_filepaths[i], cv::IMREAD_COLOR_RGB);
                if (image.empty()) image = cv::Mat::zeros(cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), CV_8UC3);
                current_batch->original_images[i] = image.clone();
                temp_preprocessed_images[i] = preprocessImage(image);
            }
            });
        auto end_prep = std::chrono::high_resolution_clock::now();
        metrics.preprocessing_times.push_back(std::chrono::duration<double, std::milli>(end_prep - start_prep).count());

        while (temp_preprocessed_images.size() < BATCH_SIZE) {
            temp_preprocessed_images.push_back(temp_preprocessed_images.back().clone());
        }

        auto start_batch = std::chrono::high_resolution_clock::now();
        current_batch->blob = cv::dnn::blobFromImages(
            temp_preprocessed_images, 1.0, cv::Size(CROP_WIDTH, CROP_HEIGHT), cv::Scalar(), false, false, CV_32F
        );
        if (!current_batch->blob.isContinuous()) current_batch->blob = current_batch->blob.clone();
        auto end_batch = std::chrono::high_resolution_clock::now();
        metrics.batch_prep_times.push_back(std::chrono::duration<double, std::milli>(end_batch - start_batch).count());

        queue_out.push(current_batch);
    }
}

// Thread 2: inference
void inferenceThread(Ort::Session& session,
    std::vector<const char*> input_names,
    std::vector<const char*> output_names,
    std::vector<int64_t> base_input_shape,
    BoundedThreadSafeQueue<std::shared_ptr<BatchData>>& queue_in,
    BoundedThreadSafeQueue<std::shared_ptr<InferenceResult>>& queue_out,
    PerformanceMetrics& metrics
) {

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> current_input_shape = base_input_shape;
    current_input_shape[0] = BATCH_SIZE;

    while (g_is_running) {
        std::shared_ptr<BatchData> batch;

        if (!queue_in.pop(batch)) break;

        // ZERO-COPY TENSOR CREATION
        // Directly use the memory address of the pre-built blob.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            reinterpret_cast<float*>(batch->blob.data),
            batch->blob.total(),
            current_input_shape.data(),
            current_input_shape.size()
        );

        auto start_gpu = std::chrono::high_resolution_clock::now();
        // Execute GPU Inference
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            output_names.size()
        );
        auto end_gpu = std::chrono::high_resolution_clock::now();
        metrics.gpu_times.push_back(std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count());

        auto result = std::make_shared<InferenceResult>();
        result->batch_info = batch;

        float* pred_scores = nullptr;
        float* anomaly_maps = nullptr;

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            std::string node_name = output_names[i];
            auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

            if (node_name.find("score") != std::string::npos) {
                pred_scores = output_tensors[i].GetTensorMutableData<float>();
                result->pred_scores.assign(pred_scores, pred_scores + BATCH_SIZE);
            }
            else if (node_name.find("map") != std::string::npos) {
                anomaly_maps = output_tensors[i].GetTensorMutableData<float>();
                result->map_h = shape[shape.size() - 2];
                result->map_w = shape[shape.size() - 1];
                
                size_t total_elements = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
                result->anomaly_maps.assign(anomaly_maps, anomaly_maps + total_elements);
            }
            else if (node_name.find("mask") != std::string::npos) {
                bool* pred_masks = output_tensors[i].GetTensorMutableData<bool>();
                result->mask_h = shape[shape.size() - 2];
                result->mask_w = shape[shape.size() - 1];

                size_t total_elements = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();

                result->pred_masks.resize(total_elements);
                for (size_t k = 0; k < total_elements; ++k) {
                    result->pred_masks[k] = pred_masks[k] ? 255 : 0;
                }
                result->has_masks = true;
            }
        }
        if (!queue_out.push(result)) break;
    }
}

// Thread 3: postprocessing
void postprocessingThread(
    const std::string& output_dir,
    BoundedThreadSafeQueue<std::shared_ptr<InferenceResult>>& queue_in,
    PerformanceMetrics& metrics
) {
    while (g_is_running) {
        std::shared_ptr<InferenceResult> result;
        if (!queue_in.pop(result)) break;

        int valid_image_count = result->batch_info->valid_images;
        auto start_post = std::chrono::high_resolution_clock::now();

        cv::parallel_for_(cv::Range(0, valid_image_count), [&](const cv::Range& range) {
            for (int j = range.start; j < range.end; j++) {

                cv::Mat original_bgr = result->batch_info->original_images[j];
                float score = result->pred_scores[j];
                std::string filename = result->batch_info->filenames[j];

                size_t elements_per_item = result->anomaly_maps.size() / BATCH_SIZE;
                int offset_map = j * elements_per_item;
                cv::Mat map_anomaly(result->map_h, result->map_w, CV_32F, result->anomaly_maps.data() + offset_map);

                if (map_anomaly.empty() || map_anomaly.dims != 2 || map_anomaly.size().area() == 0) continue;

                cv::Mat heatmap_norm, heatmap_colored, overlay_img;
                cv::normalize(map_anomaly, heatmap_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::applyColorMap(heatmap_norm, heatmap_colored, cv::COLORMAP_JET);

                if (heatmap_colored.size() != original_bgr.size()) {
                    cv::resize(heatmap_colored, heatmap_colored, original_bgr.size(), 0, 0, cv::INTER_NEAREST);
                }
                cv::addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0, overlay_img);

                cv::Mat segmentation_img = original_bgr.clone();
                cv::Mat binary_mask;
                std::vector<std::vector<cv::Point>> contours;

                if (result->has_masks) {
                    size_t mask_elements = result->pred_masks.size() / BATCH_SIZE;
                    cv::Mat raw_mask(result->mask_h, result->mask_w, CV_8U, result->pred_masks.data() + (j * mask_elements));
                    cv::resize(raw_mask, binary_mask, original_bgr.size(), 0, 0, cv::INTER_NEAREST);
                }
                else {
                    cv::threshold(heatmap_norm, binary_mask, 150, 255, cv::THRESH_BINARY);
                    cv::resize(binary_mask, binary_mask, original_bgr.size(), 0, 0, cv::INTER_NEAREST);
                }

                cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(segmentation_img, contours, -1, cv::Scalar(0, 0, 255), 2);

                std::string score_text = fmt::format("Anomaly score: {:.4f}", score);
                cv::putText(overlay_img, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::putText(overlay_img, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                cv::Mat combined_result;
                std::vector<cv::Mat> images_to_concat = { original_bgr, overlay_img, segmentation_img };
                cv::hconcat(images_to_concat, combined_result);

                std::string out_path = (fs::path(output_dir) / filename).string();
                cv::imwrite(out_path, combined_result);
            }
            });

        auto end_post = std::chrono::high_resolution_clock::now();
        metrics.postprocessing_times.push_back(std::chrono::duration<double, std::milli>(end_post - start_post).count());
    }
}

/*
Parameters:
    - argv[1]: Path to the TensorRT engine cache directory (e.g., "D:\\tensorrt_cache")
    - argv[2]: Path to the ONNX model (e.g., "D:\\emanuele\\Code\\inference-gpu\\model\\20260407_173343_EfficientAd_wide_resnet50_2.onnx")
    - argv[3]: Path to the input images directory (e.g., "D:\\emanuele\\Code\\inference-gpu\\images")
    - argv[4]: Path to the output directory (e.g., "D:\\emanuele\\Code\\inference-gpu\\output")
*/
int main(int argc, char* argv[]) {
    if (argc < 5) {
        fmt::print(stderr, "Usage: {} <trt_cache_dir> <onnx_model_path> <input_images_dir> <output_dir>\n", argv[0]);
        return -1;
    }

    fs::path trt_cache_path(argv[1]);
    fs::path model_file_path(argv[2]);
    fs::path input_dir_path(argv[3]);
    fs::path output_dir_path(argv[4]);

    if (!fs::exists(model_file_path) || !fs::is_regular_file(model_file_path)) {
        fmt::print(stderr, "Error: Model file does not exist or is not a file: {}\n", model_file_path.string());
        return -1;
    }
    if (!fs::exists(input_dir_path) || !fs::is_directory(input_dir_path) || fs::is_empty(input_dir_path)) {
        fmt::print(stderr, "Error: Input directory is invalid or empty: {}\n", input_dir_path.string());
        return -1;
    }

    try {
        if (!fs::exists(trt_cache_path)) fs::create_directories(trt_cache_path);
        if (!fs::exists(output_dir_path)) fs::create_directories(output_dir_path);
    }
    catch (const fs::filesystem_error& e) {
        fmt::print(stderr, "Filesystem error while creating directories: {}\n", e.what());
        return -1;
    }

    std::wstring model_path_ws = model_file_path.wstring();
    std::string trt_cache_str = trt_cache_path.string();
    std::string output_dir_str = output_dir_path.string();
    std::string input_dir_str = input_dir_path.string();

    // ONNX Runtime inizialization
    Ort::Env inferenceEnv(ORT_LOGGING_LEVEL_WARNING, "GpuInference");
    Ort::SessionOptions session_options;

    // Enable TensorRT
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_fp16_enable = 1;
    trt_options.trt_engine_cache_enable = 1;
    trt_options.trt_engine_cache_path = trt_cache_str.c_str();

    try {
        session_options.AppendExecutionProvider_TensorRT(trt_options);
    }
    catch (const std::exception& e) {
        fmt::print(stderr, "TensorRT not supported or DLL missing: {}\n", e.what());
        return -1;
    }

    // Enable CUDA as fallback
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 1;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

    try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    catch (const std::exception& e) {
        fmt::print(stderr, "CUDA not supported or DLL missing: {}\n", e.what());
        return -1;
    }

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.AddConfigEntry("session.free_memory_dimension_overrides", "1");
    session_options.AddConfigEntry("session.memory.enable_memory_arena_shrinkage", "gpu:0");
    session_options.AddConfigEntry("session.use_env_allocators", "0");

    try {
        const wchar_t* model_path = model_path_ws.c_str();
        Ort::Session session(inferenceEnv, model_path, session_options);
        fmt::print("Model loaded successfully on GPU!\n");

        Ort::AllocatorWithDefaultOptions allocator;

        // Extract input node information
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        std::string input_name = input_name_ptr.get();
        std::vector<int64_t> base_input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // Extract output node information dynamically
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<std::string> output_names_str;
        std::vector<const char*> output_names;
        output_names_str.reserve(num_output_nodes);
        output_names.reserve(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_names_str.push_back(output_name_ptr.get());
            output_names.push_back(output_names_str.back().c_str());
        }

        std::vector<const char*> input_names = { input_name.c_str() };

        // GPU warm-up
        fmt::print("Executing GPU Warm-up...\n");
        const int WARMUP_ITERATIONS = 3;
        std::vector<int64_t> warmup_shape = base_input_shape;
        warmup_shape[0] = BATCH_SIZE;

        size_t warmup_tensor_size = BATCH_SIZE * 3 * CROP_HEIGHT * CROP_WIDTH;
        std::vector<float> dummy_data(warmup_tensor_size, 0.0f);
        auto warmup_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value dummy_tensor = Ort::Value::CreateTensor<float>(
            warmup_memory_info, dummy_data.data(), dummy_data.size(), warmup_shape.data(), warmup_shape.size()
        );

        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &dummy_tensor, 1, output_names.data(), output_names.size());
        }
        fmt::print("GPU Warm-up completed. Tensor Cores are ready.\n\n");

        // Multithreading pipiline
        fmt::print("Starting Producer-Consumer Pipeline...\n");
        auto start_pipeline = std::chrono::high_resolution_clock::now();

        // Initialize queues
        BoundedThreadSafeQueue<std::shared_ptr<BatchData>> q_preprocessed(5);
        BoundedThreadSafeQueue<std::shared_ptr<InferenceResult>> q_inferred(10);
        PerformanceMetrics metrics;

        // Start threads
        std::thread t1(preprocessingThread, input_dir_str, std::ref(q_preprocessed), std::ref(metrics));
        std::thread t2(inferenceThread, std::ref(session), input_names, output_names, base_input_shape, std::ref(q_preprocessed), std::ref(q_inferred), std::ref(metrics));
        std::thread t3(postprocessingThread, output_dir_str, std::ref(q_inferred), std::ref(metrics));

        
        t1.join();
        fmt::print("Thread 1 (Preprocessing) finished reading all files.\n");

        q_preprocessed.stop();
        t2.join();
        fmt::print("Thread 2 (GPU Inference) finished processing all batches.\n");

        q_inferred.stop();
        t3.join();
        fmt::print("Thread 3 (Postprocessing) finished saving all results.\n");

        auto end_pipeline = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(end_pipeline - start_pipeline).count();

        // Avg calculation lambda function
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
        fmt::print("Estimated total images processed: {}\n", total_images);
        fmt::print("----------------------------------------\n");
        fmt::print("Amortized average times per SINGLE image:\n");
        fmt::print("  - Read & preprocess (CPU): {:.3f} ms\n", avg_preprocessing_per_image);
        fmt::print("  - Batch prep (CPU memory): {:.3f} ms\n", avg_batch_prep_per_image);
        fmt::print("  - GPU inference          : {:.3f} ms  (Total Batch: {:.3f} ms)\n", avg_gpu_per_image, avg_gpu_batch);
        fmt::print("  - Postprocess & write    : {:.3f} ms\n", avg_postprocessing_per_image);
        fmt::print("----------------------------------------\n");

        double t1_cost = avg_preprocessing_per_image + avg_batch_prep_per_image;
        double t2_cost = avg_gpu_per_image;
        double t3_cost = avg_postprocessing_per_image;
        double bottleneck = std::max({ t1_cost, t2_cost, t3_cost });

        fmt::print("Total real-world pipeline time: {:.2f} ms\n", total_time_ms);
        fmt::print("Theoretical max throughput    : {:.2f} FPS (Limited by slowest thread: {:.3f} ms/img)\n", 1000.0 / bottleneck, bottleneck);
        fmt::print("Actual measured throughput    : {:.2f} FPS\n", (total_images * 1000.0) / total_time_ms);
        fmt::print("========================================\n");

    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "Model loading error: {}\n", e.what());
        return -1;
    }
    catch (const std::exception& e) {
        fmt::print(stderr, "General error: {}\n", e.what());
        return -1;
    }

    return 0;
}