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

namespace fs = std::filesystem;

cv::Mat preprocessImage(const cv::Mat& src_image, int resize_width, int resize_height, int crop_width, int crop_height) {
    cv::Mat resized_image;
    cv::resize(src_image, resized_image, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);

    // Center crop of the image
    int top = (resize_height - crop_height) / 2;
    int left = (resize_width - crop_width) / 2;
    cv::Rect roi(left, top, crop_width, crop_height);
    cv::Mat cropped_image = resized_image(roi).clone();

    // Convert from BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(cropped_image, rgb, cv::COLOR_BGR2RGB);

    // Convert to float32 and scale pixel values from [0, 255] to [0.0, 1.0]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    return float_img;
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

    if (!fs::exists(model_file_path)) {
        fmt::print(stderr, "Error: Model file does not exist: {}\n", model_file_path.string());
        return -1;
    }
    if (!fs::is_regular_file(model_file_path)) {
        fmt::print(stderr, "Error: Model path provided is not a file: {}\n", model_file_path.string());
        return -1;
    }
    if (model_file_path.extension() != ".onnx") {
        fmt::print(stderr, "Warning: Expected a .onnx extension, but got: {}\n", model_file_path.extension().string());
    }

    if (!fs::exists(input_dir_path)) {
        fmt::print(stderr, "Error: Input images directory does not exist: {}\n", input_dir_path.string());
        return -1;
    }
    if (!fs::is_directory(input_dir_path)) {
        fmt::print(stderr, "Error: Input path is not a directory: {}\n", input_dir_path.string());
        return -1;
    }
    if (fs::is_empty(input_dir_path)) {
        fmt::print(stderr, "Error: Input directory is empty. No images to process.\n");
        return -1;
    }

    try {
        if (!fs::exists(trt_cache_path)) {
            fs::create_directories(trt_cache_path);
        }
        if (!fs::exists(output_dir_path)) {
            fs::create_directories(output_dir_path);
        }
    }
    catch (const fs::filesystem_error& e) {
        fmt::print(stderr, "Filesystem error while creating directories: {}\n", e.what());
        return -1;
    }

    std::wstring model_path_ws = model_file_path.wstring();
    std::string trt_cache_str = trt_cache_path.string();
	std::string output_dir_str = output_dir_path.string();
	std::string input_dir_str = input_dir_path.string();

    // Preprocessing configuration parameters
    const int BATCH_SIZE = 17;
    const int RESIZE_WIDTH = 256;
    const int RESIZE_HEIGHT = 256;

    // Set crop dimensions equal to resize dimensions to match the ONNX expected input shape
    const int CROP_WIDTH = 256;
    const int CROP_HEIGHT = 256;

    // Environment and Session Options Initialization
    Ort::Env inferenceEnv(ORT_LOGGING_LEVEL_WARNING, "GpuInference");
    Ort::SessionOptions session_options;

	// Enable TensorRT provider with FP16 precision and engine caching for optimal GPU performance
    OrtTensorRTProviderOptions trt_options{};
	trt_options.device_id = 0;
    trt_options.trt_fp16_enable = 1;
	trt_options.trt_engine_cache_enable = 1;
	trt_options.trt_engine_cache_path = trt_cache_str.c_str();
    

    if (!fs::exists(trt_options.trt_engine_cache_path)) {
		fs::create_directories(trt_options.trt_engine_cache_path);
    }

    try {
        session_options.AppendExecutionProvider_TensorRT(trt_options);
    }
    catch (const std::exception& e) {    
        fmt::print(stderr, "TensorRT not supported or DLL missing: {}\n", e.what());
        return -1;
    }

	// Enable CUDA provider and configure memory optimizations as backup if TensorRT is unavailable
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

    const wchar_t* model_path = model_path_ws.c_str();
    try {
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
        output_names_str.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_names_str.push_back(output_name_ptr.get());
        }

        std::vector<const char*> output_names;
        output_names.reserve(num_output_nodes);
        for (const auto& name : output_names_str) {
            output_names.push_back(name.c_str());
        }

        const char* input_names[] = { input_name.c_str() };

        std::string target_directory = input_dir_str;
        std::string output_directory = output_dir_str;

        if (!fs::exists(output_directory)) {
            fs::create_directories(output_directory);
        }

        fmt::print("Executing GPU Warm-up...\n");

        const int WARMUP_ITERATIONS = 3;

        std::vector<int64_t> warmup_shape = base_input_shape;
        warmup_shape[0] = BATCH_SIZE;

        size_t warmup_tensor_size = BATCH_SIZE * 3 * CROP_HEIGHT * CROP_WIDTH;
        std::vector<float> dummy_data(warmup_tensor_size, 0.0f);

        auto warmup_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value dummy_tensor = Ort::Value::CreateTensor<float>(
            warmup_memory_info,
            dummy_data.data(),
            dummy_data.size(),
            warmup_shape.data(),
            warmup_shape.size()
        );

        // Run empty inferences to wake up Tensor Cores and allocate VRAM buffers
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            auto w_start = std::chrono::high_resolution_clock::now();

            session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &dummy_tensor,
                1,
                output_names.data(),
                output_names.size()
            );

            auto w_end = std::chrono::high_resolution_clock::now();
            fmt::print("  - Warm-up step {}/{} completed in {} ms\n",
                i + 1,
                WARMUP_ITERATIONS,
                std::chrono::duration_cast<std::chrono::milliseconds>(w_end - w_start).count()
            );
        }
        fmt::print("GPU Warm-up completed. Tensor Cores are ready.\n\n");

        std::vector<double> gpu_times;
		std::vector<double> postprocessing_times;
        std::vector<double> loading_times;
		std::vector<double> downloading_times;
		std::vector<double> preprocessing_times;
		std::vector<double> batch_prep_times;

        // Inference Lambda Function with Dynamic Shape Padding
        auto runInference = [&](std::vector<cv::Mat>& images, std::vector<cv::Mat>& orig_images, std::vector<std::string>& filenames, int batch_index) {
            if (images.empty()) return;

            // Keep track of how many real images we actually have
            int valid_image_count = images.size();

            // BATCH PADDING
            while (images.size() < BATCH_SIZE) {
                images.push_back(images.back().clone());
            }

            int current_batch_size = images.size();
            fmt::print("\nExecuting inference for batch {} ({} real images, padded to {})\n", batch_index, valid_image_count, current_batch_size);

            auto start_batch_prep = std::chrono::high_resolution_clock::now();
            cv::Mat blob = cv::dnn::blobFromImages(images, 1.0, cv::Size(CROP_WIDTH, CROP_HEIGHT), cv::Scalar(), false, false, CV_32F);
            if (!blob.isContinuous()) {
                blob = blob.clone();
            }

            std::vector<int64_t> current_input_shape = base_input_shape;
            current_input_shape[0] = current_batch_size;

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(blob.data), blob.total(), current_input_shape.data(), current_input_shape.size());
            auto end_batch_prep = std::chrono::high_resolution_clock::now();
            double prep_time = std::chrono::duration<double, std::milli>(end_batch_prep - start_batch_prep).count();
            batch_prep_times.push_back(prep_time);

            auto start_time = std::chrono::high_resolution_clock::now();

            auto output_tensors = session.Run(
                Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names.data(), output_names.size()
            );

            auto end_time = std::chrono::high_resolution_clock::now();
			auto inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            gpu_times.push_back(inference_time);

            // Output data pointers
            float* pred_scores = nullptr;
            float* anomaly_maps = nullptr;
            bool* pred_masks = nullptr;

            int out_map_h = CROP_HEIGHT, out_map_w = CROP_WIDTH;
            int out_mask_h = CROP_HEIGHT, out_mask_w = CROP_WIDTH;
            for (size_t i = 0; i < output_tensors.size(); ++i) {
                std::string node_name = output_names_str[i];
                std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

                auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

                if (node_name.find("score") != std::string::npos) {
                    pred_scores = output_tensors[i].GetTensorMutableData<float>();
                }
                else if (node_name.find("map") != std::string::npos) {
                    anomaly_maps = output_tensors[i].GetTensorMutableData<float>();
                    if (shape.size() >= 2) {
                        out_map_h = shape[shape.size() - 2];
                        out_map_w = shape[shape.size() - 1];
                    }
                }
                else if (node_name.find("mask") != std::string::npos) {
                    pred_masks = output_tensors[i].GetTensorMutableData<bool>();
                    if (shape.size() >= 2) {
                        out_mask_h = shape[shape.size() - 2];
                        out_mask_w = shape[shape.size() - 1];
                    }
                }
            }

            if (!pred_scores || !anomaly_maps) {
                fmt::print(stderr, "Critical Error: 'score' or 'map' tensors not found.\n");
                return;
            }

            int map_area = out_map_h * out_map_w;
            int mask_area = out_mask_h * out_mask_w;

            // OUTPUT SAVING
            for (int j = 0; j < valid_image_count; j++) {
                auto start_post = std::chrono::high_resolution_clock::now();
                fmt::print("File: {} | Predicted Score: {:.4f}\n", filenames[j], pred_scores[j]);

                cv::Mat original_bgr = orig_images[j].clone();

                // PROCESS ANOMALY HEATMAP
                int offset_map = j * map_area;
                cv::Mat map_anomaly(out_map_h, out_map_w, CV_32F, anomaly_maps + offset_map);

                // Normalize heatmap using MinMax
                cv::Mat heatmap_norm;
                cv::normalize(map_anomaly, heatmap_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

                // Apply Jet colormap
                cv::Mat heatmap_colored;
                cv::applyColorMap(heatmap_norm, heatmap_colored, cv::COLORMAP_JET);

                if (heatmap_colored.size() != original_bgr.size()) {
                    cv::resize(heatmap_colored, heatmap_colored, original_bgr.size(), 0, 0, cv::INTER_CUBIC);
                }

                cv::Mat overlay_img;
                cv::addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0, overlay_img);

                std::string score_text = fmt::format("Anomaly score: {:.4f}", pred_scores[j]);
                cv::putText(overlay_img, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::putText(overlay_img, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                // PROCESS PREDICTION MASK AND CONTOUR
                cv::Mat segmentation_img = original_bgr.clone();

                if (pred_masks) {
                    int offset_mask = j * mask_area;
                    cv::Mat mask_pred(out_mask_h, out_mask_w, CV_8U, pred_masks + offset_mask);
                    cv::Mat mask_uint8;

                    // Convert binary mask to uint8 format
                    double max_val;
                    cv::minMaxLoc(mask_pred, nullptr, &max_val);
                    if (max_val <= 1.0) {
                        mask_pred.convertTo(mask_uint8, CV_8UC1, 255.0);
                    }
                    else {
                        mask_pred.copyTo(mask_uint8);
                    }

                    if (mask_uint8.size() != original_bgr.size()) {
                        cv::resize(mask_uint8, mask_uint8, original_bgr.size(), 0, 0, cv::INTER_NEAREST);
                    }

                    // Extract boundaries and draw them
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(mask_uint8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    cv::drawContours(segmentation_img, contours, -1, cv::Scalar(0, 0, 255), 2);
                    cv::drawContours(overlay_img, contours, -1, cv::Scalar(0, 0, 255), 2);
                }

                
                // CONCATENATE IMAGES
                cv::Mat combined_result;
                std::vector<cv::Mat> images_to_concat = { original_bgr, overlay_img, segmentation_img };
                cv::hconcat(images_to_concat, combined_result);

                auto end_post = std::chrono::high_resolution_clock::now();
                double duration_postprocessing = std::chrono::duration<double, std::milli>(end_post - start_post).count();
                postprocessing_times.push_back(duration_postprocessing);

                // SAVE IMAGE TO DISK
                auto start_downloading = std::chrono::high_resolution_clock::now();

                std::string base_name = fs::path(filenames[j]).filename().string();
               
                std::string save_name = fmt::format("{:.3f}_{}", pred_scores[j], base_name);
                std::string out_path = (fs::path(output_directory) / save_name).string();

                cv::imwrite(out_path, combined_result);

                auto end_downloading = std::chrono::high_resolution_clock::now();
                auto downloading_time = std::chrono::duration<double, std::milli>(end_downloading - start_downloading).count();
                downloading_times.push_back(downloading_time);
            }

            images.clear();
            orig_images.clear();
            filenames.clear();
        };

        std::vector<cv::Mat> current_batch_images;
        std::vector<cv::Mat> current_original_images;
        std::vector<std::string> current_batch_filenames;

        current_batch_images.reserve(BATCH_SIZE);
        current_original_images.reserve(BATCH_SIZE);
        current_batch_filenames.reserve(BATCH_SIZE);

        int batch_counter = 0;

        fmt::print("Scanning directory and preprocessing on-the-fly...\n");
        for (const auto& entry : fs::directory_iterator(target_directory)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::string filename = entry.path().filename().string();

                // Read image explicitly in 3-channel BGR mode
				auto start_loading = std::chrono::high_resolution_clock::now();
                cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
				auto end_loading = std::chrono::high_resolution_clock::now();
				auto loading_time = std::chrono::duration<double, std::milli>(end_loading - start_loading).count();
				loading_times.push_back(loading_time);

                if (image.empty()) {
                    fmt::print(stderr, "Cannot read image: {}\n", file_path);
                    continue;
                }

                current_original_images.push_back(image.clone());

				auto start_preprocessing = std::chrono::high_resolution_clock::now();
                cv::Mat preprocessed_image = preprocessImage(image, RESIZE_WIDTH, RESIZE_HEIGHT, CROP_WIDTH, CROP_HEIGHT);
				auto end_preprocessing = std::chrono::high_resolution_clock::now();
                double duration_preprocessing = std::chrono::duration<double, std::milli>(end_preprocessing - start_preprocessing).count();
                preprocessing_times.push_back(duration_preprocessing);

                current_batch_images.push_back(preprocessed_image);
                current_batch_filenames.push_back(filename);

                // Trigger inference if the batch buffer is full
                if (current_batch_images.size() == BATCH_SIZE) {
                    runInference(current_batch_images, current_original_images, current_batch_filenames, batch_counter++);
                }
            }
        }

        // Process any remaining images in the last incomplete batch
        if (!current_batch_images.empty()) {
            runInference(current_batch_images, current_original_images, current_batch_filenames, batch_counter++);
        }

        auto calc_avg = [](const std::vector<double>& times) -> double {
            if (times.empty()) return 0.0;
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            return sum / times.size();
        };

        double avg_loading = calc_avg(loading_times);
        double avg_preprocessing = calc_avg(preprocessing_times);
        double avg_postprocessing = calc_avg(postprocessing_times);
        double avg_downloading = calc_avg(downloading_times);

        double avg_batch_prep = calc_avg(batch_prep_times) / BATCH_SIZE;

        double avg_gpu_batch = calc_avg(gpu_times);
        double avg_gpu_per_image = avg_gpu_batch / BATCH_SIZE;

        double total_pipeline_per_image = avg_loading + avg_preprocessing + avg_batch_prep + avg_gpu_per_image + avg_postprocessing + avg_downloading;

        fmt::print("\nInference completed successfully!\n");
        fmt::print("\n========================================\n");
        fmt::print("        DETAILED PERFORMANCE REPORT      \n");
        fmt::print("========================================\n");
        fmt::print("Batch Size used: {}\n", BATCH_SIZE);
        fmt::print("Total Batches processed: {}\n", batch_counter);
        fmt::print("Total Images processed: {}\n", loading_times.size());
        fmt::print("----------------------------------------\n");
        fmt::print("Average Times per SINGLE Image:\n");
        fmt::print("  - Loading (Disk -> RAM)  : {:.3f} ms\n", avg_loading);
        fmt::print("  - Preprocessing (CPU)    : {:.3f} ms\n", avg_preprocessing);
        fmt::print("  - Batch Prep (CPU Memory): {:.3f} ms\n", avg_batch_prep);
        fmt::print("  - GPU Inference          : {:.3f} ms  (Total Batch: {:.3f} ms)\n", avg_gpu_per_image, avg_gpu_batch);
        fmt::print("  - Postprocessing (CPU)   : {:.3f} ms\n", avg_postprocessing);
        fmt::print("  - Downloading (RAM->Disk): {:.3f} ms\n", avg_downloading);
        fmt::print("----------------------------------------\n");
        fmt::print("  TOTAL Pipeline Time/Image: {:.3f} ms\n", total_pipeline_per_image);
        fmt::print("  Estimated Throughput     : {:.2f} FPS\n", 1000.0 / total_pipeline_per_image);
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