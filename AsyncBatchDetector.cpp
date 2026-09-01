#include "AsyncBatchDetector.h"
#include "OrtSessionConfig.h"
#include "AsyncLogger.h"

#include <opencv2/dnn.hpp>
#include <filesystem>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// PinnedPool: page-locked host buffers allocated once and recycled.
// ---------------------------------------------------------------------------
void AsyncBatchDetector::PinnedPool::init(size_t count, size_t elems)
{
    all_.reserve(count);
    free_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        float* p = nullptr;
        if (cudaMallocHost((void**)&p, elems * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("PinnedPool cudaMallocHost failed");
        all_.push_back(p);
        free_.push_back(p);
    }
}

float* AsyncBatchDetector::PinnedPool::acquire()
{
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [&] { return !free_.empty() || stopped_; });
    if (stopped_) return nullptr;
    float* p = free_.back();
    free_.pop_back();
    return p;
}

void AsyncBatchDetector::PinnedPool::release(float* p)
{
    if (!p) return;
    std::lock_guard<std::mutex> l(m_);
    free_.push_back(p);
    cv_.notify_one();
}

void AsyncBatchDetector::PinnedPool::stop()
{
    std::lock_guard<std::mutex> l(m_);
    stopped_ = true;
    cv_.notify_all();
}

AsyncBatchDetector::PinnedPool::~PinnedPool()
{
    for (float* p : all_) if (p) cudaFreeHost(p);
}

AsyncBatchDetector::AsyncBatchDetector(const DetectorConfig& cfg, ResultCallback sink, PerformanceMetrics& metrics)
    : cfg_(cfg), sink_(std::move(sink)), metrics_(metrics)
{
    metrics_.batch_size = cfg_.batchSize;
    cv::setNumThreads((std::max)(1, cv::getNumberOfCPUs() / 2));

    if (!cfg_.trtCacheDir.empty() && !fs::exists(cfg_.trtCacheDir)) {
        fs::create_directories(cfg_.trtCacheDir);
    }
    cudaSetDevice(0);

    // ---- create the ORT sessions (one per inference thread) ----
    sessions_.resize(cfg_.numInfThreads);
    for (int i = 0; i < cfg_.numInfThreads; ++i) {
        if(cudaStreamCreateWithFlags(&sessions_[i].stream, cudaStreamNonBlocking) != cudaSuccess)
			throw std::runtime_error("cudaStreamCreate failed");
        Ort::SessionOptions options;
		sessions_[i].gpu = ConfigureOrtSessionOptions(options, "Anomaly", cfg_.partition, sessions_[i].stream);
        sessions_[i].session = std::make_unique<Ort::Session>(env_, cfg_.modelPath.c_str(), options);
    }
    Ort::Session& s0 = *sessions_[0].session;

    // ---- input geometry ----
    inputName_ = s0.GetInputNameAllocated(0, allocator_).get();
    std::vector<int64_t> inShapeModel = s0.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    modelC_ = inShapeModel.size() > 1 && inShapeModel[1] > 0 ? (int)inShapeModel[1] : cfg_.channels;
    modelH_ = inShapeModel.size() > 2 && inShapeModel[2] > 0 ? (int)inShapeModel[2] : cfg_.imgH;
    modelW_ = inShapeModel.size() > 3 && inShapeModel[3] > 0 ? (int)inShapeModel[3] : cfg_.imgW;

    // ---- detect score / map outputs (same heuristic as AnomalyEngine) ----
    size_t nOut = s0.GetOutputCount();
    std::vector<std::string> outNames;
    for (size_t idx = 0; idx < nOut; ++idx)
        outNames.emplace_back(s0.GetOutputNameAllocated(idx, allocator_).get());

    for (size_t idx = 0; idx < nOut; ++idx) {
        auto shape = s0.GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
            if (scoreIdx_ < 0) scoreIdx_ = (int)idx;
        }
        else if (shape.size() >= 3) {
            if (mapIdx_ < 0) mapIdx_ = (int)idx;
        }
    }
    if (scoreIdx_ < 0) throw std::runtime_error("Anomaly model missing scalar score output.");
    scoreName_ = outNames[scoreIdx_];
    if (mapIdx_ >= 0) mapName_ = outNames[mapIdx_];

    // ---- contract (metadata-driven preprocessing + thresholds) ----
    contract_.Load(s0);

    // ---- fixed batched I/O shapes ----
    const int B = cfg_.batchSize;
    std::vector<int64_t> inShape = { B, modelC_, modelH_, modelW_ };
    inputElems_ = (size_t)B * modelC_ * modelH_ * modelW_;

    std::vector<int64_t> scoreShape = s0.GetOutputTypeInfo(scoreIdx_).GetTensorTypeAndShapeInfo().GetShape();
    scoreShape[0] = B;
    for (size_t d = 1; d < scoreShape.size(); ++d) if (scoreShape[d] < 0) scoreShape[d] = 1;
    scoreElems_ = 1; for (int64_t d : scoreShape) scoreElems_ *= d;

    std::vector<int64_t> mapShape;
    if (mapIdx_ >= 0) {
        mapShape = s0.GetOutputTypeInfo(mapIdx_).GetTensorTypeAndShapeInfo().GetShape();
        mapShape[0] = B;
        for (size_t d = 1; d < mapShape.size(); ++d) if (mapShape[d] < 0) mapShape[d] = 1;
        mapElems_ = 1; for (int64_t d : mapShape) mapElems_ *= d;
        mapH_ = (int)mapShape[mapShape.size() - 2];
        mapW_ = (int)mapShape[mapShape.size() - 1];
    }

    imgElems_ = inputElems_ / B;
    scoreElems1_ = scoreElems_ / B;
    mapElems1_ = mapElems_ ? mapElems_ / B : 0;
    in1Shape_ = inShape;    in1Shape_[0] = 1;
    sc1Shape_ = scoreShape; sc1Shape_[0] = 1;
    mp1Shape_ = mapShape;   if (!mp1Shape_.empty()) mp1Shape_[0] = 1;

    // ---- antialiased resize coefficients (input image size -> model size) ----
    resizeCoeffs_.Build(cfg_.imgW, cfg_.imgH, modelW_, modelH_);

    // ---- pinned host staging pool (allocated once, recycled per batch) ----
    // Enough for every batch that can be simultaneously in flight between the
    // prep and inference stages: q_prep capacity + prep workers + inf workers.
    const size_t pinnedCount = 5 + (size_t)num_prep_threads_ + (size_t)cfg_.numInfThreads + 3;
    pinnedPool_.init(pinnedCount, inputElems_);
    Log::Info("Pinned pool: {} buffers x {} f32 ({:.1f} MB total).",
        pinnedCount, inputElems_, pinnedCount * inputElems_ * sizeof(float) / (1024.0 * 1024.0));

    // ---- per-session IoBinding setup ----
    for (int i = 0; i < cfg_.numInfThreads; ++i) {
        SessionCtx& ctx = sessions_[i];
        if (cudaMallocHost((void**)&ctx.h_score, scoreElems_ * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMallocHost h_score failed");
        std::memset(ctx.h_score, 0, scoreElems_ * sizeof(float));
        const size_t mapAlloc = (mapElems_ ? mapElems_ : 1);
        if (cudaMallocHost((void**)&ctx.h_map, mapAlloc * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMallocHost h_map failed");
        std::memset(ctx.h_map, 0, mapAlloc * sizeof(float));
        ctx.binding = std::make_unique<Ort::IoBinding>(*ctx.session);

        if (ctx.gpu) {
            if (cudaMalloc(&ctx.d_input, inputElems_ * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc input");
            if (cudaMalloc(&ctx.d_score, scoreElems_ * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc score");
            if (mapIdx_ >= 0 && cudaMalloc(&ctx.d_map, mapElems_ * sizeof(float)) != cudaSuccess) throw std::runtime_error("cudaMalloc map");

            const bool loop1 = cfg_.loopBatch1;
            Ort::MemoryInfo cudaMem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
            ctx.inT = Ort::Value::CreateTensor<float>(cudaMem, ctx.d_input,
                loop1 ? imgElems_ : inputElems_,
                loop1 ? in1Shape_.data() : inShape.data(),
                loop1 ? in1Shape_.size() : inShape.size());
            ctx.binding->BindInput(inputName_.c_str(), ctx.inT);
            ctx.scoreT = Ort::Value::CreateTensor<float>(cudaMem, ctx.d_score,
                loop1 ? scoreElems1_ : scoreElems_,
                loop1 ? sc1Shape_.data() : scoreShape.data(),
                loop1 ? sc1Shape_.size() : scoreShape.size());
            ctx.binding->BindOutput(scoreName_.c_str(), ctx.scoreT);
            if (mapIdx_ >= 0) {
                ctx.mapT = Ort::Value::CreateTensor<float>(cudaMem, ctx.d_map,
                    loop1 ? mapElems1_ : mapElems_,
                    loop1 ? mp1Shape_.data() : mapShape.data(),
                    loop1 ? mp1Shape_.size() : mapShape.size());
                ctx.binding->BindOutput(mapName_.c_str(), ctx.mapT);
            }
            Log::Info("[Session {}] GPU IoBinding: input {} f32, score {} f32, map {} f32 (VRAM).", i, inputElems_, scoreElems_, mapElems_);
        }
        else {
            ctx.h_input.assign(inputElems_, 0.0f);
            Ort::MemoryInfo cpuMem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            ctx.inT = Ort::Value::CreateTensor<float>(cpuMem, ctx.h_input.data(), inputElems_, inShape.data(), inShape.size());
            ctx.binding->BindInput(inputName_.c_str(), ctx.inT);
            ctx.scoreT = Ort::Value::CreateTensor<float>(cpuMem, ctx.h_score, scoreElems_, scoreShape.data(), scoreShape.size());
            ctx.binding->BindOutput(scoreName_.c_str(), ctx.scoreT);
            if (mapIdx_ >= 0) {
                ctx.mapT = Ort::Value::CreateTensor<float>(cpuMem, ctx.h_map, mapElems_, mapShape.data(), mapShape.size());
                ctx.binding->BindOutput(mapName_.c_str(), ctx.mapT);
            }
            Log::Warning("[Session {}] CPU IoBinding fallback (no GPU acceleration).", i);
        }

        // Warmup: builds/loads the TensorRT engine (cached) and locks kernels.
        if (ctx.gpu) cudaMemset(ctx.d_input, 0, inputElems_ * sizeof(float));
        Ort::RunOptions ro{ nullptr };
        for (int r = 0; r < 10; ++r) ctx.session->Run(ro, *ctx.binding);
        Log::Info("[Session {}] warmup complete.", i);
    }

    // ---- launch the pipeline ----
    for (int i = 0; i < num_prep_threads_; ++i)
        pool_prep_.emplace_back(&AsyncBatchDetector::preprocessingWorker, this);
    for (int i = 0; i < cfg_.numInfThreads; ++i)
        pool_inf_.emplace_back(&AsyncBatchDetector::inferenceWorker, this, i);
    for (int i = 0; i < num_post_threads_; ++i)
        pool_post_.emplace_back(&AsyncBatchDetector::postprocessingWorker, this);

    Log::Info("AsyncBatchDetector ready | batch {} | model {}x{}x{} | map {}x{} | prep {} inf {} post {}.",
        cfg_.batchSize, modelC_, modelH_, modelW_, mapH_, mapW_, num_prep_threads_, cfg_.numInfThreads, num_post_threads_);
}

AsyncBatchDetector::~AsyncBatchDetector()
{
    is_running_ = false;
    q_raw_.stop(); q_prep_.stop(); q_inf_.stop();
    pinnedPool_.stop();   // unblock any prep thread waiting on acquire()
    for (auto& t : pool_prep_) if (t.joinable()) t.join();
    for (auto& t : pool_inf_)  if (t.joinable()) t.join();
    for (auto& t : pool_post_) if (t.joinable()) t.join();

    for (auto& ctx : sessions_) {
		ctx.binding.reset();
		ctx.session.reset();
        if (ctx.d_input) cudaFree(ctx.d_input);
        if (ctx.d_score) cudaFree(ctx.d_score);
        if (ctx.d_map)   cudaFree(ctx.d_map);
        if (ctx.h_score) cudaFreeHost(ctx.h_score);
        if (ctx.h_map) cudaFreeHost(ctx.h_map);
		if (ctx.stream) cudaStreamDestroy(ctx.stream);
    }
}

bool AsyncBatchDetector::pushBatch(const unsigned char* slot, int frame_id, int seq)
{
    const int type = (cfg_.channels == 1) ? CV_8UC1 : CV_8UC3;
    const size_t imgBytes = (size_t)cfg_.imgW * cfg_.imgH * cfg_.channels;

    RawImageTask task;
    task.frame_id = frame_id;
    task.seq = seq;
    task.patches.reserve(cfg_.batchSize);
    for (int i = 0; i < cfg_.batchSize; ++i) {
        cv::Mat img(cfg_.imgH, cfg_.imgW, type, const_cast<unsigned char*>(slot + i * imgBytes));
        task.patches.push_back(img.clone()); // detach from the MMF slot
    }
    if (!q_raw_.try_push(std::move(task))) {
        dropped_frames_++;
        Log::Warning("[DROP] batch frame {} dropped (queue full). total={}", frame_id, dropped_frames_.load());
        return false;
    }
    return true;
}

void AsyncBatchDetector::preprocessingWorker()
{
    const int B = cfg_.batchSize;
    const size_t planeSize = (size_t)modelH_ * modelW_;
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float stddev[3] = { 0.229f, 0.224f, 0.225f };
    const bool normInGraph = contract_.normalizationInGraph;
    const bool swapRB = contract_.convertBgrToRgb;

    while (is_running_) {
        try {
            RawImageTask task;
            if (!q_raw_.pop(task)) break;

            auto batch = std::make_shared<BatchData>();
            batch->frame_id = task.frame_id;
            batch->seq = task.seq;
            batch->original_patches = task.patches;

            // batch prep: borrow a pinned staging buffer from the pool (no
            // cudaMallocHost per batch -> no device-wide sync, pipeline overlaps).
            auto t_bp0 = std::chrono::high_resolution_clock::now();
            float* pinned = pinnedPool_.acquire();
            if (!pinned) break; // pool stopped (shutdown)
            batch->pinned_blob.reset(pinned, [this](float* p) { pinnedPool_.release(p); });
            auto t_bp1 = std::chrono::high_resolution_clock::now();
            metrics_.addBatchPrepTime(std::chrono::duration<double, std::milli>(t_bp1 - t_bp0).count());

            // prep: contract preprocessing straight into the pinned NCHW buffer
            auto t_p0 = std::chrono::high_resolution_clock::now();
            cv::parallel_for_(cv::Range(0, B), [&](const cv::Range& r) {
                cv::Mat resized;
                std::vector<cv::Mat> planes(modelC_);
                for (int i = r.start; i < r.end; ++i) {
                    ResizeAntialias(resizeCoeffs_, task.patches[i], resized);
                    if (swapRB && modelC_ == 3) cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
                    cv::split(resized, planes);
                    for (int c = 0; c < modelC_; ++c) {
                        const float scale = normInGraph ? (1.0f / 255.0f) : (1.0f / (255.0f * stddev[c]));
                        const float offset = normInGraph ? 0.0f : (mean[c] / stddev[c]);
                        float* dst = pinned + (size_t)i * modelC_ * planeSize + (size_t)c * planeSize;
                        cv::Mat planeF32(modelH_, modelW_, CV_32FC1, dst);
                        planes[c].convertTo(planeF32, CV_32FC1, scale, -offset);
                    }
                }
            });
            auto t_p1 = std::chrono::high_resolution_clock::now();
            metrics_.addPreprocessingTime(std::chrono::duration<double, std::milli>(t_p1 - t_p0).count());

            if (!q_prep_.push(batch)) break;
        }
        catch (const std::exception& e) {
            dropped_frames_++;
            Log::Error("[EXC prep] {}", e.what());
        }
    }
}

void AsyncBatchDetector::inferenceWorker(int session_index)
{
    SessionCtx& ctx = sessions_[session_index];
    const int B = cfg_.batchSize;
    Ort::RunOptions ro{ nullptr };

    while (is_running_) {
        try {
            std::shared_ptr<BatchData> batch;
            if (!q_prep_.pop(batch)) break;

            auto t0 = std::chrono::high_resolution_clock::now();
            if (ctx.gpu) {
                // Explicit H2D into VRAM, Run on the bound VRAM tensors, explicit D2H.
                // All ops enqueued on ctx.stream (the ORT user compute stream), so
                // H2D -> Run -> D2H are strictly ordered. The pinned input buffer
                // stays alive until cudaStreamSynchronize below: releasing it right
                // after enqueuing the async H2D would let a prep thread reacquire
                // and overwrite it while the copy is still in flight.
                cudaMemcpyAsync(ctx.d_input, batch->pinned_blob.get(), inputElems_ * sizeof(float), cudaMemcpyHostToDevice, ctx.stream);

                if (cfg_.loopBatch1) {
                    Ort::MemoryInfo cudaMem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
                    for (int i = 0; i < B; ++i) {
                        auto in1 = Ort::Value::CreateTensor<float>(cudaMem,
                            ctx.d_input + (size_t)i * imgElems_, imgElems_, in1Shape_.data(), in1Shape_.size());
                        ctx.binding->BindInput(inputName_.c_str(), in1);
                        auto sc1 = Ort::Value::CreateTensor<float>(cudaMem,
                            +ctx.d_score + (size_t)i * scoreElems1_, scoreElems1_, sc1Shape_.data(), sc1Shape_.size());
                        ctx.binding->BindOutput(scoreName_.c_str(), sc1);
                        if (mapIdx_ >= 0) {
                            auto mp1 = Ort::Value::CreateTensor<float>(cudaMem,
                                ctx.d_map + (size_t)i * mapElems1_, mapElems1_, mp1Shape_.data(), mp1Shape_.size());
                            ctx.binding->BindOutput(mapName_.c_str(), mp1);
                        }
                        ctx.session->Run(ro, *ctx.binding);
                        
                    }
                }
                else {
                    ctx.session->Run(ro, *ctx.binding);
                }
                cudaMemcpyAsync(ctx.h_score, ctx.d_score, scoreElems_ * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                if (mapIdx_ >= 0)
                    cudaMemcpyAsync(ctx.h_map, ctx.d_map, mapElems_ * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
				cudaStreamSynchronize(ctx.stream);
                batch->pinned_blob.reset(); // H2D completed -> safe to recycle
            }
            else {
                std::memcpy(ctx.h_input.data(), batch->pinned_blob.get(), inputElems_ * sizeof(float));
                batch->pinned_blob.reset(); // done with the staging buffer -> back to the pool
                ctx.session->Run(ro, *ctx.binding); // outputs land in h_score / h_map
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            metrics_.addGpuTime(std::chrono::duration<double, std::milli>(t1 - t0).count());

            auto res = std::make_shared<InferenceResult>();
            res->batch_info = batch;
            res->scores.assign(ctx.h_score, ctx.h_score + B);
            if (mapIdx_ >= 0) res->maps.assign(ctx.h_map, ctx.h_map + mapElems_);
            res->map_h = mapH_; res->map_w = mapW_;

            if (!q_inf_.push(res)) break;
        }
        catch (const Ort::Exception& e) {
            dropped_frames_++;
            Log::Error("[EXC inf/TensorRT] {}", e.what());
        }
        catch (const std::exception& e) {
            dropped_frames_++;
            Log::Error("[EXC inf] {}", e.what());
        }
    }
}

void AsyncBatchDetector::postprocessingWorker()
{
    const int B = cfg_.batchSize;
    const size_t imgBytes = (size_t)cfg_.imgW * cfg_.imgH * cfg_.channels;
    const double alpha = 255.0 / (contract_.mapMax - contract_.mapMin);
    const double beta = -255.0 * contract_.mapMin / (contract_.mapMax - contract_.mapMin);

    while (is_running_) {
        try {
            std::shared_ptr<InferenceResult> res;
            if (!q_inf_.pop(res)) break;
            if (res->scores.empty()) throw std::runtime_error("empty score output");

            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = std::make_unique<unsigned char[]>((size_t)B * imgBytes);
            std::vector<uint8_t> statuses(B, 0);
            const bool haveMap = (mapIdx_ >= 0) && !res->maps.empty();

            cv::parallel_for_(cv::Range(0, B), [&](const cv::Range& r) {
                cv::Mat u8heat, largeHeat, colorHeat, overlay, maskFull;
                std::vector<std::vector<cv::Point>> contours;
                for (int j = r.start; j < r.end; ++j) {
                    const cv::Mat& orig = res->batch_info->original_patches[j];
                    const float score = res->scores[j];
                    const bool reject = score >= contract_.scoreThreshold;
                    statuses[j] = reject ? 1 : 0;

                    cv::Mat dest(cfg_.imgH, cfg_.imgW, CV_8UC3, out.get() + (size_t)j * imgBytes);

                    if (haveMap) {
                        cv::Mat map(res->map_h, res->map_w, CV_32F, res->maps.data() + (size_t)j * res->map_h * res->map_w);
                        map.convertTo(u8heat, CV_8UC1, alpha, beta);
                        cv::resize(u8heat, largeHeat, cv::Size(cfg_.imgW, cfg_.imgH), 0, 0, cv::INTER_LINEAR);
                        cv::applyColorMap(largeHeat, colorHeat, cv::COLORMAP_JET);
                        cv::addWeighted(orig, 1.0 - m_blendAlpha_, colorHeat, m_blendAlpha_, 0.0, overlay);

                        if (contract_.hasPixelThreshold) {
                            cv::Mat maskModel = map >= contract_.pixelThreshold;
                            cv::resize(maskModel, maskFull, cv::Size(cfg_.imgW, cfg_.imgH), 0, 0, cv::INTER_NEAREST);
                            contours.clear();
                            cv::findContours(maskFull, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                            cv::drawContours(overlay, contours, -1, cv::Scalar(255, 0, 0), 2);
                        }
                        overlay.copyTo(dest);
                    }
                    else {
                        orig.copyTo(dest); // no map output: publish the original frame
                    }
                }
            });

            auto t1 = std::chrono::high_resolution_clock::now();
            metrics_.addPostprocessingTime(std::chrono::duration<double, std::milli>(t1 - t0).count());
            metrics_.printRollingAverage(10);

            if (sink_) {
                BatchOutput outp;
                outp.frame_id = res->batch_info->frame_id;
                outp.seq = res->batch_info->seq;
                outp.map_h = res->map_h; outp.map_w = res->map_w;
                outp.scores = res->scores;
                outp.statuses = std::move(statuses);
                outp.overlay = std::move(out);
                outp.overlayBytesPerImage = imgBytes;
                sink_(std::move(outp));
            }
        }
        catch (const std::exception& e) {
            dropped_frames_++;
            Log::Error("[EXC post] {}", e.what());
        }
    }
}
