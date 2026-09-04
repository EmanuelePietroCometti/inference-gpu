#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "ContractMetadata.h"
#include "PerformanceMetrics.h"
#include "RealTimeConfig.h"

//
// AsyncBatchDetector
// The batch-17, 3-stage asynchronous pipeline of the original inference-gpu,
// preserved (prep -> inf -> post + PerformanceMetrics) but restructured so it
// can be driven over MMF instead of a local simulator:
//   - source: pushBatch() fed by the WorkerGPU ingest thread (from the input MMF)
//   - sink:   a ResultCallback that publishes results back into the output MMFs
//
// It adopts from ONNX_inference: the model "contract" (metadata-driven
// preprocessing + threshold), the antialiased resize, and IoBinding with
// VRAM-resident input/score/map tensors so H2D/D2H transfers are explicit and
// the ONNX Runtime internal staging copies are eliminated.
//

// One completed batch, handed to the sink. Move-only: it owns the overlay buffer.
struct BatchOutput {
    int frame_id = 0;
    int seq = 0;               // monotonic batch index (output ring slot key)
    int map_h = 0, map_w = 0;
    std::vector<float>   scores;        // kBatchSize raw anomaly scores
    std::vector<uint8_t> statuses;      // 0 = OK, 1 = REJECT
    std::unique_ptr<unsigned char[]> overlay; // batch * overlayBytesPerImage
    size_t overlayBytesPerImage = 0;
};

using ResultCallback = std::function<void(BatchOutput&&)>;

struct DetectorConfig {
    std::wstring modelPath;
    std::string  trtEngineCacheDir = "trt_engine_cache";
	std::string  trtTimingCacheDir = "trt_timing_cache";
	std::string  tag = "AnomalyDetector";
    int   imgW = 0;            // width  of one input image (== model image width)
    int   imgH = 0;            // height of one input image
    int   channels = 3;        // bpp/8 (3 for 24bpp BGR)
    int   batchSize = 17;
    bool  loopBatch1 = true;
    int   numInfThreads = 2;   // concurrent ORT sessions (decision: keep 2)
    RT::CpuPartition partition;// core slice for OrtSessionConfig
};

class AsyncBatchDetector {
public:
    AsyncBatchDetector(const DetectorConfig& cfg, ResultCallback sink, PerformanceMetrics& metrics);
    ~AsyncBatchDetector();

    // Enqueue one batch. 'slot' points at kBatchSize contiguous images
    // (imgW*imgH*channels bytes each). 'seq' is the monotonic batch index used
    // to slot the result in the output ring; 'frame_id' is the producer's id.
    // Returns false if the queue is full (frame dropped) — never blocks ingest.
    bool pushBatch(const unsigned char* slot, int frame_id, int seq);

    int64_t getDroppedFramesCount() const { return dropped_frames_.load(); }
    void resetDroppedFramesCount() { dropped_frames_.store(0); }

private:
    // ---- pipeline payloads ----
    struct RawImageTask { std::vector<cv::Mat> patches; int frame_id; int seq; };

    struct CudaPinnedDeleter { void operator()(float* p) const { if (p) cudaFreeHost(p); } };
    struct BatchData {
        std::shared_ptr<float> pinned_blob;      // [B*C*H*W] host-pinned NCHW
        std::vector<cv::Mat> original_patches;   // for the overlay
        int frame_id;
        int seq;
    };
    struct InferenceResult {
        std::shared_ptr<BatchData> batch_info;
        std::vector<float> scores;   // B
        std::vector<float> maps;     // B*mH*mW
        int map_h = 0, map_w = 0;
    };

    enum class QPop {
        Ok, 
        Timeout,
        Stopped
    };

    // ---- bounded thread-safe queue (verbatim from the original) ----
    template<typename T>
    class BoundedQueue {
        std::queue<T> q_; std::mutex m_;
        std::condition_variable cv_push_, cv_pop_;
        size_t max_; bool stopped_ = false;
    public:
        explicit BoundedQueue(size_t max) : max_(max) {}
        bool push(T item) {
            std::unique_lock<std::mutex> l(m_);
            cv_push_.wait(l, [&] { return q_.size() < max_ || stopped_; });
            if (stopped_) return false;
            q_.push(std::move(item)); cv_pop_.notify_one(); return true;
        }
        bool try_push(T item) {
            std::lock_guard<std::mutex> l(m_);
            if (q_.size() >= max_ || stopped_) return false;
            q_.push(std::move(item)); cv_pop_.notify_one(); return true;
        }
        bool pop(T& item) {
            std::unique_lock<std::mutex> l(m_);
            cv_pop_.wait(l, [&] { return !q_.empty() || stopped_; });
            if (stopped_ && q_.empty()) return false;
            item = std::move(q_.front()); q_.pop(); cv_push_.notify_one(); return true;
        }
        template<class Rep, class Period>
        QPop pop_for(T& item, const std::chrono::duration<Rep, Period>& rel_time) {
            std::unique_lock<std::mutex> l(m_);
            if (!cv_pop_.wait_for(l, rel_time, [&] { return !q_.empty() || stopped_; }))
                return QPop::Timeout;
            if (stopped_ && q_.empty()) return QPop::Stopped;
            item = std::move(q_.front()); q_.pop(); cv_push_.notify_one(); 
            return QPop::Ok;
		}
        void stop() { std::lock_guard<std::mutex> l(m_); stopped_ = true; cv_push_.notify_all(); cv_pop_.notify_all(); }
    };

    // Fixed pool of page-locked (pinned) host staging buffers, allocated ONCE.
    // cudaMallocHost / cudaFreeHost synchronize the whole device, so calling them
    // per batch serializes the pipeline and stalls the GPU. The prep stage borrows
    // a buffer, the inference stage returns it right after the H2D copy.
    class PinnedPool {
        std::vector<float*> all_;      // owned buffers (freed in dtor)
        std::vector<float*> free_;     // currently available
        std::mutex m_;
        std::condition_variable cv_;
        bool stopped_ = false;
    public:
        void init(size_t count, size_t elems);
        float* acquire();              // blocks until a buffer frees; nullptr if stopped
        void release(float* p);
        void stop();
        ~PinnedPool();

    };


    // Fixed pool of plain (non-pinned) host output buffers for the overlay
    // image, allocated ONCE. Mirrors PinnedPool's design for the input side: the
    // post stage was allocating and freeing a fresh ~B*imgBytes buffer (12+ MB
    // for a 512x512x3, batch-17 config) via make_unique EVERY batch -- pure
    // allocator + first-touch-page-fault churn, and the most plausible cause of
    // the largest observed latency outliers (177 ms max vs 81 ms avg).
    class OverlayPool {
        std::vector<unsigned char*> all_;
        std::vector<unsigned char*> free_;
        std::mutex m_;
        std::condition_variable cv_;
        bool stopped_ = false;
    public:
        void init(size_t count, size_t bytesPerBuffer);
        unsigned char* acquire();   // blocks until a buffer frees; nullptr if stopped
        void release(unsigned char* p);
        void stop();
        ~OverlayPool();
    };

    // Per-session ONNX + IoBinding context (one per inference thread)
    struct SessionCtx {
        std::unique_ptr<Ort::Session>   session;
        std::unique_ptr<Ort::IoBinding> binding;
        bool  gpu = false;

		cudaStream_t stream = nullptr;
        cudaEvent_t evStart = nullptr, evAfterH2D = nullptr, evAfterRun = nullptr, evAfterD2H = nullptr;
        // Device (VRAM) buffers for zero-copy tensors
        float* d_input = nullptr;
        float* d_score = nullptr;
        float* d_map = nullptr;
        // Host landing buffers (D2H targets, or the bound tensors on CPU fallback)
        std::vector<float> h_input;   // CPU-EP input staging only (no stream, no async)
        float* h_score = nullptr;     // pinned host: D2H landing + CPU-EP output tensor
        float* h_map = nullptr;
        float* h_warm = nullptr;
        Ort::Value inT{ nullptr }, scoreT{ nullptr }, mapT{ nullptr };
    };

    void preprocessingWorker();
    void inferenceWorker(int session_index);
    void postprocessingWorker();

    void submitBatch(SessionCtx& ctx, const float* pinned_input, bool recordMetrics, std::shared_ptr<float> pinnedBlob = nullptr);

    DetectorConfig cfg_;
    ResultCallback sink_;
    PerformanceMetrics& metrics_;
    // Queue wait quantum: how long an inference thread blocks on q_prep_ before
    // re-checking is_running_. Short = responsive shutdown; it is NOT the
    // keep-warm period.
    const int kQueueWaitQuantumMs_ = 20;
    // Continuous idle time before a dummy batch is submitted to hold the GPU
    // clocks up. Must be well above the frame interval, otherwise the dummy
    // work contends with real batches. 0 disables keep-warm entirely.
    const int keepWarmIdleMs_ = 250;

    std::atomic<bool> is_running_{ true };
    std::atomic<int64_t> dropped_frames_{ 0 };

    // Declared BEFORE the queues so it is destroyed AFTER them: queue teardown
    // releases any held BatchData (returning its pinned buffer to this pool),
    // then the pool frees the underlying page-locked memory.
    PinnedPool pinnedPool_;
    OverlayPool overlayPool_;

    BoundedQueue<RawImageTask> q_raw_{ 20 };
    BoundedQueue<std::shared_ptr<BatchData>> q_prep_{ 10 };
    BoundedQueue<std::shared_ptr<InferenceResult>> q_inf_{ 5 };

    std::vector<std::thread> pool_prep_, pool_inf_, pool_post_;
    const int num_prep_threads_ = 5;
    const int num_post_threads_ = 5;

    Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "AsyncBatchInference" };
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<SessionCtx> sessions_;

    ContractMetadata contract_;
    ResizeCoeffs resizeCoeffs_;

    std::string inputName_;
    std::string scoreName_, mapName_;
    int scoreIdx_ = -1, mapIdx_ = -1;

    int modelC_ = 3, modelH_ = 0, modelW_ = 0;
    int mapH_ = 0, mapW_ = 0;
    size_t inputElems_ = 0, scoreElems_ = 0, mapElems_ = 0;
    size_t imgElems_ = 0, scoreElems1_ = 0, mapElems1_ = 0;
	std::vector<int64_t> in1Shape_, sc1Shape_, mp1Shape_;


    float m_blendAlpha_ = 0.5f;
};
