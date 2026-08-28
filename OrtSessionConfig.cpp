#include "OrtSessionConfig.h"
#include "AsyncLogger.h"
#include <onnxruntime_session_options_config_keys.h>
#include <windows.h>
#include <process.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <unordered_map>

//
// OrtSessionConfig.h
// Single source of truth for ONNX Runtime execution-provider selection and
// session-option tuning. Every engine (Anomaly, Classification, ...) calls
// ConfigureOrtSessionOptions() so the configuration is guaranteed identical
// across all of them and maintained in one place.
//
// Build is selected at compile time via exactly one of:
//   ORT_EP_GPU       -> TensorRT (FP16 + engine/timing cache) -> CUDA -> CPU
//   ORT_EP_OPENVINO  -> OpenVINO (single stream) -> CPU
//   ORT_EP_CPU       -> built-in ORT CPU EP
//


namespace {

// Custom thread factory for the ORT intra-op pool.
//
// CRITICAL for the real-time budget: on the CPU EP the convolutions run on the
// intra-op pool threads, NOT on the worker thread that calls Run(). Boosting
// only the worker with SetThreadPriority would leave the threads doing the
// actual compute at NORMAL priority, where any background process can preempt
// them mid-frame. This factory creates the pool threads through
// _beginthreadex and raises each one to TIME_CRITICAL before it starts
// serving kernels.
struct OrtThreadCtx {
    OrtThreadWorkerFn fn;
    void* param;
};

unsigned __stdcall OrtRtThreadEntry(void* arg)
{
    std::unique_ptr<OrtThreadCtx> ctx(static_cast<OrtThreadCtx*>(arg));
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    ctx->fn(ctx->param);
    return 0;
}

OrtCustomThreadHandle OrtRtCreateThread(void* /*options*/, OrtThreadWorkerFn fn, void* param)
{
    auto* ctx = new OrtThreadCtx{ fn, param };
    const uintptr_t handle = _beginthreadex(nullptr, 0, &OrtRtThreadEntry, ctx, 0, nullptr);
    if (handle == 0) {
        delete ctx;
        return nullptr;
    }
    return reinterpret_cast<OrtCustomThreadHandle>(handle);
}

void OrtRtJoinThread(OrtCustomThreadHandle handle)
{
    HANDLE h = (HANDLE)handle;
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
}

} // namespace

// Configures execution providers + session options for one session.
// Returns true if a hardware-accelerated EP (TensorRT / CUDA / OpenVINO) was
// appended, false if the run will happen on the built-in CPU EP.
// 'tag' is only used to prefix log lines (e.g. "Anomaly", "Classification").
// 'cpuPartition' is the core slice owned by this session; empty = whole machine.
bool ConfigureOrtSessionOptions(Ort::SessionOptions& so, const std::string& tag,
    const RT::CpuPartition& cpuPartition)
{
    bool hardwareAccelerated = false;

    // Threads available to THIS session: its core slice when partitioned,
    // otherwise all usable physical cores minus the reserved one. The real
    // topology comes from RT (GetLogicalProcessorInformationEx intersected
    // with the process affinity mask): hardware_concurrency()/2 would be
    // WRONG on the production target, where hyper-threading is disabled and
    // that heuristic halves the real core count.
    unsigned sliceThreads;
    if (cpuPartition.logicalProcessors.empty()) {
        const unsigned computeCores = RT::PhysicalCoreCount();
        sliceThreads = computeCores > 1 ? computeCores - 1 : 1;
    }
    else {
        sliceThreads = static_cast<unsigned>(cpuPartition.logicalProcessors.size());
    }

#if defined(ORT_EP_GPU)
    // GPU build: TensorRT -> CUDA -> CPU
    // ORT graph fusions still run before the EP partitions the graph, so keep
    // them enabled; TensorRT/CUDA then take the fused subgraphs.
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const OrtApi& api = Ort::GetApi();

    // Absolute cache paths. Relative paths ("./trt_cache") break the moment the
    // working directory differs from the exe directory (service, launcher, or a
    // caller that changed CWD), silently disabling the cache.
    static const std::string engineCache =
        std::filesystem::absolute("trt_engine_cache").string();
    static const std::string timingCache =
        std::filesystem::absolute("trt_timing_cache").string();

    // TensorRT
    try {
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2*)>
            trtGuard(trt, [](OrtTensorRTProviderOptionsV2* p) {
            Ort::GetApi().ReleaseTensorRTProviderOptions(p);
                });

        const char* keys[] = {
            "device_id",
            "trt_fp16_enable",                      // FP16 kernels: big speedup on modern GPUs
            "trt_engine_cache_enable",              // reuse the serialized engine across process runs
            "trt_engine_cache_path",
            "trt_timing_cache_enable",              // reuse kernel-timing results: much faster (re)builds
            "trt_timing_cache_path",
            "trt_builder_optimization_level",       // 5 = most aggressive builder search (slower build, faster engine)
			"trt_max_workspace_size",         
			"trt_detailed_build_log",               // 1 = verbose builder log (useful for debugging)
			"trt_dump_subgraphs",                   // 1 = dump the subgraph ONNX to disk (useful for debugging)
        };
        const char* values[] = {
			"0", "1", "1", engineCache.c_str(), "1", timingCache.c_str(), "5", "12884901888", "0", "0"
        };
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
            trt, keys, values, sizeof(keys) / sizeof(keys[0])));

        so.AppendExecutionProvider_TensorRT_V2(*trt);
        hardwareAccelerated = true;
        Log::Info("[{}] TensorRT EP appended (FP16 + engine/timing cache).", tag);
    }
    catch (const Ort::Exception& e) {
        Log::Warning("[{}] TensorRT unavailable, trying CUDA: {}", tag, e.what());
    }

    // CUDA fallback (also covers subgraphs TensorRT cannot handle)
    try {
        OrtCUDAProviderOptionsV2* cuda = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
        std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>
            cudaGuard(cuda, [](OrtCUDAProviderOptionsV2* p) {
            Ort::GetApi().ReleaseCUDAProviderOptions(p);
                });

        const char* keys[] = { "device_id", "cudnn_conv_use_max_workspace" };
        const char* values[] = { "0", "1" };  // let cuDNN pick the fastest conv algo
        Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
            cuda, keys, values, sizeof(keys) / sizeof(keys[0])));

        so.AppendExecutionProvider_CUDA_V2(*cuda);
        hardwareAccelerated = true;
        Log::Info("[{}] CUDA EP appended.", tag);
    }
    catch (const Ort::Exception& e) {
        Log::Warning("[{}] CUDA unavailable, falling back to CPU: {}", tag, e.what());
    }

#elif defined(ORT_EP_OPENVINO)
    // OpenVINO build
    // OpenVINO recompiles the graph internally; ORT-side fusions are wasted work
    // (or interfere), so disable them and let OpenVINO own the optimization.
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Define active targets dynamically to adjust based on the selected backend
    std::string targetDevice = "GPU";
    std::string targetPrecision = "FP16"; // Critical Optimization: Intel UHD GPUs process FP16 natively and much faster than FP32

    try {
        std::unordered_map<std::string, std::string> ov;
        ov["device_type"] = targetDevice;
        ov["precision"] = targetPrecision;
        ov["num_streams"] = "1"; // Single stream to minimize single-frame latency

        // Thread limits apply if OpenVINO offloads certain operators to the CPU
        ov["num_of_threads"] = std::to_string(sliceThreads);

        so.AppendExecutionProvider_OpenVINO_V2(ov);

        hardwareAccelerated = true;
        Log::Info("[{}] OpenVINO EP successfully appended. Active Device: {} (Precision: {}, 1 stream, {} threads).",
            tag, targetDevice, targetPrecision, sliceThreads);
    }
    catch (const Ort::Exception& e) {
        // DYNAMIC FALLBACK Level 1: If the Intel GPU initialization fails, attempt to run OpenVINO on the CPU
        Log::Warning("[{}] Failed to initialize OpenVINO on GPU: {}. Attemping fallback to OpenVINO CPU...", tag, e.what());

        try {
            targetDevice = "CPU";
            targetPrecision = "FP32"; // The CPU handles standard FP32 math operations more efficiently

            std::unordered_map<std::string, std::string> ov_cpu;
            ov_cpu["device_type"] = targetDevice;
            ov_cpu["precision"] = targetPrecision;
            ov_cpu["num_streams"] = "1";
            ov_cpu["num_of_threads"] = std::to_string(sliceThreads);

            so.AppendExecutionProvider_OpenVINO_V2(ov_cpu);
            hardwareAccelerated = true; // Still accelerated via OpenVINO engine, but executing on CPU
            Log::Info("[{}] OpenVINO EP appended via Fallback. Active Device: {} (Precision: {}, 1 stream, {} threads).",
                tag, targetDevice, targetPrecision, sliceThreads);
        }
        catch (const Ort::Exception& e_cpu) {
            // FINAL FALLBACK Level 2: Entirely disable OpenVINO and revert to native ONNX Runtime CPU EP
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            hardwareAccelerated = false;
            Log::Error("[{}] OpenVINO completely unavailable. Falling back to native ONNX Runtime CPU: {}", tag, e_cpu.what());
        }
    }

#elif defined(ORT_EP_CPU)
    // CPU build: built-in ORT CPU EP
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Log::Info("[{}] CPU build: using the default ONNX Runtime CPU EP.", tag);

#else
#error "No execution provider selected: define one of ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_CPU."
#endif

    // ---- Threading + numeric policy, driven by whether an accelerator loaded ----
    if (hardwareAccelerated) {
        // The accelerator runs the heavy compute on its own threads/streams;
        // a single ORT intra-op thread just feeds it and avoids context switching.
        so.SetIntraOpNumThreads(1);
    }
    else {
        // CPU path: the ORT intra-op pool actually runs the convolutions.
        so.EnableCpuMemArena();                                     // avoid per-run OS allocations
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);         // no inter-op locking overhead
        so.SetIntraOpNumThreads(static_cast<int>(sliceThreads));    // this session's core slice

        // Flush denormals to zero ON THE ORT WORKER THREADS. Setting MXCSR by hand
        // (_MM_SET_FLUSH_ZERO_MODE / _MM_SET_DENORMALS_ZERO_MODE) only affects the
        // calling thread, not the intra-op pool that runs the conv kernels, so it
        // is effectively a no-op there. This config entry is applied by ORT to
        // every intra-op thread, which is what actually matters for CNNs.
        so.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

        // Pin the pool threads inside this session's core slice. ORT expects
        // intra_op_num_threads - 1 affinity entries: the calling thread (our
        // worker, pinned by RT::ConfigureInferenceThread to slice core 0) is
        // the pool's implicit first member. ORT processor ids are 1-BASED.
        if (cpuPartition.logicalProcessors.size() > 1) {
            std::string affinity;
            for (size_t t = 1; t < cpuPartition.logicalProcessors.size(); ++t) {
                if (!affinity.empty()) affinity += ';';
                affinity += std::to_string(cpuPartition.logicalProcessors[t] + 1);
            }
            so.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, affinity.c_str());
            Log::Info("[{}] Intra-op pool pinned inside the slice (1-based ids: '{}')", tag, affinity);
        }

        // Spinning policy: on a DEDICATED slice, busy-waiting between ops burns
        // only cores this session owns and removes wake-up latency (tens of us
        // per parallel section) -> keep it on. Without a partition, or when
        // slices overlap, spinning fights the other sessions -> turn it off.
        const bool dedicatedSlice = !cpuPartition.logicalProcessors.empty() && !cpuPartition.shared;
        so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, dedicatedSlice ? "1" : "0");

        // Real-time intra-op pool: the conv kernels run on these threads, not
        // on the caller of Run(). Create them via the custom factory above so
        // each pool thread starts at TIME_CRITICAL priority and cannot be
        // preempted mid-frame by normal-priority background work.
        so.SetCustomCreateThreadFn(OrtRtCreateThread);
        so.SetCustomJoinThreadFn(OrtRtJoinThread);
        Log::Info("[{}] Intra-op pool: {} TIME_CRITICAL threads, spinning {}.",
            tag, sliceThreads, dedicatedSlice ? "ON" : "OFF");
    }

    return hardwareAccelerated;
}