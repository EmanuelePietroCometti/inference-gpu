#include "OrtSessionConfig.h"
#include "AsyncLogger.h"
#include <onnxruntime_session_options_config_keys.h>
#include <windows.h>
#include <process.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

//
// OrtSessionConfig.cpp
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
// Within ORT_EP_GPU, TENSORRT_NORMAL selects classic TensorRT; otherwise the
// TensorRT-RTX plugin path is used. Both fall back to CUDA, then CPU.
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
// 'cfg_.tag' is only used to prefix log lines (e.g. "Anomaly", "Classification").
// 'cpuPartition' is the core slice owned by this session; empty = whole machine.
bool ConfigureOrtSessionOptions(Ort::Env& env, Ort::SessionOptions& so, DetectorConfig cfg_,
    const RT::CpuPartition& cpuPartition, void* userComputeStream)
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
    // GPU family: classic TensorRT | TensorRT-RTX -> CUDA -> CPU. Runtime selection.
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    so.AddConfigEntry("session.free_dimension_override.batch", "17");

    const OrtApi& api = Ort::GetApi();

#if defined(TENSORRT_NORMAL)
    static const std::string engineCache = std::filesystem::absolute(cfg_.trtEngineCacheDir).string();
    static const std::string timingCache = std::filesystem::absolute(cfg_.trtTimingCacheDir).string();

    try {
        OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

        // Smart pointer with a custom deleter to ensure proper cleanup of the options struct
        std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2*)>
            trtGuard(tensorrt_options, [](OrtTensorRTProviderOptionsV2* p) {
            Ort::GetApi().ReleaseTensorRTProviderOptions(p);
                });

        std::vector<const char*> option_keys = {
            "device_id",
            "trt_max_workspace_size",
            "trt_max_partition_iterations",
            "trt_min_subgraph_size",
            "trt_fp16_enable",
            "trt_dump_subgraphs",
            // Below options are strongly recommended for performance!
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
            "trt_timing_cache_enable",
            "trt_timing_cache_path",
        };

        std::vector<const char*> option_values = {
            "0",                    // device_id
            "2147483648",           // trt_max_workspace_size (2GB)
            "10",                   // trt_max_partition_iterations
            "5",                    // trt_min_subgraph_size
            "1",                    // trt_fp16_enable
            "1",                    // trt_dump_subgraphs
            "1",                    // trt_engine_cache_enable
            engineCache.c_str(),    // trt_engine_cache_path
            "1",                    // trt_timing_cache_enable
            timingCache.c_str(),    // trt_timing_cache_path
        };

        // Apply standard key-value string options
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(), option_keys.size()));

        // Caller's compute stream (similar to the RTX/CUDA branches).
        // Passes the raw void* and implicitly sets has_user_compute_stream=1.
        // Note: Some older ORT docs mistakenly use `cuda_options` in their snippet;
        // the correct struct here is `tensorrt_options`.
        if (userComputeStream) {
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(
                tensorrt_options, "user_compute_stream", userComputeStream));
        }

        // Dump the actual applied options after setting everything up (verification step)
        OrtAllocator* alloc = nullptr;
        Ort::ThrowOnError(api.GetAllocatorWithDefaultOptions(&alloc));

        char* dump = nullptr;
        Ort::ThrowOnError(api.GetTensorRTProviderOptionsAsString(tensorrt_options, alloc, &dump));

        const std::string opts(dump);
        Log::Info("[{}] Actual TRT options ({} chars):", cfg_.tag, opts.size());
        for (size_t i = 0; i < opts.size(); i += 200) {
            Log::Info("  {}", opts.substr(i, 200));
        }
        alloc->Free(alloc, dump);

        // Append the configured TensorRT Execution Provider to the SessionOptions (so)
        so.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
        hardwareAccelerated = true;
        Log::Info("[{}] TensorRT EP appended (FP16 + engine/timing cache).", cfg_.tag);
    }
    catch (const Ort::Exception& e) {
        // Fallback strategy: if TRT fails to initialize, the calling code should try CUDA
        Log::Warning("[{}] TensorRT unavailable, trying CUDA: {}", cfg_.tag, e.what());
    }
#else
    try {
        static const std::string nvCache =
            std::filesystem::absolute("nv_runtime_cache").string();

        std::vector<std::string> okeys = {
            "device_id", "enable_cuda_graph", "nv_runtime_cache_path", "nv_detailed_build_log"
        };
        std::vector<std::string> ovals = {
            "0", "1" /* loopBatch1 rebinds at every Run -> CUDA graph OFF */, nvCache, "0"
        };
        if (userComputeStream) {
            char b[32];
            snprintf(b, sizeof(b), "%llu", (unsigned long long)(uintptr_t)userComputeStream);
            okeys.push_back("user_compute_stream"); ovals.push_back(b);
        }

        // The plugin is registered on the Env (see AsyncBatchDetector); find the device.
        const OrtEpDevice* const* devs = nullptr; size_t n = 0;
        Ort::ThrowOnError(api.GetEpDevices(env, &devs, &n));
        const OrtEpDevice* rtx = nullptr;
        for (size_t i = 0; i < n; ++i)
            if (std::strcmp(api.EpDevice_EpName(devs[i]), ep::kNvRtxName) == 0) { rtx = devs[i]; break; }
        if (!rtx) throw std::runtime_error("NvTensorRTRTX device not found (plugin not registered?)");

        std::vector<const char*> ck, cvv;
        for (auto& s : okeys) ck.push_back(s.c_str());
        for (auto& s : ovals) cvv.push_back(s.c_str());
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_V2(
            so, env, &rtx, 1, ck.data(), cvv.data(), ck.size()));

        hardwareAccelerated = true;
        Log::Info("[{}] TensorRT-RTX EP appended (JIT + runtime cache).", cfg_.tag);
    }
    catch (const std::exception& e) {
        Log::Warning("[{}] TensorRT-RTX unavailable, trying CUDA: {}", cfg_.tag, e.what());
    }
#endif

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
        if (userComputeStream) {
            const char* sk[] = { "has_user_compute_stream" };
            const char* sv[] = { "1" };
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
                cuda, sk, sv, 1));
            Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(
                cuda, "user_compute_stream", userComputeStream));
        }

        so.AppendExecutionProvider_CUDA_V2(*cuda);
        hardwareAccelerated = true;
        Log::Info("[{}] CUDA EP appended.", cfg_.tag);
    }
    catch (const Ort::Exception& e) {
        Log::Warning("[{}] CUDA unavailable, falling back to CPU: {}", cfg_.tag, e.what());
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
            cfg_.tag, targetDevice, targetPrecision, sliceThreads);
    }
    catch (const Ort::Exception& e) {
        // DYNAMIC FALLBACK Level 1: If the Intel GPU initialization fails, attempt to run OpenVINO on the CPU
        Log::Warning("[{}] Failed to initialize OpenVINO on GPU: {}. Attemping fallback to OpenVINO CPU...", cfg_.tag, e.what());

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
                cfg_.tag, targetDevice, targetPrecision, sliceThreads);
        }
        catch (const Ort::Exception& e_cpu) {
            // FINAL FALLBACK Level 2: Entirely disable OpenVINO and revert to native ONNX Runtime CPU EP
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            hardwareAccelerated = false;
            Log::Error("[{}] OpenVINO completely unavailable. Falling back to native ONNX Runtime CPU: {}", cfg_.tag, e_cpu.what());
        }
    }

#elif defined(ORT_EP_CPU)
    // CPU build: built-in ORT CPU EP
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Log::Info("[{}] CPU build: using the default ONNX Runtime CPU EP.", cfg_.tag);

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
            Log::Info("[{}] Intra-op pool pinned inside the slice (1-based ids: '{}')", cfg_.tag, affinity);
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
            cfg_.tag, sliceThreads, dedicatedSlice ? "ON" : "OFF");
    }

    return hardwareAccelerated;
}