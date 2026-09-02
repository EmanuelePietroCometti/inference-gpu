#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include "RealTimeConfig.h"
#include "AsyncBatchDetector.h"

namespace ep {
    inline constexpr const char* kNvRtxName = "NvTensorRTRTXExecutionProvider";
    inline constexpr const ORTCHAR_T* kNvRtxLib = ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll");
}

// true if INFGPU_BACKEND=rtx (default false = classic TensorRT).
bool UseTensorRtRtxBackend();

bool ConfigureOrtSessionOptions(Ort::Env& env, Ort::SessionOptions& so,  DetectorConfig cfg_,
    const RT::CpuPartition& cpuPartition, void* userComputeStream = nullptr);
