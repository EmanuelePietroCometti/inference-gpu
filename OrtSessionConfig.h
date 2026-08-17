#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include "RealTimeConfig.h"

// Configures execution providers + session options for one session.
// cpuPartition is the core slice owned by the session (see RealTimeConfig.h):
// on the CPU path the intra-op pool is sized on the slice and its threads are
// pinned inside it. An empty partition falls back to the whole-machine pool
// (single-session behavior).
bool ConfigureOrtSessionOptions(Ort::SessionOptions& so, const std::string& tag,
    const RT::CpuPartition& cpuPartition);
