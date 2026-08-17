#pragma once

#include <vector>

//
// RealTimeConfig
// Centralizes every Windows scheduling knob used to reach hard real-time
// behavior: process priority class, per-thread priority, CPU affinity and the
// partitioning of physical cores among concurrent inference sessions.
//
// Compile-time switch: define RT_DISABLE_AFFINITY (project-wide, e.g.
// C/C++ > Preprocessor > /DRT_DISABLE_AFFINITY) to drop every
// SetThreadAffinityMask call and let the Windows scheduler (and Intel Thread
// Director on hybrid P/E CPUs) place threads freely. Priorities, priority
// class and the 1 ms timer resolution stay active. ComputeCpuPartition then
// returns an EMPTY partition, which the rest of the code already interprets
// as "no pinning": each ORT session sizes its intra-op pool on all compute
// cores (not on a slice) and intra-op spinning is forced OFF, because a
// thread that does not own a core must never busy-wait on it.
//
// Core layout policy (multiple concurrent control points, different models):
//   - Cores OUTSIDE the process affinity mask are never touched: on the
//     production target the Codesys runtime reserves whole cores for the PLC
//     and removes them from the mask; pinning there would fail or disturb
//     the PLC cycle.
//   - The first AVAILABLE physical core is RESERVED for the OS share, the
//     ManagerONNX control thread and the AsyncLogger consumer (all the
//     non-deadline work).
//   - The remaining physical cores are split into disjoint slices, one per
//     worker. Each worker's ORT session sizes its intra-op pool on its own
//     slice and pins the pool threads inside it, so concurrent sessions never
//     compete for the same core (M sessions each spawning a full-machine pool
//     would oversubscribe every core and destroy tail latency).
//   - One logical processor per physical core. The production target runs
//     with hyper-threading DISABLED (Codesys requirement), where this is the
//     identity; on HT-enabled dev machines it skips the SMT siblings, since
//     two compute threads on one physical core slow each other down on dense
//     conv kernels.
//

namespace RT {

// The set of logical processors owned by ONE inference session: the worker
// thread runs on logicalProcessors[0] and the ORT intra-op pool threads on
// the remaining entries (0-based Windows logical processor ids).
struct CpuPartition {
    std::vector<unsigned> logicalProcessors;
    // true when workers outnumber the available cores and slices overlap:
    // real-time guarantees degrade, spinning must stay off
    bool shared = false;
};

// Process-wide setup, call once at startup BEFORE creating any worker:
//  - timeBeginPeriod(1): 1 ms system timer resolution, so WaitForSingleObject
//    wake-ups happen with ~1 ms granularity instead of the default ~15.6 ms
//  - SetPriorityClass(REALTIME_PRIORITY_CLASS): requires the process to run
//    elevated (SeIncreaseBasePriorityPrivilege); without it Windows silently
//    downgrades, so the result is verified and HIGH_PRIORITY_CLASS is used as
//    the explicit fallback.
// Returns true if the REALTIME class was actually granted.
bool EnableRealTimeProcess();

// Releases the 1 ms timer resolution, call once at shutdown
void DisableRealTimeProcess();

// Number of PHYSICAL cores this process may use (topology intersected with
// the process affinity mask). Do NOT estimate this as hardware_concurrency()/2:
// with hyper-threading disabled that heuristic halves the real core count.
unsigned PhysicalCoreCount();

// Splits the machine's physical cores (minus the reserved core 0) among
// workerCount concurrent workers and returns the slice owned by workerIndex
// (0-based). Slices are contiguous and disjoint; when workerCount exceeds the
// available cores each worker falls back to a single (possibly shared) core
// and the partition is flagged shared.
CpuPartition ComputeCpuPartition(unsigned workerIndex, unsigned workerCount);

// Call from INSIDE each inference worker thread:
//  - SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)
//  - SetThreadAffinityMask to the FIRST core of the worker's slice (ORT never
//    sets affinity on the calling thread, which participates in the parallel
//    kernels as the pool's implicit first member; the remaining slice cores
//    are assigned to the pool threads via OrtsessionConfig)
void ConfigureInferenceThread(unsigned workerIndex, const CpuPartition& partition);

// Call from INSIDE background/service threads (logger consumer):
//  - SetThreadPriority(THREAD_PRIORITY_LOWEST)
//  - pinned to the reserved core, away from the inference cores
void ConfigureBackgroundThread();

// Call from the manager/control thread (the one running ManagerONNX::Run):
// pins it to the reserved core at default priority, so it never steals a
// time slice from a compute core mid-frame. Event handling stays responsive
// because the reserved core only hosts low-priority background work.
void ConfigureControlThread();

} // namespace RT
