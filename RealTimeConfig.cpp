#include "RealTimeConfig.h"
#include "AsyncLogger.h"

#include <windows.h>
#include <timeapi.h>
#include <algorithm>
#include <thread>

#pragma comment(lib, "winmm.lib")

#define RT_DISABLE_AFFINITY 1

namespace {

// Enumerates the PHYSICAL cores this process is actually allowed to use and
// returns one 0-based logical processor id per core.
//
// Two filters are applied:
//  - Physical topology (GetLogicalProcessorInformationEx): one logical
//    processor per core. On the production target (Codesys machine) SMT/
//    hyper-threading is disabled in the BIOS, so core == logical processor
//    and this is a no-op; the code stays correct on HT-enabled dev machines.
//  - Process affinity mask (GetProcessAffinityMask): the Codesys runtime
//    reserves whole cores for the PLC and removes them from the mask. Any
//    core outside the mask MUST be skipped: SetThreadAffinityMask onto it
//    would fail, and contending it would disturb the PLC cycle.
//
// NOTE: limited to processor group 0, i.e. at most 64 logical processors;
// beyond that SetThreadGroupAffinity and group-aware masks are required.
std::vector<unsigned> AvailableComputeCores()
{
    std::vector<unsigned> cores;

    DWORD_PTR processMask = 0;
    DWORD_PTR systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask) || processMask == 0) {
        processMask = ~static_cast<DWORD_PTR>(0); // query failed: assume all allowed
    }

    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && length > 0) {
        std::vector<char> buffer(length);
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            for (char* p = buffer.data(); p < buffer.data() + length;) {
                auto* rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
                if (rec->Relationship == RelationProcessorCore && rec->Processor.GroupCount >= 1
                    && rec->Processor.GroupMask[0].Group == 0) {
                    // Only the part of the core usable by THIS process: a core
                    // fully reserved (e.g. by the Codesys runtime) drops out here
                    KAFFINITY mask = rec->Processor.GroupMask[0].Mask & processMask;
                    unsigned bit = 0;
                    while (mask != 0 && (mask & 1) == 0) { mask >>= 1; ++bit; }
                    if (mask != 0 && bit < 64) {
                        cores.push_back(bit);
                    }
                }
                p += rec->Size;
            }
        }
    }

    if (cores.empty()) {
        // Fallback: no topology info, treat every allowed logical processor as
        // a core (exact on the HT-disabled production target)
        const unsigned logical = (std::max)(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < logical && i < 64; ++i) {
            if ((processMask >> i) & 1) {
                cores.push_back(i);
            }
        }
    }

    std::sort(cores.begin(), cores.end());
    return cores;
}

} // namespace

namespace RT {

bool EnableRealTimeProcess()
{
    // 1 ms scheduler/timer granularity: without this, event wake-ups and any
    // timed wait quantize to ~15.6 ms, which alone breaks a 4-5 ms budget
    timeBeginPeriod(1);

    HANDLE hProcess = GetCurrentProcess();

    // REALTIME_PRIORITY_CLASS puts the process above every normal system
    // activity. Windows only grants it to elevated processes: when the
    // privilege is missing the call "succeeds" but the class is silently
    // downgraded, so the actual class must be read back and verified.
    if (SetPriorityClass(hProcess, REALTIME_PRIORITY_CLASS)) {
        Log::Info("[RT] Process priority class: REALTIME");
        return true;
    }

    DWORD rtError = GetLastError();
    Log::Warning("[RT] REALTIME class not granted (Error: {}). Falling back to HIGH priority class.", rtError);

    if (SetPriorityClass(hProcess, HIGH_PRIORITY_CLASS)) {
        Log::Info("[RT] Process priority class: HIGH");
        return true;
    }
    DWORD highError = GetLastError();
    Log::Error("### ERROR: Impossible setting REALTIME or HIGH priority class. Final Error: {}", highError);

    return false;
}

void DisableRealTimeProcess()
{
    timeEndPeriod(1);
}

unsigned PhysicalCoreCount()
{
    return static_cast<unsigned>(AvailableComputeCores().size());
}

CpuPartition ComputeCpuPartition(unsigned workerIndex, unsigned workerCount)
{
    CpuPartition part;
#if RT_DISABLE_AFFINITY
    // Affinity disabled at build time: return an empty partition. Downstream
    // this means no SetThreadAffinityMask, ORT pool sized on all compute
    // cores, intra-op spinning OFF (see RealTimeConfig.h).
    (void)workerIndex;
    (void)workerCount;
    Log::Info("[RT] RT_DISABLE_AFFINITY: worker {} not pinned, thread placement left to the OS scheduler",
        workerIndex);
    return part;
#else
    if (workerCount == 0) workerCount = 1;

    std::vector<unsigned> cores = AvailableComputeCores();

    // Reserve the first AVAILABLE physical core for the OS share, the manager
    // and the logger (not hardcoded core 0: on a Codesys machine the low cores
    // may belong to the PLC runtime and be outside the process affinity mask)
    if (cores.size() > 1) {
        cores.erase(cores.begin());
    }
    const unsigned available = static_cast<unsigned>(cores.size());

    if (workerCount >= available) {
        // More workers than compute cores: give each worker one core, wrapping
        // around. Slices overlap, so downstream tuning must stay conservative.
        part.logicalProcessors.push_back(cores[workerIndex % available]);
        part.shared = (workerCount > available);
        if (part.shared) {
            Log::Warning("[RT] {} workers on {} compute cores: worker {} SHARES core {} "
                "(real-time guarantees degraded, reduce concurrent control points)",
                workerCount, available, workerIndex, part.logicalProcessors.front());
        }
        return part;
    }

    // Contiguous disjoint slices; the remainder cores go to the first workers
    const unsigned base = available / workerCount;
    const unsigned remainder = available % workerCount;
    const unsigned begin = workerIndex * base + (std::min)(workerIndex, remainder);
    const unsigned count = base + (workerIndex < remainder ? 1u : 0u);

    for (unsigned k = 0; k < count; ++k) {
        part.logicalProcessors.push_back(cores[begin + k]);
    }

    Log::Info("[RT] Worker {} owns {} physical core(s): first logical processor {}",
        workerIndex, count, part.logicalProcessors.front());
    return part;
#endif // RT_DISABLE_AFFINITY
}

void ConfigureInferenceThread(unsigned workerIndex, const CpuPartition& partition)
{
    HANDLE hThread = GetCurrentThread();

    // TIME_CRITICAL: base priority 15 in HIGH class, 31 in REALTIME class.
    // The worker must preempt everything else the moment the frame event fires.
    if (SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL)) {
        Log::Info("[RT] Worker {}: SetThreadPriority to THREAD_PRIORITY_TIME_CRITICAL", workerIndex);
    }
    else {
        DWORD rtError = GetLastError();
        Log::Error("[RT] Worker {}: SetThreadPriority failed with code {}", workerIndex, rtError);
    }
#if not RT_DISABLE_AFFINITY
    if (partition.logicalProcessors.empty()) {
        // Expected under RT_DISABLE_AFFINITY: priority is raised, placement
        // stays with the OS scheduler / Thread Director.
        Log::Warning("[RT] Worker {}: empty CPU partition, affinity pinning skipped", workerIndex);
        return;
    }
    // The worker runs on the FIRST core of its slice; the ORT pool threads of
    // its session are pinned to the remaining slice cores (OrtsessionConfig)
    const unsigned core = partition.logicalProcessors.front();

    // NOTE: This bitwise shift is only safe if 'core' < 64 (on x64 architectures).
    // For systems with >64 logical processors, SetThreadGroupAffinity must be used instead.
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << core;

    if (SetThreadAffinityMask(hThread, mask) == 0) {
        Log::Error("[RT] Worker {}: SetThreadAffinityMask(core {}) failed with code {}",
            workerIndex, core, GetLastError());
    }
    else {
        Log::Info("[RT] Worker {}: TIME_CRITICAL, pinned to core {} (slice of {} core(s){})",
            workerIndex, core, partition.logicalProcessors.size(),
            partition.shared ? ", SHARED" : "");
    }
#endif
}
void ConfigureBackgroundThread()
{
    HANDLE hThread = GetCurrentThread();

    // Background service work (console I/O) runs at the lowest priority on the
    // reserved core: it only executes when the compute cores have nothing
    // pending. The reserved core is the first AVAILABLE one, not hardcoded 0
    // (which may belong to the Codesys runtime and be outside our affinity mask).
    SetThreadPriority(hThread, THREAD_PRIORITY_LOWEST);

#if !RT_DISABLE_AFFINITY
    const std::vector<unsigned> cores = AvailableComputeCores();
    if (cores.size() > 2) {
        SetThreadAffinityMask(hThread, static_cast<DWORD_PTR>(1) << cores.front());
    }
#else
    Log::Info("[RT] Backgorund thread not pinned, thread placement left to the OS scheduler");
#endif
}

void ConfigureControlThread()
{
#if !RT_DISABLE_AFFINITY
    // Default priority is kept: the manager only waits on IPC events. Pinning
    // it to the reserved core guarantees it is never scheduled onto a compute
    // core in the middle of a frame.
    const std::vector<unsigned> cores = AvailableComputeCores();
    if (cores.size() > 2) {
        const unsigned core = cores.front();
        if (SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << core) != 0) {
            Log::Info("[RT] Control thread pinned to reserved core {}", core);
        }
    }
#else
    Log::Info("[RT] Control thread not pinned, thread placement left to the OS scheduler");
#endif
}

} // namespace RT
