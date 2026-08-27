#include "WorkerGPU.h"
#include "AsyncLogger.h"

#include <fmt/core.h>
#include <fmt/xchar.h>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

namespace {
// CreateFileMapping split of a 64-bit size into (high, low) DWORDs
inline HANDLE CreateSharedMapping(SIZE_T bytes, const std::wstring& name) {
    DWORD hi = static_cast<DWORD>((static_cast<unsigned long long>(bytes) >> 32) & 0xFFFFFFFFull);
    DWORD lo = static_cast<DWORD>(static_cast<unsigned long long>(bytes) & 0xFFFFFFFFull);
    return CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, hi, lo, name.c_str());
}
} // namespace

WorkerGPU::WorkerGPU(PTcontrolPoint pPoint, unsigned workerIndex_, unsigned workerCount)
    : controlPointData(pPoint),
      workerIndex(workerIndex_),
      cpuPartition(RT::ComputeCpuPartition(workerIndex_, workerCount))
{
    InitializeLocalPC();
}

WorkerGPU::~WorkerGPU()
{
    Stop();
    detector_.reset();
    if (pImageBuffer)    UnmapViewOfFile(pImageBuffer);
    if (pResImageBuffer) UnmapViewOfFile(pResImageBuffer);
    if (pResultBuffer)   UnmapViewOfFile(pResultBuffer);
    if (hMMFImage)    CloseHandle(hMMFImage);
    if (hMMFResImage) CloseHandle(hMMFResImage);
    if (hMMFResult)   CloseHandle(hMMFResult);
    if (hLocalMutex)  CloseHandle(hLocalMutex);
    if (hEventReady)  CloseHandle(hEventReady);
    if (hEventResults)CloseHandle(hEventResults);
}

void WorkerGPU::Start()
{
    running_ = true;
    ingestThread_ = std::thread(&WorkerGPU::IngestLoop, this);
}

void WorkerGPU::Stop()
{
    if (!running_.exchange(false)) {
        if (ingestThread_.joinable()) ingestThread_.join();
        return;
    }
    DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
    if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED) {
        controlPointData->status = PointState::QUIT;
        ReleaseMutex(hLocalMutex);
    }
    if (hEventReady) SetEvent(hEventReady); // wake the ingest thread
    if (ingestThread_.joinable()) ingestThread_.join();
    Log::Info(">> Worker {} stopped!", controlPointData->idPunto);
}

void WorkerGPU::MarkAsConfigured()
{
    DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
    if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED) {
        controlPointData->status = PointState::CONFIGURED;
        Log::Info("Worker {} CONFIGURED", controlPointData->idPunto);
        ReleaseMutex(hLocalMutex);
    }
}

void WorkerGPU::MarkAsError()
{
    DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
    if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED) {
        controlPointData->status = PointState::ERROR_DETECTED;
        Log::Error("Worker {} ERROR_DETECTED", controlPointData->idPunto);
        ReleaseMutex(hLocalMutex);
    }
}

void WorkerGPU::InitializeLocalPC()
{
    if (controlPointData->batchSize == 0) controlPointData->batchSize = kBatchSize;
    // The controller imposes the ring depth; only fall back if it left it unset.
    if (controlPointData->ringSlots == 0) controlPointData->ringSlots = 3;

    hLocalMutex   = OpenMutex(MUTEX_ALL_ACCESS, FALSE, controlPointData->mutexName);
    hEventReady   = OpenEvent(EVENT_ALL_ACCESS, FALSE, controlPointData->eventReadyName);
    hEventResults = OpenEvent(EVENT_ALL_ACCESS, FALSE, controlPointData->resultsEventName);

    // ---- input ring MMF ----
    const SIZE_T inBytes = IpcInputMmfBytes(*controlPointData);
    std::wstring inName = fmt::format(L"MMF_{}_IMAGE", controlPointData->idPunto);
    hMMFImage = CreateSharedMapping(inBytes, inName);
    if (!hMMFImage) throw std::runtime_error("Failed to create input MMF");
    pImageBuffer = MapViewOfFile(hMMFImage, FILE_MAP_READ, 0, 0, inBytes);
    if (!pImageBuffer) throw std::runtime_error("MapViewOfFile input failed");

    // ---- output overlay MMF ----
    const SIZE_T outImgBytes = IpcResultImageMmfBytes(*controlPointData);
    std::wstring outName = fmt::format(L"MMF_{}_RESIMAGE", controlPointData->idPunto);
    hMMFResImage = CreateSharedMapping(outImgBytes, outName);
    if (!hMMFResImage) throw std::runtime_error("Failed to create result-image MMF");
    pResImageBuffer = MapViewOfFile(hMMFResImage, FILE_MAP_WRITE, 0, 0, outImgBytes);
    if (!pResImageBuffer) throw std::runtime_error("MapViewOfFile result-image failed");

    // ---- output result-block MMF (ring of ringSlots blocks) ----
    const SIZE_T resBytes = IpcResultBlockMmfBytes(*controlPointData);
    std::wstring resName = fmt::format(L"MMF_{}_RESULT", controlPointData->idPunto);
    hMMFResult = CreateSharedMapping(resBytes, resName);
    if (!hMMFResult) throw std::runtime_error("Failed to create result MMF");
    pResultBuffer = MapViewOfFile(hMMFResult, FILE_MAP_WRITE, 0, 0, resBytes);
    if (!pResultBuffer) throw std::runtime_error("MapViewOfFile result failed");

    if (controlPointData->inferenceType != InferenceType::ANOMALY) {
        throw std::runtime_error("WorkerGPU only supports ANOMALY inference type");
    }

    // ---- build the batch pipeline ----
    DetectorConfig dc;
    dc.modelPath = std::wstring(controlPointData->pathModello);
    dc.imgW = (int)controlPointData->sizeX;
    dc.imgH = (int)controlPointData->sizeY;
    dc.channels = (int)(controlPointData->bpp / 8);
    dc.batchSize = (int)controlPointData->batchSize;
    // Number of concurrent ORT/TensorRT sessions (GPU streams). Default 2, but
    // overridable at runtime for experiments via the INFGPU_SESSIONS env var
    // (e.g. set INFGPU_SESSIONS=3), clamped to [1, 8]. No rebuild needed.
	dc.numInfThreads = (int)controlPointData->inferenceThreads;
    dc.partition = cpuPartition;
    dc.trtCacheDir = "trt_engine_cache";

    metrics_.batch_size = dc.batchSize;
    detector_ = std::make_unique<AsyncBatchDetector>(
        dc,
        [this](BatchOutput&& o) { PublishResult(std::move(o)); },
        metrics_);

    Log::Info("Worker {} initialized (ANOMALY, {}x{}x{}, batch {}, {} sessions).",
        controlPointData->idPunto, dc.imgW, dc.imgH, dc.channels, dc.batchSize, dc.numInfThreads);
}

void WorkerGPU::IngestLoop()
{
    RT::ConfigureInferenceThread(workerIndex, cpuPartition);
    Log::Info(">> Ingest thread started for control point: {}", controlPointData->idPunto);

    auto* header = reinterpret_cast<const batchInputHeader*>(pImageBuffer);
    const auto* base = reinterpret_cast<const unsigned char*>(pImageBuffer);
    const LONG ringSlots = (LONG)IpcRingSlots(*controlPointData); // controller-imposed
    LONG consumedSeq = 0;

    while (running_.load()) {
        WaitForSingleObject(hEventReady, INFINITE);

        // state check
        DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
        bool quit = false;
        if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED) {
            quit = (controlPointData->status == PointState::QUIT ||
                    controlPointData->status == PointState::UPDATE_PENDING);
            ReleaseMutex(hLocalMutex);
        }
        if (quit || !running_.load()) break;

        // Drain every published-but-unconsumed batch (auto-reset events coalesce)
        LONG seq = header->publishSeq; // aligned LONG read is atomic on x64
        if (seq - consumedSeq > ringSlots) {
            LONG skipped = (seq - ringSlots) - consumedSeq;
            Log::Warning("Worker {} ingest lagged: skipping {} batch(es) (producer lapped the ring)",
                controlPointData->idPunto, skipped);
            consumedSeq = seq - ringSlots;
        }
        while (consumedSeq < seq) {
            const DWORD slot = (DWORD)(consumedSeq % ringSlots);
            const unsigned char* slotPtr = base + IpcInputSlotOffset(*controlPointData, slot);
            const int frameId = (int)header->frameId[slot];
            detector_->pushBatch(slotPtr, frameId, (int)consumedSeq);
            ++consumedSeq;
        }
    }
    Log::Info(">> Ingest thread stopped for control point: {}", controlPointData->idPunto);
}

void WorkerGPU::PublishResult(BatchOutput&& o)
{
    // Lock-free ring publish: several post-stage threads may publish different
    // batches concurrently, but each writes a DISTINCT slot (seq % N), and the
    // producer's N-in-flight backpressure guarantees the slot is free. No mutex
    // needed; 'state' is written LAST with a release barrier so a reader that
    // sees RESULT_READY sees a fully written block + overlay.
    const DWORD slot = (DWORD)(o.seq % (int)IpcRingSlots(*controlPointData));
    const DWORD B = controlPointData->batchSize;

    // Overlay batch -> its ring slot
    if (pResImageBuffer && o.overlay) {
        auto* dst = reinterpret_cast<unsigned char*>(pResImageBuffer)
            + IpcResultImageSlotOffset(*controlPointData, slot);
        std::memcpy(dst, o.overlay.get(), o.overlayBytesPerImage * (size_t)B);
    }

    // Result block -> its ring slot
    auto* rb = reinterpret_cast<batchResultBlock*>(
        reinterpret_cast<unsigned char*>(pResultBuffer) + (size_t)slot * sizeof(batchResultBlock));
    rb->seq = (DWORD)o.seq;
    rb->frameId = (DWORD)o.frame_id;
    rb->batchSize = B;
    rb->mapH = (DWORD)o.map_h;
    rb->mapW = (DWORD)o.map_w;
    rb->overlayBytesPerImage = (DWORD)o.overlayBytesPerImage;
    for (DWORD k = 0; k < B && k < kBatchSize; ++k) {
        rb->results[k].anomalyScore = o.scores[k];
        rb->results[k].status = (o.statuses[k] != 0) ? PatchStatus::REJECT : PatchStatus::OK;
    }

    MemoryBarrier();                              // publish payload before the flag
    rb->state = InferenceState::RESULT_READY;     // release: reader keys on this
    SetEvent(hEventResults);
}
