#pragma once

#include "IPC_Shared.h"
#include "AsyncBatchDetector.h"
#include "PerformanceMetrics.h"
#include "RealTimeConfig.h"

#include <windows.h>
#include <thread>
#include <atomic>
#include <memory>

//
// WorkerGPU
// Owns the batch-17 pipeline for ONE control point and bridges it to MMF:
//   - opens the per-point MMFs (input ring, result-image, result-block) and
//     the local mutex/events;
//   - an ingest thread drains the input ring on eventReady and feeds pushBatch;
//   - the pipeline sink writes overlays + scores/statuses back into the output
//     MMFs and signals eventResults, correlating by frameId.
// Structurally mirrors ONNX_inference/WorkerONNX, extended for the batch wire.
//
class WorkerGPU {
public:
    WorkerGPU(PTcontrolPoint pPoint, unsigned workerIndex, unsigned workerCount);
    ~WorkerGPU();

    void Start();
    void Stop();
    void MarkAsConfigured();
    void MarkAsError();

private:
    PTcontrolPoint controlPointData;
    unsigned workerIndex;
    RT::CpuPartition cpuPartition;

    PerformanceMetrics metrics_;
    std::unique_ptr<AsyncBatchDetector> detector_;

    std::thread ingestThread_;
    std::atomic<bool> running_{ false };

    // Sync handles
    HANDLE hLocalMutex = NULL;
    HANDLE hEventReady = NULL;
    HANDLE hEventResults = NULL;

    // MMF input ring
    HANDLE hMMFImage = NULL;
    LPVOID pImageBuffer = NULL;
    // MMF output overlays
    HANDLE hMMFResImage = NULL;
    LPVOID pResImageBuffer = NULL;
    // MMF output result block
    HANDLE hMMFResult = NULL;
    LPVOID pResultBuffer = NULL;

    void InitializeLocalPC();
    void IngestLoop();
    void PublishResult(BatchOutput&& out);
};
