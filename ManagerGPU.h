#pragma once

#include "IPC_Shared.h"
#include "WorkerGPU.h"

#include <windows.h>
#include <map>
#include <memory>

//
// ManagerGPU
// Global IPC coordinator for the batch inference_gpu provider. 1:1 in structure
// with ONNX_inference/ManagerONNX: it owns the global CONTROLPOINTLIST MMF and
// the LIST* mutex/events, waits for a configuration trigger, and spawns one
// WorkerGPU (which owns a batch-17 pipeline) per ANOMALY control point.
//
class ManagerGPU {
public:
    ManagerGPU();
    ~ManagerGPU();

    void Run();

private:
    HANDLE hMapFile = NULL;
    PTcontrolPointsList pSharedList = NULL;
    HANDLE hGlobalMutex = NULL;
    HANDLE hEventTrigger = NULL;
    HANDLE hEventAck = NULL;

    std::map<DWORD, std::unique_ptr<WorkerGPU>> activeWorkers;

    bool InitializeIPC();
    void HandleConfiguration();
    void HandleTermination();
};
