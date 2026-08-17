#include "ManagerGPU.h"
#include "AsyncLogger.h"
#include <stdexcept>

ManagerGPU::ManagerGPU()
{
    if (!InitializeIPC()) {
        Log::Error("### ERROR: Global IPC initialization failed!");
        throw std::runtime_error("Global IPC initialization failed");
    }
}

ManagerGPU::~ManagerGPU()
{
    if (pSharedList) UnmapViewOfFile(pSharedList);
    if (hMapFile) CloseHandle(hMapFile);
    if (hGlobalMutex) CloseHandle(hGlobalMutex);
    if (hEventTrigger) CloseHandle(hEventTrigger);
    if (hEventAck) CloseHandle(hEventAck);
}

void ManagerGPU::Run()
{
    Log::Info("GPU manager started. Waiting for signals...");

    bool isRunning = true;
    while (isRunning) {
        WaitForSingleObject(hEventTrigger, INFINITE);

        // WAIT_ABANDONED still grants ownership: handle it like WAIT_OBJECT_0.
        DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
        if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
            ReleaseMutex(hGlobalMutex);
        }

        if (pSharedList->state == ListState::QUIT) {
            HandleTermination();
            isRunning = false;
        }
        else if (pSharedList->state == ListState::UPDATE_PENDING) {
            HandleConfiguration();
        }
    }
    Log::Info("GPU manager correctly terminated.");
}

bool ManagerGPU::InitializeIPC()
{
    hGlobalMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
    hEventTrigger = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTTRIGGER"));
    hEventAck = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));

    hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
        sizeof(controlPointsList), L"CONTROLPOINTLIST");
    if (hMapFile == NULL) {
        Log::Error("### ERROR: CreateFileMapping failed with code: {}", GetLastError());
        return false;
    }

    pSharedList = (PTcontrolPointsList)MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));
    if (pSharedList == NULL) {
        DWORD dwErr = GetLastError();
        CloseHandle(hMapFile); hMapFile = NULL;
        Log::Error("### ERROR: MapViewOfFile failed with code: {}", dwErr);
        return false;
    }

    return (hGlobalMutex && hEventTrigger && hEventAck);
}

void ManagerGPU::HandleConfiguration()
{
    Log::Info(">> Configuration request received (UPDATE_PENDING)!");

    activeWorkers.clear();
    bool configSuccess = true;

    DWORD i = 0;
    try {
        for (i = 0; i < pSharedList->numPunti; i++) {
            DWORD idPunto = pSharedList->points[i].idPunto;
            activeWorkers[idPunto] = std::make_unique<WorkerGPU>(
                &pSharedList->points[i], i, pSharedList->numPunti);
            activeWorkers[idPunto]->Start();
            activeWorkers[idPunto]->MarkAsConfigured();
        }
    }
    catch (const std::exception& e) {
        DWORD idPunto = pSharedList->points[i].idPunto;
        Log::Error("### ERROR DURING CONFIGURATION: {}", e.what());

        auto it = activeWorkers.find(idPunto);
        if (it != activeWorkers.end() && it->second) {
            it->second->MarkAsError();
        }
        else {
            pSharedList->points[i].status = PointState::ERROR_DETECTED;
            Log::Error("Worker {} ERROR_DETECTED (construction failed)", idPunto);
            activeWorkers.erase(idPunto);
        }
        configSuccess = false;
    }

    DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
    if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
        pSharedList->state = configSuccess ? ListState::CONFIGURED : ListState::ERROR_DETECTED;
        ReleaseMutex(hGlobalMutex);
    }

    SetEvent(hEventAck);
    Log::Info(">> Configuration completed. Ack sent!");
}

void ManagerGPU::HandleTermination()
{
    Log::Info(">> Shutdown request received (QUIT)!");
    activeWorkers.clear();
    SetEvent(hEventAck);
    Log::Info(">> Shutdown completed. Ack sent!");
}
