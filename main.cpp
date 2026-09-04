// inference_gpu : MMF-driven batch-17 anomaly inference provider.
//
// Mirrors ONNX_inference/main.cpp: bring up the async logger and the real-time
// scheduling policy, then run the manager loop that services MMF configuration
// and termination requests from the external controller.

#include "ManagerGPU.h"
#include "AsyncLogger.h"
#include "RealTimeConfig.h"
#include "stdlib.h"

int main()
{
	// Disable lazy loading of CUDA modules to guarantee that the first inference call is not delayed by a lazy load.
	_putenv_s("CUDA_MODULE_LOADING", "LAZY"); 
    AsyncLogger::Instance().Start();

    RT::EnableRealTimeProcess();
    RT::ConfigureControlThread();

    try {
        ManagerGPU manager;
        manager.Run();
    }
    catch (const std::exception& e) {
        Log::Error("Fatal exception: {}", e.what());
    }

    RT::DisableRealTimeProcess();
    AsyncLogger::Instance().Shutdown();
    return 0;
}
