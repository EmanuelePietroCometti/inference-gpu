cmake_minimum_required (VERSION 3.17)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("inference-gpu")

# UNICODE build: TCHAR == wchar_t so the MMF struct layout matches ONNX_inference
# and the test controller, and Win32 calls resolve to the wide (...W) versions.
# NOMINMAX: stop <windows.h> from defining min()/max() macros that clobber
# std::min / std::max (used in ContractMetadata / AsyncBatchDetector / OpenCV).
add_compile_definitions(UNICODE _UNICODE NOMINMAX)

# =====================================================================
# TARGET ARCHITECTURE
# =====================================================================
set(TENSORRT_TARGET_SM "89" CACHE STRING "Target Compute Capability for TensorRT DLLs")
message(STATUS "Installation will target TensorRT SM ${TENSORRT_TARGET_SM}")

# =====================================================================
# RELATIVE PATHS CONFIGURATION
# =====================================================================
# Define a base directory for all SDKs starting from the current folder.
# By default it expects an "SDK" folder in the same directory as this CMakeLists.txt.
# It can be overridden via CLI: cmake -DSDK_ROOT="C:/another/path/SDK" ..
set(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/SDK" CACHE PATH "Base directory for all SDK dependencies")
message(STATUS "Using SDK_ROOT: ${SDK_ROOT}")

# =====================================================================
# PATHS & DEPENDENCIES
# =====================================================================

set(ONNX_ROOT "${SDK_ROOT}/onnxruntime-win-x64-gpu_cuda13-1.29.0")
include_directories("${ONNX_ROOT}/include")
link_directories("${ONNX_ROOT}/lib")

set(OpenCV_DIR "${SDK_ROOT}/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(FMT_ROOT "${SDK_ROOT}/fmt-11.1.4")
include_directories("${FMT_ROOT}/include")

set(TENSORRT_ROOT "${SDK_ROOT}/TensorRT-10.16.1.11")

set(TENSORRT_RTX_ROOT "${SDK_ROOT}/TensorRT-RTX-1.5.0.114")
set(NV_RTX_EP_ROOT "${SDK_ROOT}/TensorRT-RTX-EP-ABI-v0.3.0-cu13")

# CUDA relies on standard system environment variables, which is already PC-independent
find_package(CUDAToolkit REQUIRED)
set(CUDA_BIN_PATH "$ENV{CUDA_PATH}/bin/x64")

# =====================================================================
# TENSORRT DLL FILTERING LOGIC (for installation only)
# =====================================================================
file(GLOB TENSORRT_ALL_DLLS "${TENSORRT_ROOT}/bin/*.dll")

# Filter the DLLs
set(TENSORRT_FILTERED_DLLS "")
foreach(DLL_PATH IN LISTS TENSORRT_ALL_DLLS)
    get_filename_component(DLL_NAME "${DLL_PATH}" NAME)
    
    if(DLL_NAME MATCHES "nvinfer_builder_resource_sm([0-9]+)")
        if(CMAKE_MATCH_1 STREQUAL TENSORRT_TARGET_SM)
            list(APPEND TENSORRT_FILTERED_DLLS "${DLL_PATH}")
        endif()
    else()
        list(APPEND TENSORRT_FILTERED_DLLS "${DLL_PATH}")
    endif()
endforeach()

file(GLOB OPENCV_ALL_DLLS "${OpenCV_DIR}/x64/vc16/bin/*.dll")
set(OPENCV_RELEASE_DLLS "")
foreach(DLL_PATH IN LISTS OPENCV_ALL_DLLS)
    get_filename_component(DLL_NAME "${DLL_PATH}" NAME)
    if(NOT DLL_NAME MATCHES "d\\.dll$") 
        list(APPEND OPENCV_RELEASE_DLLS "${DLL_PATH}")
    endif()
endforeach()

file(GLOB CUDA_ESSENTIAL_DLLS 
    "${CUDA_BIN_PATH}/cudart64_*.dll"           
    "${CUDA_BIN_PATH}/cublas64_*.dll"           
    "${CUDA_BIN_PATH}/cublasLt64_*.dll"         
    "${CUDA_BIN_PATH}/cudnn*.dll"
    "${CUDA_BIN_PATH}/cufft64_*.dll"            
    "${CUDA_BIN_PATH}/zlibwapi.dll"
)

file(GLOB ONNX_DLLS "${ONNX_ROOT}/lib/*.dll")
file(GLOB FMT_RELEASE_DLLS "${FMT_ROOT}/build/bin/Release/fmt*.dll")
file(GLOB TENSORRT_RTX_DLLS "${TENSORRT_RTX_ROOT}/bin/*.dll")


# =====================================================================
# TARGET DEFINITION
# =====================================================================

# Modular MMF-driven batch provider (replaces the old inference-gpu.cpp monolith,
# which is kept on disk for reference but no longer compiled).
set(INFERENCE_GPU_SOURCES
    "main.cpp"
    "ManagerGPU.cpp"
    "WorkerGPU.cpp"
    "AsyncBatchDetector.cpp"
    "ContractMetadata.cpp"
    "OrtSessionConfig.cpp"
    "RealTimeConfig.cpp"
    "AsyncLogger.cpp"
)

add_executable(inference-gpu ${INFERENCE_GPU_SOURCES})

# Select the ONNX Runtime execution-provider stack (ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_cpu)
target_compile_definitions(inference-gpu PRIVATE ORT_EP_GPU)

if (MSVC)
    target_compile_options(inference-gpu PRIVATE /utf-8)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET inference-gpu PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(inference-gpu
    onnxruntime
    ${OpenCV_LIBS}
    debug "${FMT_ROOT}/build/lib/Debug/fmtd.lib"
    optimized "${FMT_ROOT}/build/lib/Release/fmt.lib"
    CUDA::cudart
)

# NOTE: the test controller (inference-gpu-test) has moved to its own repo
# (inference_gpu_test) together with a byte-identical copy of IPC_Shared.h.

# =====================================================================
# POST-BUILD COPY COMMANDS
# =====================================================================

add_custom_command(TARGET inference-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${ONNX_ROOT}/lib" 
    "$<TARGET_FILE_DIR:inference-gpu>"
    COMMENT "Copying ONNX Runtime DLLs to output directory..."
)

add_custom_command(TARGET inference-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${OpenCV_DIR}/x64/vc16/bin"
    "$<TARGET_FILE_DIR:inference-gpu>"
    COMMENT "Copying OpenCV DLLs to output directory..."
)

# =====================================================================
# DYNAMIC POST-BUILD & INSTALL RULES based on target definition
# =====================================================================
# Use a CMake variable to control the logic flow
set(USE_TENSORRT_RTX OFF) # Set to OFF to use standard TensorRT

if(USE_TENSORRT_RTX)

    # Set TENSORRT_NORMAL or TENSORRT_RTX for conditional compilation in the code
    target_compile_definitions(inference-gpu PRIVATE TENSORRT_RTX)
    
    # Copy ONLY TensorRT-RTX and its specific EP plugin
    add_custom_command(TARGET inference-gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TENSORRT_RTX_ROOT}/bin"
        "$<TARGET_FILE_DIR:inference-gpu>"
        COMMENT "Copying TensorRT-RTX DLLs to output directory..."
    )
    add_custom_command(TARGET inference-gpu POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy_directory
         "${TENSORRT_RTX_ROOT}/lib"
         "$<TARGET_FILE_DIR:inference-gpu>"
         COMMENT "Copying TensorRT-RTX lib DLLs (tensorrt_plugins.dll etc.)..."
    )
    add_custom_command(TARGET inference-gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TENSORRT_RTX_ROOT}/bin"
        "$<TARGET_FILE_DIR:inference-gpu>"
        COMMENT "Copying TensorRT-RTX DLLs to output directory..."
    )
    add_custom_command(TARGET inference-gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${NV_RTX_EP_ROOT}"
        "$<TARGET_FILE_DIR:inference-gpu>"
        COMMENT "Copying TensorRT-RTX EP plugin to output directory..."
    )

    file(GLOB NV_RTX_EP_FILES "${NV_RTX_EP_ROOT}/*.dll")
    
    # Overwrite ALL_RUNTIME_DLLS to exclude standard TRT
    set(ALL_RUNTIME_DLLS 
        ${TENSORRT_RTX_DLLS}
        ${NV_RTX_EP_FILES}
        ${OPENCV_RELEASE_DLLS}
        ${CUDA_ESSENTIAL_DLLS}
        ${ONNX_DLLS}
        ${FMT_RELEASE_DLLS}
    )
else()
    target_compile_definitions(inference-gpu PRIVATE TENSORRT_NORMAL)
    
    # Copy ONLY standard TensorRT
    add_custom_command(TARGET inference-gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TENSORRT_ROOT}/bin" 
        "$<TARGET_FILE_DIR:inference-gpu>"
        COMMENT "Copying standard TensorRT DLLs to output directory..."
    )
    
    # Overwrite ALL_RUNTIME_DLLS to exclude RTX TRT
    set(ALL_RUNTIME_DLLS 
        ${TENSORRT_FILTERED_DLLS}
        ${OPENCV_RELEASE_DLLS}
        ${CUDA_ESSENTIAL_DLLS}
        ${ONNX_DLLS}
        ${FMT_RELEASE_DLLS}
    )
endif()

# =====================================================================
# INSTALLATION RULES (Exports ONLY the filtered architecture)
# =====================================================================
install(TARGETS inference-gpu DESTINATION bin)
install(FILES ${ALL_RUNTIME_DLLS} DESTINATION bin)