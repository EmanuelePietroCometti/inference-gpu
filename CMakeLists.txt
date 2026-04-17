cmake_minimum_required (VERSION 3.17)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("inference-gpu")

# =====================================================================
# TARGET ARCHITECTURE
# =====================================================================
set(TENSORRT_TARGET_SM "89" CACHE STRING "Target Compute Capability for TensorRT DLLs")
message(STATUS "Installation will target TensorRT SM ${TENSORRT_TARGET_SM}")

# =====================================================================
# PATHS & DEPENDENCIES
# =====================================================================

set(ONNX_ROOT "D:/emanuele/SDK/onnxruntime-win-x64-gpu-1.24.4")
include_directories("${ONNX_ROOT}/include")
link_directories("${ONNX_ROOT}/lib")

set(OpenCV_DIR "D:/emanuele/SDK/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(FMT_ROOT "D:/emanuele/SDK/fmt-11.1.4")
include_directories("${FMT_ROOT}/include")

set(TENSORRT_ROOT "D:/emanuele/SDK/TensorRT-10.16.1.11")

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

set(ALL_RUNTIME_DLLS 
    ${TENSORRT_FILTERED_DLLS}
    ${OPENCV_RELEASE_DLLS}
    ${CUDA_ESSENTIAL_DLLS}
    ${ONNX_DLLS}
    ${FMT_RELEASE_DLLS}
)

# =====================================================================
# TARGET DEFINITION
# =====================================================================

add_executable(inference-gpu "inference-gpu.cpp" "inference-gpu.h")

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

add_custom_command(TARGET inference-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${TENSORRT_ROOT}/bin" 
    "$<TARGET_FILE_DIR:inference-gpu>"
    COMMENT "Copying ALL TensorRT DLLs to output directory..."
)

# =====================================================================
# INSTALLATION RULES (Exports ONLY the filtered architecture)
# =====================================================================
install(TARGETS inference-gpu DESTINATION bin)
install(FILES ${ALL_RUNTIME_DLLS} DESTINATION bin)