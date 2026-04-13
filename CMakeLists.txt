# CMakeList.txt: CMake project for inference-gpu. 
# Define project-specific logic here.
cmake_minimum_required (VERSION 3.17)

# Enable Hot Reload for MSVC compilers, if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("inference-gpu")

set(ONNX_ROOT "D:/emanuele/SDK/onnxruntime-win-x64-gpu-1.24.4")
include_directories("${ONNX_ROOT}/include")
link_directories("${ONNX_ROOT}/lib")

set(OpenCV_DIR "D:/emanuele/SDK/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(FMT_ROOT "D:/emanuele/SDK/fmt-11.1.4")
include_directories("${FMT_ROOT}/include")

set(TENSORRT_ROOT "D:/emanuele/SDK/TensorRT-10.16.1.11")

add_executable (inference-gpu "inference-gpu.cpp" "inference-gpu.h")


if (MSVC)
    target_compile_options(inference-gpu PRIVATE /utf-8)
endif()
target_link_libraries(inference-gpu
    onnxruntime 
    ${OpenCV_LIBS} 
    debug "${FMT_ROOT}/build/lib/Debug/fmtd.lib"
    optimized "${FMT_ROOT}/build/lib/Release/fmt.lib"
)

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
    COMMENT "Copying TensorRT DLLs to output directory..."
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET inference-gpu PROPERTY CXX_STANDARD 20)
endif()