cmake_minimum_required(VERSION 3.18)
project(MyGemmLibrary LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

# thirdparty 내부 CMakeLists.txt 포함
add_subdirectory(thirdparty)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/gemm.cpp
    src/cuda_gemm_handler.cpp
    src/gemm_naive.cu
)

add_library(gemm STATIC ${SOURCES})
set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gemm PRIVATE rmm::rmm)

enable_testing()
add_subdirectory(unittest)
