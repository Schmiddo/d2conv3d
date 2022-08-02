cmake_minimum_required(VERSION 3.16)
project(fast_grouped_conv3d LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("${TORCH_INCLUDE_DIRS}")

set(
        SOURCES
        csrc/grouped_conv3d.h
        csrc/grouped_conv3d.cu
)

add_library(fast_grouped_conv3d ${SOURCES})
target_link_libraries(fast_grouped_conv3d ${TORCH_LIBRARIES})

add_executable(sample csrc/main.cpp)
target_link_libraries(sample fast_grouped_conv3d ${TORCH_LIBRARIES})
set_property(TARGET sample PROPERTY CXX_STANDARD 14)
set_target_properties(sample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")