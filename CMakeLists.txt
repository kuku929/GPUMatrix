cmake_minimum_required(VERSION 3.22)
project(matrix LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")

#NOTE : change this to your GPU architecture, -arch=sm_xx , where xx is the compute capability. mine is 8.6
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")
find_package(CUDAToolkit REQUIRED)

#include headers
include_directories(include/)

#adding files
add_library(matrix ./src/GPUMatrix.cu)
add_executable(main main.cpp)
target_link_libraries(main matrix)

#cuda files settings
set_target_properties(matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#this should set the appropriate architecture but, I see a decrease in performance compared to -arch=sm_86
#set_property(TARGET matrix PROPERTY CUDA_ARCHITECTURES native)

