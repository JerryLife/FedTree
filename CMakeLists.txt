cmake_minimum_required(VERSION 3.15)
#cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(FedTree CXX)

#include_directories(/Library/WinNTL-11_5_1/include)

if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

#option(BUILD_SHARED_LIBS "Build as a shared library" ON)
option(USE_CUDA "Compile with CUDA" OFF)
set(BUILD_TESTS OFF CACHE BOOL "Build Tests")

#find_package(Threads)
find_package(OpenMP REQUIRED)
#find_package(Python COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
if (MSVC AND BUILD_SHARED_LIBS)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()


#find_package(OpenCLHeaders REQUIRED)

if (USE_CUDA)
    message("Compile with CUDA")
    find_package(CUDA REQUIRED QUIET)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo --expt-extended-lambda --default-stream per-thread")
else ()
    message("Complie without CUDA")
    #set(Thrust_DIR "${PROJECT_SOURCE_DIR}/thrust/cmake/")
    add_subdirectory(${PROJECT_SOURCE_DIR}/thrust/)
    find_package(Thrust REQUIRED)
    thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)
endif ()
#add_subdirectory(${PROJECT_SOURCE_DIR}/pybind11)
#find_package(pybind11 REQUIRED)
if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# for easylogging++ configuration
add_definitions("-DELPP_FEATURE_PERFORMANCE_TRACKING")
add_definitions("-DELPP_THREAD_SAFE")
add_definitions("-DELPP_STL_LOGGING")
add_definitions("-DELPP_NO_LOG_TO_FILE")


set(CMAKE_CXX_FLAGS_DEBUG "-O3")

add_executable(test_um predict.cpp)
# includes
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
set(DATASET_DIR ${PROJECT_SOURCE_DIR}/data/)
configure_file(include/FedTree/config.h.in config.h)

message("COMMON_INCLUDES: ${COMMON_INCLUDES}")

include_directories(${COMMON_INCLUDES})
include_directories(/usr/local/include/)
include_directories(/usr/include/openssl)

if (USE_CUDA)
    include_directories(${PROJECT_SOURCE_DIR}/cub)
endif ()
add_subdirectory(src/FedTree)

if (BUILD_TESTS)
    message("Building tests")
    add_subdirectory(src/test)
endif ()



# configuration file

