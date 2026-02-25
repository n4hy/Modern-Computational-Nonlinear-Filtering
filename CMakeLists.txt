cmake_minimum_required(VERSION 3.14)
project(NonlinearFiltering)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compiler Optimization Flags for Release builds ---
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Check LTO support but DO NOT enable globally - it conflicts with CUDA nvlink
    # We'll apply LTO to specific targets after dependencies are loaded
    include(CheckIPOSupported)
    check_ipo_supported(RESULT NLF_LTO_SUPPORTED OUTPUT lto_error)
    if(NLF_LTO_SUPPORTED)
        message(STATUS "LTO/IPO supported - will apply to local targets only (not CUDA)")
    else()
        message(STATUS "LTO/IPO not supported: ${lto_error}")
    endif()
    add_compile_definitions(EIGEN_NO_DEBUG EIGEN_FAST_MATH=1)
    add_compile_options(-ffast-math -march=native)
endif()

# Helper function to enable LTO on non-CUDA targets
function(enable_lto_if_supported target)
    if(NLF_LTO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# --- OpenMP Support ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel particle filter")
else()
    message(STATUS "OpenMP not found, particle filter will run single-threaded")
endif()

include(FetchContent)

# Module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Dependency: Eigen3 ---
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching via FetchContent...")

    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_HTM_DOC OFF CACHE BOOL "Disable Eigen html docs" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkgconfig" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)

    FetchContent_Declare(
        Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(Eigen3)

    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

# --- Dependency: OptimizedKernelsForRaspberryPi5 ---
message(STATUS "Using local OptimizedKernelsForRaspberryPi5_NvidiaCUDA...")
FetchContent_Declare(
    OptimizedKernels
    GIT_REPOSITORY $ENV{HOME}/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
    GIT_TAG main
)
FetchContent_MakeAvailable(OptimizedKernels)

# Ensure headers are found. Try both casing conventions.
if(DEFINED OptimizedKernels_SOURCE_DIR)
    include_directories(${OptimizedKernels_SOURCE_DIR}/include)
endif()
if(DEFINED optimizedkernels_SOURCE_DIR)
    include_directories(${optimizedkernels_SOURCE_DIR}/include)
endif()

# Add subdirectories
add_subdirectory(EKF)
add_subdirectory(UKF)
add_subdirectory(PKF)
add_subdirectory(RBPKF)
add_subdirectory(Benchmarks)
