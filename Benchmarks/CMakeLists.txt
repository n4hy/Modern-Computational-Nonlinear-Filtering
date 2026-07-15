cmake_minimum_required(VERSION 3.20)
project(FilterBenchmarks)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages. Guard find_package so the FetchContent-provided Eigen
# target (which installs no Eigen3Config.cmake) is used when Eigen is not on the
# system — matching the pattern in EKF/UKF/PKF/RBPKF. A bare find_package(Eigen3
# REQUIRED) here would abort configure on hosts without a system Eigen.
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/UKF/include
    ${CMAKE_SOURCE_DIR}/Common/include
)

# Benchmark executable
add_executable(run_benchmarks
    src/run_benchmarks.cpp
)

target_link_libraries(run_benchmarks
    PRIVATE
    Eigen3::Eigen
    OptMathKernels
)

# -O3 and the native-tuning flag come from the root: it probes the correct spelling
# (-mcpu=native on aarch64, -march=native on x86) and honours NLF_ENABLE_NATIVE_ARCH,
# and its add_compile_options are inherited here. Repeating -march=native locally was
# wrong twice over: it bypassed that opt-out, so -DNLF_ENABLE_NATIVE_ARCH=OFF left the
# one installed binary non-portable; and on aarch64 it landed *alongside* the inherited
# -mcpu=native, where it overrides the arch component and drops the tuning the probe
# selected. Only the numerical setting is target-specific.
target_compile_options(run_benchmarks PRIVATE -fno-fast-math)
target_compile_definitions(run_benchmarks PRIVATE EIGEN_FAST_MATH=0)
enable_lto_if_supported(run_benchmarks)

# Installation
install(TARGETS run_benchmarks DESTINATION bin)
