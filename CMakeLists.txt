cmake_minimum_required(VERSION 3.14)
project(NonlinearFiltering LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA Detection and Configuration ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Detect CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Base multi-arch list: Turing (75), Ampere (80,86), Ada Lovelace (89),
        # Hopper (90). Blackwell (datacenter SM 100, consumer RTX 50-series SM 120)
        # is ONLY added when nvcc is CUDA 13.x+, because CUDA 12.x rejects
        # compute_100/compute_120 ("Unsupported gpu architecture"). This makes the
        # documented Blackwell host (RTX 5070 Ti / SM 120, CUDA 13.1) build by
        # default; on CUDA 12.x it silently stays at 75–90. Override any time with
        # -DCMAKE_CUDA_ARCHITECTURES=native. See DEVELOPMENT_NOTES.md.
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
            list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
            message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} >= 13: adding Blackwell SM 100/120")
        endif()
    endif()

    message(STATUS "CUDA enabled - Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Define CUDA macro for FilterMath
    add_compile_definitions(OPTMATH_USE_CUDA)

    # OPTMATH_USE_CUDA makes the OptMath headers pull in <cuda_runtime.h> and call into
    # cudart from our pure-C++ targets. The OptimizedKernels dependency links CUDA::cudart
    # PRIVATEly, so the toolkit include dir + cudart are NOT propagated to consumers.
    # Supply them globally here so g++-compiled .cpp targets find the headers and link.
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_libraries(CUDA::cudart)

    set(NLF_HAS_CUDA TRUE)
else()
    message(STATUS "CUDA not found - GPU acceleration disabled")
    set(NLF_HAS_CUDA FALSE)
endif()

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
    add_compile_definitions(EIGEN_NO_DEBUG)
    # Host optimization flags. nvcc rejects -march=native directly ("Unknown option"),
    # so CXX gets it straight while CUDA forwards it to the host compiler via -Xcompiler.
    # Both languages MUST enable the same arch: -march=native raises Eigen's
    # EIGEN_MAX_ALIGN_BYTES from 16 to 32, and mixing 16- and 32-byte-aligned Eigen objects
    # across the C++/CUDA translation-unit boundary corrupts the heap ("free(): invalid
    # pointer", e.g. a VectorXf allocated in a .cu and freed in a .cpp). The OptimizedKernels
    # dependency inherits these directory options, so this keeps its .cpp/.cu TUs ABI-consistent.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>
                        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native>)
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

# --- Python Virtual Environment (for plotting scripts) ---
# Opt-out so the C++ build does not require Python3 + live PyPI access. The venv
# is only needed to run the scripts/*.py plotting helpers; turn it off for
# offline / CI / air-gapped builds with -DNLF_BUILD_PYTHON_VENV=OFF.
option(NLF_BUILD_PYTHON_VENV "Create the Python venv for plotting scripts as part of the build" ON)

if(NLF_BUILD_PYTHON_VENV)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python3 found: ${Python3_EXECUTABLE} (${Python3_VERSION})")

set(NLF_VENV_DIR "${CMAKE_SOURCE_DIR}/.nlfvenv")
set(NLF_VENV_PYTHON "${NLF_VENV_DIR}/bin/python3")
set(NLF_VENV_PIP "${NLF_VENV_DIR}/bin/pip3")
set(NLF_REQUIREMENTS "${CMAKE_SOURCE_DIR}/requirements.txt")

# Create venv if it doesn't exist
add_custom_command(
    OUTPUT "${NLF_VENV_DIR}/pyvenv.cfg"
    COMMAND ${Python3_EXECUTABLE} -m venv --system-site-packages "${NLF_VENV_DIR}"
    COMMENT "Creating Python virtual environment at ${NLF_VENV_DIR}"
)

# Install requirements when venv or requirements.txt changes
add_custom_command(
    OUTPUT "${NLF_VENV_DIR}/.requirements_installed"
    DEPENDS "${NLF_VENV_DIR}/pyvenv.cfg" "${NLF_REQUIREMENTS}"
    COMMAND "${NLF_VENV_PIP}" install --quiet --upgrade pip
    COMMAND "${NLF_VENV_PIP}" install --quiet -r "${NLF_REQUIREMENTS}"
    COMMAND ${CMAKE_COMMAND} -E touch "${NLF_VENV_DIR}/.requirements_installed"
    COMMENT "Installing Python requirements into virtual environment"
)

add_custom_target(python_venv ALL
    DEPENDS "${NLF_VENV_DIR}/.requirements_installed"
)

message(STATUS "Python venv: ${NLF_VENV_DIR}")
message(STATUS "  Activate with: source ${NLF_VENV_DIR}/bin/activate")
else()
    message(STATUS "Python venv disabled (NLF_BUILD_PYTHON_VENV=OFF); plotting scripts skipped")
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

# --- Dependency: OptimizedKernelsForRaspberryPi5_NvidiaCUDA (OptMathKernels) ---
#
# OptMathKernels provides the NEON / SVE2 / Vulkan / CUDA compute backends that
# FilterMath dispatches to. This build REQUIRES the OptMath source tree to be
# present as a sibling directory and builds it directly (add_subdirectory), so
# the filters compile against the local working tree / latest `main` rather than
# a pinned release tag. See README.md ("Building") and DEVELOPMENT_NOTES.md.
#
# Provisioning contract (see also ./bootstrap.sh):
#   * If OPTMATH_DIR exists, it is used as-is.
#   * If it is missing and -DAUTO_CLONE_DEPS=ON, it is cloned over HTTPS (main).
#   * If it is missing and AUTO_CLONE_DEPS is OFF (the default), configuration
#     FAILS with instructions rather than silently fetching from the network.
set(OPTMATH_DIR "${CMAKE_SOURCE_DIR}/../OptimizedKernelsForRaspberryPi5_NvidiaCUDA"
    CACHE PATH "Path to the required OptimizedKernels (OptMathKernels) sibling source tree")
set(OPTMATH_HTTPS_URL "https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA.git"
    CACHE STRING "Public HTTPS URL used to clone OptMathKernels when AUTO_CLONE_DEPS=ON")
option(AUTO_CLONE_DEPS
    "If the OptMathKernels sibling directory is missing, git-clone it (latest main) over HTTPS instead of failing"
    OFF)

# Configure OptimizedKernels options before add_subdirectory
set(ENABLE_NEON ON CACHE BOOL "Enable ARM NEON optimizations" FORCE)
set(ENABLE_SVE2 ON CACHE BOOL "Enable ARM SVE2 optimizations" FORCE)
set(ENABLE_VULKAN ON CACHE BOOL "Enable Vulkan GPU compute" FORCE)

if(NLF_HAS_CUDA)
    set(ENABLE_CUDA ON CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE)
else()
    set(ENABLE_CUDA OFF CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
endif()

# Require the OptMath sibling directory; optionally clone it over HTTPS.
if(NOT EXISTS "${OPTMATH_DIR}/CMakeLists.txt")
    if(AUTO_CLONE_DEPS)
        find_package(Git REQUIRED)
        message(STATUS "OptMathKernels not found at ${OPTMATH_DIR}; cloning ${OPTMATH_HTTPS_URL} (main)...")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" clone --branch main "${OPTMATH_HTTPS_URL}" "${OPTMATH_DIR}"
            RESULT_VARIABLE _optmath_clone_rc)
        if(_optmath_clone_rc)
            message(FATAL_ERROR
                "Failed to clone OptMathKernels from ${OPTMATH_HTTPS_URL} (git exit ${_optmath_clone_rc}).")
        endif()
    else()
        message(FATAL_ERROR
            "Required dependency OptMathKernels was not found at:\n"
            "    ${OPTMATH_DIR}\n"
            "This project requires the OptimizedKernelsForRaspberryPi5_NvidiaCUDA source tree to be\n"
            "present as a sibling directory. To provision it, either:\n"
            "  * run   ./bootstrap.sh                 (interactive: offers to clone + build), or\n"
            "  * reconfigure with  -DAUTO_CLONE_DEPS=ON   (clones latest main over HTTPS), or\n"
            "  * clone it manually:\n"
            "        git clone ${OPTMATH_HTTPS_URL} \"${OPTMATH_DIR}\"\n"
            "  * or point  -DOPTMATH_DIR=<path>  at an existing clone.")
    endif()
endif()

# Build OptMathKernels from the local sibling tree (latest main / working tree).
message(STATUS "OptMathKernels source (local): ${OPTMATH_DIR}")
set(OPTMATH_SOURCE_DIR "${OPTMATH_DIR}")
set(OPTMATH_BINARY_DIR "${CMAKE_BINARY_DIR}/optmath-build")
add_subdirectory("${OPTMATH_DIR}" "${OPTMATH_BINARY_DIR}")
include_directories("${OPTMATH_SOURCE_DIR}/include")

# Add subdirectories
add_subdirectory(EKF)
add_subdirectory(UKF)
add_subdirectory(PKF)
add_subdirectory(RBPKF)
add_subdirectory(Benchmarks)

# --- Install rules ---
# This project is primarily a header library (the filters are header-only
# templates); the reusable artifact is the set of public headers. Install them
# into a single include/ root, mirroring the multiple -I roots used at build
# time, so `#include "UKF.h"`, `#include "rbpf/rbpf_core.hpp"`, etc. resolve for
# downstream consumers. (`run_benchmarks` also installs to bin via Benchmarks/.)
install(DIRECTORY
        Common/include/
        EKF/include/
        UKF/include/
        PKF/include/
        RBPKF/include/
        DESTINATION include/nlf
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# --- CTest Integration ---
enable_testing()

add_test(NAME EKF_Test COMMAND ekf_test)
add_test(NAME UKF_Test COMMAND ukf_test)
add_test(NAME SRUKF_Test COMMAND srukf_test)
add_test(NAME SRUKF_AngularWrap COMMAND srukf_angular_wrap_test)
add_test(NAME SRUKF_Smoother COMMAND srukf_smoother_test)
add_test(NAME PKF_Test COMMAND pkf_test)
add_test(NAME PKF_Example COMMAND pkf_example)
add_test(NAME RBPF_Basic COMMAND test_rbpf_basic)
add_test(NAME RBPF_CTRV COMMAND example_rbpf_ctrv)
add_test(NAME Benchmarks COMMAND run_benchmarks)
