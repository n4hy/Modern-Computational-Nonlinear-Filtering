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
        # Default multi-arch list: Turing (75), Ampere (80,86), Ada Lovelace (89),
        # Hopper (90). Blackwell (datacenter SM 100, consumer RTX 50-series SM 120)
        # needs CUDA 13.x and is NOT in this list — build for it explicitly with
        # -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON. See
        # DEVELOPMENT_NOTES.md (verified on RTX 5070 Ti / SM 120, CUDA 13.1).
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
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

# --- Python Virtual Environment (Required) ---
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
# FilterMath dispatches to. The upstream project publishes semantic-version
# release tags (v0.5.x); we PIN to a specific tag rather than tracking a moving
# branch so that builds are reproducible and a kernel update is a deliberate,
# audited bump of OPTMATH_RELEASE_TAG below (see DEVELOPMENT_NOTES.md ->
# "OptMathKernels Dependency - Release Audit & Pinning Policy"). To adopt a
# newer kernel release: audit the upstream diff,
# update this tag, rebuild, retest, rebenchmark, then commit.
#
# Pinned: v0.5.17 — "fix TrsvLower64x64 test tolerance; document build deps"
#   (v0.5.16 added x86_64 desktop RTX 5090 benchmark numbers — docs only;
#    v0.5.17 relaxed the NEON TrsvLower64x64 unit-test tolerance to 5e-3 to match
#    its TrsvUpper twin, and documented apt/build deps — glslang-tools/glslc for
#    Vulkan SPIR-V shaders — in requirements.txt. No public API (include/) or
#    compute-backend (src/) change vs v0.5.15.)
set(OPTMATH_RELEASE_TAG "v0.5.17" CACHE STRING "Pinned OptMathKernels release tag")
message(STATUS "Using OptimizedKernelsForRaspberryPi5_NvidiaCUDA @ ${OPTMATH_RELEASE_TAG}")

# Configure OptimizedKernels options before FetchContent
set(ENABLE_NEON ON CACHE BOOL "Enable ARM NEON optimizations" FORCE)
set(ENABLE_SVE2 ON CACHE BOOL "Enable ARM SVE2 optimizations" FORCE)
set(ENABLE_VULKAN ON CACHE BOOL "Enable Vulkan GPU compute" FORCE)

if(NLF_HAS_CUDA)
    set(ENABLE_CUDA ON CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE)
else()
    set(ENABLE_CUDA OFF CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
endif()

# GIT_TAG follows the pinned release tag. The repository is the local clone in
# $HOME (kept in sync with github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA);
# FetchContent checks out OPTMATH_RELEASE_TAG into _deps/optimizedkernels-src.
FetchContent_Declare(
    OptimizedKernels
    GIT_REPOSITORY $ENV{HOME}/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
    GIT_TAG ${OPTMATH_RELEASE_TAG}
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

# --- CTest Integration ---
enable_testing()

add_test(NAME EKF_Test COMMAND ekf_test)
add_test(NAME UKF_Test COMMAND ukf_test)
add_test(NAME SRUKF_Test COMMAND srukf_test)
add_test(NAME PKF_Test COMMAND pkf_test)
add_test(NAME PKF_Example COMMAND pkf_example)
add_test(NAME RBPF_Basic COMMAND test_rbpf_basic)
add_test(NAME RBPF_CTRV COMMAND example_rbpf_ctrv)
add_test(NAME Benchmarks COMMAND run_benchmarks)
