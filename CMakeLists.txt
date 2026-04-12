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
        # Support Turing (75), Ampere (80,86), Ada Lovelace (89), Hopper (90)
        # Note: Blackwell (100) requires CUDA 13+ - see DEVELOPMENT_NOTES.md
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()

    message(STATUS "CUDA enabled - Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Define CUDA macro for FilterMath
    add_compile_definitions(OPTMATH_USE_CUDA)

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
    add_compile_options(-O3 -march=native)
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

# --- Dependency: OptimizedKernelsForRaspberryPi5 ---
message(STATUS "Using local OptimizedKernelsForRaspberryPi5_NvidiaCUDA...")

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
