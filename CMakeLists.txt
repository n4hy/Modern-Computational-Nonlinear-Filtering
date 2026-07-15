cmake_minimum_required(VERSION 3.14)
project(NonlinearFiltering VERSION 3.4.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Default build type ---
# Single-config generators leave CMAKE_BUILD_TYPE empty unless told otherwise, so a
# bare `cmake ..` compiles at -O0 with no NDEBUG. Every -O3, native-tuning and LTO
# decision below is guarded on $<CONFIG:Release>, which means an empty build type
# yields an unoptimized library that still runs the benchmarks and still prints
# plausible-looking step times -- the numbers are simply measuring different code.
#
# NLF never actually chose Release: the value arrived as a side effect of the
# OptimizedKernels dependency's add_subdirectory defaulting it into the cache. That
# made our optimization level a property of a third-party file, and it silently
# reverts to -O0 if that dependency ever stops doing it. Choose it here instead, ahead
# of the add_subdirectory. Multi-config generators (Ninja Multi-Config, VS, Xcode)
# ignore CMAKE_BUILD_TYPE by design and select the config at build time, so skip them.
get_property(_nlf_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _nlf_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not specified - defaulting to Release")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)

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

# --- Target architecture detection ---
# The filters are portable C++; the SIMD backends are not. Detect the TARGET
# architecture (CMAKE_SYSTEM_PROCESSOR, which respects cross-compilation) so the
# ARM-only options below are never forced onto a non-ARM target. FilterMath.h
# performs the matching decision at compile time via FILTERMATH_ARM64 and falls
# back to Eigen, so every tier here is optional.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv8)")
    set(NLF_ARCH_ARM TRUE)
    set(NLF_ARCH_NAME "ARM64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM)")
    set(NLF_ARCH_ARM TRUE)
    set(NLF_ARCH_NAME "ARM32")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)")
    set(NLF_ARCH_X86 TRUE)
    set(NLF_ARCH_NAME "x86")
else()
    set(NLF_ARCH_NAME "${CMAKE_SYSTEM_PROCESSOR} (generic)")
endif()
message(STATUS "Target architecture: ${NLF_ARCH_NAME}")

# --- Compiler Optimization Flags for Release builds ---
# Guarded by generator expressions ($<CONFIG:Release>), NOT `if(CMAKE_BUILD_TYPE
# STREQUAL "Release")`. The STREQUAL form silently no-ops twice: on a first
# configure from an empty cache (CMAKE_BUILD_TYPE is not yet set when this line
# runs, even though the dependency later defaults it to Release in the cache),
# and under every multi-config generator (Ninja Multi-Config, Visual Studio, Xcode),
# where CMAKE_BUILD_TYPE is empty by design. Both cases produced binaries with no
# -O3/-march/LTO while CMakeCache.txt read "Release".
include(CheckIPOSupported)
check_ipo_supported(RESULT NLF_LTO_SUPPORTED OUTPUT lto_error)
if(NLF_LTO_SUPPORTED)
    message(STATUS "LTO/IPO supported - will apply to local targets only (not CUDA)")
else()
    message(STATUS "LTO/IPO not supported: ${lto_error}")
endif()

add_compile_definitions($<$<CONFIG:Release>:EIGEN_NO_DEBUG>)

# Native-CPU tuning. OFF by default when cross-compiling, since "native" then
# means the BUILD host and would emit instructions the target cannot execute.
if(CMAKE_CROSSCOMPILING)
    set(_nlf_native_default OFF)
else()
    set(_nlf_native_default ON)
endif()
option(NLF_ENABLE_NATIVE_ARCH
    "Tune Release builds for the building machine's CPU (-march=native / -mcpu=native). Turn OFF for portable/redistributable binaries."
    ${_nlf_native_default})

# The native-tuning spelling is compiler- and architecture-specific: x86 GCC/Clang
# take -march=native; aarch64 GCC/Clang want -mcpu=native (-march=native is rejected
# by some aarch64 toolchains and by Apple clang); MSVC has no equivalent. Probe
# rather than assume, so an unknown toolchain degrades to portable codegen instead
# of failing to configure.
set(NLF_ARCH_FLAG "")
if(NLF_ENABLE_NATIVE_ARCH AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    if(NLF_ARCH_ARM)
        check_cxx_compiler_flag("-mcpu=native" NLF_HAS_MCPU_NATIVE)
        if(NLF_HAS_MCPU_NATIVE)
            set(NLF_ARCH_FLAG "-mcpu=native")
        endif()
    else()
        check_cxx_compiler_flag("-march=native" NLF_HAS_MARCH_NATIVE)
        if(NLF_HAS_MARCH_NATIVE)
            set(NLF_ARCH_FLAG "-march=native")
        endif()
    endif()
endif()

if(MSVC)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
else()
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>)
endif()

if(NLF_ARCH_FLAG)
    message(STATUS "Native CPU tuning: ${NLF_ARCH_FLAG}")
    # Both languages MUST agree on the arch flag. On x86, -march=native raises Eigen's
    # EIGEN_MAX_ALIGN_BYTES from 16 to 32, and mixing 16- and 32-byte-aligned Eigen
    # objects across the C++/CUDA translation-unit boundary corrupts the heap
    # ("free(): invalid pointer" — e.g. a VectorXf allocated in a .cu and freed in a
    # .cpp). aarch64 is not exposed to that specific hazard (baseline armv8-a already
    # implies NEON and EIGEN_MAX_ALIGN_BYTES stays 16 either way), but keeping the two
    # languages in lockstep is correct on every target. nvcc rejects the flag directly
    # ("Unknown option"), so CUDA forwards it to the host compiler via -Xcompiler.
    # The OptimizedKernels dependency inherits these directory options, keeping its
    # .cpp/.cu TUs ABI-consistent with ours.
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${NLF_ARCH_FLAG}>)
    if(NLF_HAS_CUDA)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=${NLF_ARCH_FLAG}>)
    endif()
else()
    message(STATUS "Native CPU tuning: disabled (portable codegen)")
endif()

# Helper function to enable LTO on non-CUDA targets
# Enable LTO on a target for the Release configuration only.
#
# Uses the per-config INTERPROCEDURAL_OPTIMIZATION_RELEASE property rather than a
# `CMAKE_BUILD_TYPE STREQUAL "Release"` guard around the config-agnostic one. The
# guard was dead under every multi-config generator (CMAKE_BUILD_TYPE is empty
# there by design), so NLF's targets silently lost LTO while the dependency's kept
# it; and single-config builds only passed the guard incidentally, because the
# dependency's add_subdirectory defaults CMAKE_BUILD_TYPE into the cache before
# these call sites are reached. The per-config property is evaluated at generate
# time and is correct for both generator kinds, without pinning LTO onto Debug.
function(enable_lto_if_supported target)
    if(NLF_LTO_SUPPORTED)
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
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

# Configure OptimizedKernels options before add_subdirectory.
#
# These are gated on the TARGET architecture rather than forced ON unconditionally.
# NEON and SVE2 are ARM-only instruction sets: requesting them on an x86 or other
# target asks the dependency for something it cannot build there. (It tolerates the
# request today by internally re-checking ON_ARM, so the old unconditional force was
# not fatal — but it made this project's build inputs a lie about its own target, and
# left NLF one dependency-refactor away from breaking on every non-ARM machine.)
#
# Each is exposed as an NLF-level option so a user can still override it, and the
# user's choice is what propagates to the dependency. Every backend is optional:
# FilterMath.h resolves the same tiers at compile time and falls back to Eigen, so a
# target with none of them still builds and produces identical filter results.
option(NLF_ENABLE_NEON  "Build the ARM NEON kernels (ARM targets only)" ${NLF_ARCH_ARM})
option(NLF_ENABLE_SVE2  "Build the ARM SVE2 kernels (ARM targets only; skipped when the CPU lacks SVE2)" ${NLF_ARCH_ARM})
option(NLF_ENABLE_VULKAN "Build the Vulkan compute backend (portable; auto-skipped when no SDK is found)" ON)

set(ENABLE_NEON  ${NLF_ENABLE_NEON}  CACHE BOOL "Enable ARM NEON optimizations" FORCE)
set(ENABLE_SVE2  ${NLF_ENABLE_SVE2}  CACHE BOOL "Enable ARM SVE2 optimizations" FORCE)
set(ENABLE_VULKAN ${NLF_ENABLE_VULKAN} CACHE BOOL "Enable Vulkan GPU compute" FORCE)

if(NLF_ARCH_ARM)
    message(STATUS "SIMD backends: NEON=${NLF_ENABLE_NEON} SVE2=${NLF_ENABLE_SVE2} (SVE2 sources are skipped unless the CPU implements it)")
else()
    message(STATUS "SIMD backends: none for ${NLF_ARCH_NAME} - filter math uses the Eigen fallback path")
endif()

if(NLF_HAS_CUDA)
    set(ENABLE_CUDA ON CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE)
else()
    set(ENABLE_CUDA OFF CACHE BOOL "Enable NVIDIA CUDA support" FORCE)
endif()

# The dependency's test suite FetchContent's GoogleTest, which installs itself by
# default -- dumping gtest/gmock headers, libs and CMake config into our install
# prefix. We are not redistributing GoogleTest; it is a test-only dependency of a
# dependency.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

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

# --- Warnings, for NLF's own targets only ---
# Placed deliberately AFTER the dependency's add_subdirectory above and BEFORE ours
# below: directory options are inherited by subdirectories added after the call, so
# this covers code we own without burying the build in warnings from third-party
# sources we are not in a position to fix.
#
# Previously -Wall/-Wextra were set on exactly two targets (pkf_example, pkf_test).
# Every filter, every smoother, the RBPF and the benchmark compiled with no warning
# flags whatsoever, so a silent build was not evidence of much.
#
# Deliberately NOT -Werror. This is numerical code compiled by a wide range of
# toolchains, and each new compiler release invents warnings that would turn a
# previously working checkout into a hard build failure for someone who just wants to
# use the library. These are signal for maintainers, not a tripwire for users.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# Add subdirectories
add_subdirectory(EKF)
add_subdirectory(UKF)
add_subdirectory(PKF)
add_subdirectory(RBPKF)
add_subdirectory(Benchmarks)

# ============================================================================
#  Install / export
# ============================================================================
#
# The consumer contract is a CMake package:
#
#     find_package(nlf REQUIRED)
#     target_link_libraries(my_app PRIVATE nlf::nlf)
#     #include "UKF.h"        // or <nlf/UKF.h> -- both work, see below
#
# Previously the install was a bare install(DIRECTORY) that flattened five -I
# roots into include/nlf and declared nothing else. That could not work, for two
# reasons that a config file alone would not have fixed:
#
#  1. The include root was ambiguous. Headers use quoted includes ("FilterMath.h")
#     which resolve against a FLAT root, but the nlf/ prefix invited
#     <nlf/rbpf/rbpf_core.hpp>, under which those quoted includes do not resolve.
#     Consumers had to guess, and half the headers failed either way.
#  2. FilterMath.h is a PUBLIC header that does `#include <optmath/neon_kernels.hpp>`
#     and calls optmath::. OptMathKernels ships no install() or export rules at
#     all, so nothing put its headers or its library in the install tree: the
#     installed headers could not compile, and would not have linked if they had.
#
# Both are fixed by exporting a real target that carries BOTH include roots:
# <prefix>/include (so <nlf/UKF.h> and <optmath/...> resolve) and
# <prefix>/include/nlf (so the headers' own quoted includes resolve). Consumers
# inherit them from nlf::nlf and never guess.
#
# OptMathKernels is installed and exported as part of this package. We do not own
# it, but our public ABI hard-depends on it and it provides no install rules of
# its own, so shipping our build of it is what makes the install usable at all.
# It was already export-ready ($<BUILD_INTERFACE>/$<INSTALL_INTERFACE> on its
# include dirs); only the install() rules were missing.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Umbrella target for the header-only filters (EKF/UKF/SRUKF/PKF live in headers;
# EKF/ and UKF/ have no library target of their own, only directory-scope includes).
add_library(nlf INTERFACE)
add_library(nlf::nlf ALIAS nlf)

target_include_directories(nlf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Common/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/EKF/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/UKF/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/PKF/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/RBPKF/include>
    # Flat root: the headers' own quoted includes resolve against this.
    # <prefix>/include comes along via OptMathKernels' INSTALL_INTERFACE, which is
    # what makes <nlf/UKF.h> work too.
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nlf>)

target_compile_features(nlf INTERFACE cxx_std_20)
target_link_libraries(nlf INTERFACE Eigen3::Eigen OptMathKernels rbpf_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nlf INTERFACE OpenMP::OpenMP_CXX)
endif()

install(TARGETS nlf rbpf_core PKF_Lib OptMathKernels
        EXPORT nlfTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# This project's headers -> <prefix>/include/nlf (flattened; the headers are
# written for a flat root and include each other unqualified).
install(DIRECTORY
        Common/include/
        EKF/include/
        UKF/include/
        PKF/include/
        RBPKF/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nlf
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# OptMathKernels' headers -> <prefix>/include/optmath/, preserving the directory
# name, because our public FilterMath.h includes them as <optmath/...>.
install(DIRECTORY "${OPTMATH_SOURCE_DIR}/include/optmath"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(EXPORT nlfTargets
        FILE nlfTargets.cmake
        NAMESPACE nlf::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlf)

# Whether the build ACTUALLY linked Vulkan, as opposed to whether it was asked for.
# NLF_ENABLE_VULKAN defaults ON and is FORCEd into the ENABLE_VULKAN cache entry above,
# so ENABLE_VULKAN reads ON even on a host with no SDK, where the dependency quietly
# skips the backend (its own auto-disable is a directory-scoped normal variable inside
# add_subdirectory and is invisible from here). Interpolating that request into the
# package config emitted `if(ON) find_dependency(Vulkan)` unconditionally, so
# find_package(nlf) hard-failed on every SDK-less consumer -- for a backend that build
# never contained, and contradicting NLF_ENABLE_VULKAN's own "auto-skipped when no SDK
# is found" contract. Vulkan_LIBRARY is set by the dependency's find_package(Vulkan)
# and is a cache entry, so unlike Vulkan_FOUND it is readable at this scope.
set(NLF_VULKAN_ACTIVE OFF)
if(ENABLE_VULKAN AND Vulkan_LIBRARY)
    set(NLF_VULKAN_ACTIVE ON)
endif()
message(STATUS "Vulkan backend: requested=${NLF_ENABLE_VULKAN} active=${NLF_VULKAN_ACTIVE}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nlfConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nlfConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlf)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nlfConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nlfConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nlfConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlf)

# --- CTest Integration ---
enable_testing()

add_test(NAME EKF_Test COMMAND ekf_test)
add_test(NAME UKF_Test COMMAND ukf_test)
add_test(NAME SRUKF_Test COMMAND srukf_test)
add_test(NAME SRUKF_AngularWrap COMMAND srukf_angular_wrap_test)
add_test(NAME SRUKF_Smoother COMMAND srukf_smoother_test)
add_test(NAME UKF_Smoother COMMAND ukf_smoother_test)
add_test(NAME EKF_Smoother COMMAND ekf_smoother_test)
add_test(NAME PKF_Test COMMAND pkf_test)
add_test(NAME PKF_Example COMMAND pkf_example)
add_test(NAME RBPF_Basic COMMAND test_rbpf_basic)
add_test(NAME RBPF_CTRV COMMAND example_rbpf_ctrv)
add_test(NAME Benchmarks COMMAND run_benchmarks)
