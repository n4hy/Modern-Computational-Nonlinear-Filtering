cmake_minimum_required(VERSION 3.10)

# Create library target
add_library(rbpf_core
    src/resampling.cpp
)

target_include_directories(rbpf_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/Common/include
)

target_compile_features(rbpf_core PUBLIC cxx_std_20)

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()
target_link_libraries(rbpf_core PUBLIC Eigen3::Eigen)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(rbpf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Enable LTO on library
enable_lto_if_supported(rbpf_core)

# Examples
add_executable(example_rbpf_ctrv examples/example_rbpf_ctrv.cpp)
target_link_libraries(example_rbpf_ctrv PRIVATE rbpf_core OptMathKernels)
enable_lto_if_supported(example_rbpf_ctrv)

# Tests
add_executable(test_rbpf_basic tests/test_rbpf_basic.cpp)
target_link_libraries(test_rbpf_basic PRIVATE rbpf_core OptMathKernels)
enable_lto_if_supported(test_rbpf_basic)
