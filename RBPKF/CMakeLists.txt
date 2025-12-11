cmake_minimum_required(VERSION 3.10)

# Create library target
add_library(rbpf_core
    src/resampling.cpp
)

target_include_directories(rbpf_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(rbpf_core PUBLIC cxx_std_20)

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(rbpf_core PUBLIC Eigen3::Eigen)

# Examples
add_executable(example_rbpf_ctrv examples/example_rbpf_ctrv.cpp)
target_link_libraries(example_rbpf_ctrv PRIVATE rbpf_core)

# Tests
add_executable(test_rbpf_basic tests/test_rbpf_basic.cpp)
target_link_libraries(test_rbpf_basic PRIVATE rbpf_core)
