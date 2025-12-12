# PKF CMakeLists.txt

# Find Eigen3
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Header-only library for PKF core
add_library(PKF_Lib INTERFACE)
target_include_directories(PKF_Lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PKF_Lib INTERFACE Eigen3::Eigen)

# Example Executable
add_executable(pkf_example src/example_main.cpp)
target_link_libraries(pkf_example PRIVATE PKF_Lib OptMathKernels)

# Test Executable
add_executable(pkf_test tests/test_particle.cpp)
target_link_libraries(pkf_test PRIVATE PKF_Lib OptMathKernels)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pkf_example PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(pkf_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

# Optimization for Release
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pkf_example PRIVATE -O3)
    target_compile_options(pkf_test PRIVATE -O3)
endif()
