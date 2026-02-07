# UKF/CMakeLists.txt

# Find Eigen3
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Include Directories
if(Eigen3_INCLUDE_DIRS)
    include_directories(${Eigen3_INCLUDE_DIRS})
endif()
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/Common/include)

# Source Files
set(SOURCES
    main.cpp
)

# Executable
add_executable(ukf_test ${SOURCES})
target_link_libraries(ukf_test Eigen3::Eigen OptMathKernels)

# SRUKF Test Executable
add_executable(srukf_test main_srukf.cpp)
target_link_libraries(srukf_test Eigen3::Eigen OptMathKernels)

# UKF vs SRUKF Comparison
add_executable(compare_ukf_srukf compare_ukf_srukf.cpp)
target_link_libraries(compare_ukf_srukf Eigen3::Eigen OptMathKernels)
target_compile_options(compare_ukf_srukf PRIVATE -O3 -march=native)

# Simple Comparison (stable test problem)
add_executable(simple_comparison simple_comparison.cpp)
target_link_libraries(simple_comparison Eigen3::Eigen OptMathKernels)
target_compile_options(simple_comparison PRIVATE -O3 -march=native)
