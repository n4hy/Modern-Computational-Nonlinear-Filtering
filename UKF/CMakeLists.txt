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
enable_lto_if_supported(ukf_test)

# SRUKF Test Executable
add_executable(srukf_test main_srukf.cpp)
target_link_libraries(srukf_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(srukf_test)

# UKF vs SRUKF Comparison (disabled - source files not present)
# add_executable(compare_ukf_srukf compare_ukf_srukf.cpp)
# add_executable(simple_comparison simple_comparison.cpp)
