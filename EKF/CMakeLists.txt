# EKF/CMakeLists.txt

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
    src/EKF.cpp
    src/EKFFixedLag.cpp
    main.cpp
)

# Executable
add_executable(ekf_test ${SOURCES})
target_link_libraries(ekf_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(ekf_test)
