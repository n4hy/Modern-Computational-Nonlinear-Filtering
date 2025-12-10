# EKF/CMakeLists.txt

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include Directories
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/Common/include)

# Source Files
set(SOURCES
    src/EKF.cpp
    src/FixedLagSmoother.cpp
    main.cpp
)

# Executable
add_executable(ekf_test ${SOURCES})
target_link_libraries(ekf_test Eigen3::Eigen)
