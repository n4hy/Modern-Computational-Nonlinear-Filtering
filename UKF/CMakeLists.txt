# UKF/CMakeLists.txt

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED)

# Include Directories
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/Common/include)

# Source Files
# UKF and Smoother are now header-only templated classes.
# We only compile main.cpp
set(SOURCES
    main.cpp
)

# Executable
add_executable(ukf_test ${SOURCES})
target_link_libraries(ukf_test Eigen3::Eigen)
