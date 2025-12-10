cmake_minimum_required(VERSION 3.10)
project(EKF_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 using standard CMake mechanism
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(ekf_test
    main.cpp
    src/EKF.cpp
    src/FixedLagSmoother.cpp
)

# Link Eigen and include local headers
target_link_libraries(ekf_test PRIVATE Eigen3::Eigen)
target_include_directories(ekf_test PRIVATE include)
