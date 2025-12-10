cmake_minimum_required(VERSION 3.10)
project(EKF_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point directly to the downloaded Eigen source
include_directories("/tmp/eigen_install/eigen3")

add_executable(ekf_test
    main.cpp
    src/EKF.cpp
    src/FixedLagSmoother.cpp
)
target_include_directories(ekf_test PRIVATE include)

add_executable(ukf_test
    main_ukf.cpp
    src/UKF.cpp
    src/UnscentedFixedLagSmoother.cpp
)
target_include_directories(ukf_test PRIVATE include)
