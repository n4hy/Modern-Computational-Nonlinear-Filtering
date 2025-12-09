cmake_minimum_required(VERSION 3.10)
project(EKF_Project)

set(CMAKE_CXX_STANDARD 14)

# Include Eigen from temporary location
include_directories(/tmp/eigen_install/eigen3)

include_directories(include)

add_executable(ekf_test
    main.cpp
    src/EKF.cpp
    src/FixedLagSmoother.cpp
)
