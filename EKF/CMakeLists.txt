cmake_minimum_required(VERSION 3.10)
project(EKF_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 using standard mechanism
# If Eigen3 is not in a standard path, set Eigen3_DIR
# e.g., cmake -D Eigen3_DIR=/path/to/eigen3/share/eigen3/cmake ..
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(ekf_test
    main.cpp
    src/EKF.cpp
    src/FixedLagSmoother.cpp
)
target_link_libraries(ekf_test PRIVATE Eigen3::Eigen)
target_include_directories(ekf_test PRIVATE include)

add_executable(ukf_test
    main_ukf.cpp
    src/UKF.cpp
    src/UnscentedFixedLagSmoother.cpp
)
target_link_libraries(ukf_test PRIVATE Eigen3::Eigen)
target_include_directories(ukf_test PRIVATE include)
