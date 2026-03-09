cmake_minimum_required(VERSION 3.20)
project(AircraftNav)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# Find Google Test for unit tests
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/UKF/include
    ${CMAKE_SOURCE_DIR}/Common/include
    ${EIGEN3_INCLUDE_DIR}
)

# Main simulation executable
add_executable(aircraft_nav_simulation
    src/aircraft_nav_simulation.cpp
)

target_link_libraries(aircraft_nav_simulation
    PRIVATE
    Eigen3::Eigen
        Threads::Threads
)

# Disable fast-math and Eigen optimizations for numerical stability in Kalman filters
target_compile_options(aircraft_nav_simulation PRIVATE -O3 -march=native -fno-fast-math)
# Override global EIGEN defines that cause numerical instability
target_compile_definitions(aircraft_nav_simulation PRIVATE EIGEN_FAST_MATH=0)
# Undefine EIGEN_NO_DEBUG to keep safety checks (must come before Eigen headers)
target_compile_options(aircraft_nav_simulation PRIVATE -UEIGEN_NO_DEBUG)
enable_lto_if_supported(aircraft_nav_simulation)

# Monte Carlo analysis executable
add_executable(monte_carlo_analysis
    src/monte_carlo_analysis.cpp
)

target_link_libraries(monte_carlo_analysis
    PRIVATE
    Eigen3::Eigen
        Threads::Threads
)

# Disable fast-math and Eigen optimizations for numerical stability in Kalman filters
target_compile_options(monte_carlo_analysis PRIVATE -O3 -march=native -fno-fast-math)
# Override global EIGEN defines that cause numerical instability
target_compile_definitions(monte_carlo_analysis PRIVATE EIGEN_FAST_MATH=0)
# Undefine EIGEN_NO_DEBUG to keep safety checks (must come before Eigen headers)
target_compile_options(monte_carlo_analysis PRIVATE -UEIGEN_NO_DEBUG)
enable_lto_if_supported(monte_carlo_analysis)

# Unit tests
enable_testing()

# Test: Aircraft Dynamics
add_executable(test_aircraft_dynamics
    tests/test_aircraft_dynamics.cpp
)

target_link_libraries(test_aircraft_dynamics
    PRIVATE
    Eigen3::Eigen
        GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME AircraftDynamicsTest COMMAND test_aircraft_dynamics)

# Test: INS Error Model
add_executable(test_ins_error
    tests/test_ins_error.cpp
)

target_link_libraries(test_ins_error
    PRIVATE
    Eigen3::Eigen
        GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME INSErrorTest COMMAND test_ins_error)

# Test: Convergence
add_executable(test_convergence
    tests/test_convergence.cpp
)

target_link_libraries(test_convergence
    PRIVATE
    Eigen3::Eigen
        GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME ConvergenceTest COMMAND test_convergence)

# Test: Monte Carlo
add_executable(test_monte_carlo
    tests/test_monte_carlo.cpp
)

target_link_libraries(test_monte_carlo
    PRIVATE
    Eigen3::Eigen
        GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME MonteCarloTest COMMAND test_monte_carlo)

# Installation
install(TARGETS aircraft_nav_simulation monte_carlo_analysis DESTINATION bin)

# Print summary
message(STATUS "AircraftNav module configured")
message(STATUS "  Main executables: aircraft_nav_simulation, monte_carlo_analysis")
message(STATUS "  Unit tests: test_aircraft_dynamics, test_ins_error, test_convergence, test_monte_carlo")
