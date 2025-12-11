cmake_minimum_required(VERSION 3.16)
project(NonlinearFiltering)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Dependency: Eigen3
# -----------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 3.4 not found. Downloading and building from source...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(Eigen3)

    # Eigen's CMake might not alias Eigen3::Eigen if built via add_subdirectory/FetchContent
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
else()
    message(STATUS "Found Eigen3: ${Eigen3_DIR}")
endif()

# -----------------------------------------------------------------------------
# Library: NonlinearFiltering (EKF + UKF)
# -----------------------------------------------------------------------------
add_library(NonlinearFiltering
    EKF/src/EKF.cpp
    EKF/src/EKFFixedLag.cpp
)

# Include Directories for the Library
target_include_directories(NonlinearFiltering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Common/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/EKF/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/UKF/include>
    $<INSTALL_INTERFACE:include/NonlinearFiltering>
)

target_link_libraries(NonlinearFiltering PUBLIC Eigen3::Eigen)

# -----------------------------------------------------------------------------
# Executable: PKF Demo
# -----------------------------------------------------------------------------
add_executable(pkf_demo PKF/src/example_main.cpp)
target_include_directories(pkf_demo PRIVATE PKF/include)
target_link_libraries(pkf_demo PRIVATE Eigen3::Eigen)

# -----------------------------------------------------------------------------
# Executables: EKF and UKF Tests (linking against the library)
# -----------------------------------------------------------------------------
add_executable(ekf_test EKF/main.cpp)
target_link_libraries(ekf_test PRIVATE NonlinearFiltering)

add_executable(ukf_test UKF/main.cpp)
target_link_libraries(ukf_test PRIVATE NonlinearFiltering)

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

# Install the library
install(TARGETS NonlinearFiltering
    EXPORT NonlinearFilteringTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install Headers
# We install them into include/NonlinearFiltering/ to namespace them
install(DIRECTORY Common/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NonlinearFiltering/Common)
install(DIRECTORY EKF/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NonlinearFiltering/EKF)
install(DIRECTORY UKF/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NonlinearFiltering/UKF)

# Export Targets (Optional but good practice)
install(EXPORT NonlinearFilteringTargets
    FILE NonlinearFilteringTargets.cmake
    NAMESPACE NonlinearFiltering::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NonlinearFiltering
)
