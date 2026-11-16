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
nlf_add_warning_flags(ukf_test)

# SRUKF Test Executable
add_executable(srukf_test main_srukf.cpp)
target_link_libraries(srukf_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(srukf_test)
nlf_add_warning_flags(srukf_test)

# SRUKF angular-observation innovation wrap (R32) + NIS exposure (R33) regression test
add_executable(srukf_angular_wrap_test tests/test_srukf_angular_wrap.cpp)
target_link_libraries(srukf_angular_wrap_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(srukf_angular_wrap_test)

# Full-interval square-root unscented RTS smoother + iterated smoothing regression test
add_executable(srukf_smoother_test tests/test_srukf_smoother.cpp)
target_link_libraries(srukf_smoother_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(srukf_smoother_test)

# Full-interval unscented (plain-covariance) RTS smoother + iterated smoothing test
add_executable(ukf_smoother_test tests/test_ukf_smoother.cpp)
target_link_libraries(ukf_smoother_test Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(ukf_smoother_test)

# UKF vs SRUKF Comparison (disabled - source files not present)
# add_executable(compare_ukf_srukf compare_ukf_srukf.cpp)
# add_executable(simple_comparison simple_comparison.cpp)

# --- Regression tests introduced by the audit-remediation pass ---
add_executable(test_ukf_numerical tests/test_ukf_numerical.cpp)
target_link_libraries(test_ukf_numerical Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(test_ukf_numerical)
nlf_add_warning_flags(test_ukf_numerical)

add_executable(test_srukf_initialize tests/test_srukf_initialize.cpp)
target_link_libraries(test_srukf_initialize Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(test_srukf_initialize)
nlf_add_warning_flags(test_srukf_initialize)

# 2026-11-15 audit: one check per demonstrated defect (Wc0 floor, pivot-aware
# LDLT sqrt, NaN measurement, angular wrap, smoother time convention, fixed-lag
# re-init, non-finite rollback, SRUKF kappa sentinel).
add_executable(test_audit_regressions tests/test_audit_regressions.cpp)
target_link_libraries(test_audit_regressions Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(test_audit_regressions)
nlf_add_warning_flags(test_audit_regressions)

# Iterated smoothers (2026-11-16): IEKS (EKF) and IPLS (UKF/SRUKF) vs an exact
# double RTS on a linear model, MAP stationarity, UKF/SRUKF agreement, angular wrap.
add_executable(test_iterated_smoothers tests/test_iterated_smoothers.cpp ${CMAKE_SOURCE_DIR}/EKF/src/EKF.cpp)
target_include_directories(test_iterated_smoothers PRIVATE ${CMAKE_SOURCE_DIR}/EKF/include)
target_link_libraries(test_iterated_smoothers Eigen3::Eigen OptMathKernels)
enable_lto_if_supported(test_iterated_smoothers)
nlf_add_warning_flags(test_iterated_smoothers)
