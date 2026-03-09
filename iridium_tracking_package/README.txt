Iridium-Next UKF AOA/Doppler Tracking Package
=============================================

This package contains files for Iridium-Next satellite tracking using a
two-antenna coherent receiver with UKF, AOA+Doppler measurements, and
multi-satellite tracking.

Files included:
---------------

include/optmath/
  - ukf_aoa_tracking.hpp           - Base UKF AOA tracking implementation
  - ukf_aoa_doppler_tracking.hpp   - AOA + Doppler tracking extension
  - iridium_burst_demodulator.hpp  - Iridium burst demodulation
  - multi_satellite_tracker.hpp    - Multi-satellite tracking system

examples/
  - iridium_aoa_tracking.cpp       - Basic AOA tracking simulation
  - compare_aoa_doppler.cpp        - AOA vs AOA+Doppler comparison tool
  - iridium_tracking_complete.cpp  - Complete tracking system demo

tests/
  - test_ukf_aoa.cpp               - Unit tests for UKF AOA tracking

Patch files:
------------
  - examples_CMakeLists_additions.patch  - CMake config for examples
  - tests_CMakeLists_additions.patch     - CMake config for tests

Integration notes:
------------------
These files were originally developed in OptimizedKernelsForRaspberryPi5_NvidiaCUDA
but belong in Modern-Computational-Nonlinear-Filtering for use with Square Root UKF.

You may need to adjust include paths and CMake configuration for the target repository.
