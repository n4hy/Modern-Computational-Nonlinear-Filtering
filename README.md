# Nonlinear Filtering with Optimized Kernels (NEON + Vulkan)

This repository implements various nonlinear filters (EKF, UKF, PKF, RBPKF) optimized for the Raspberry Pi 5 using NEON intrinsics (for matrix operations) and Vulkan (for large-scale parallel operations).

## Optimization Overview

*   **Precision:** All filters now use single-precision floating point (`float`) to leverage NEON vectorization and reduced memory bandwidth.
*   **Matrix Operations (EKF, UKF, RBPKF):** Matrix multiplications are accelerated using NEON-optimized GEMM kernels (`optmath::neon::neon_gemm`).
*   **Parallel Operations (PKF):** The Particle Filter's noise addition step uses Vulkan compute shaders (`optmath::vulkan::vulkan_vec_add`) when the particle count exceeds 100.

## Dependencies

*   **CMake:** 3.15+
*   **Compiler:** GCC 10+ or Clang 11+ (C++20 support required)
*   **Eigen3:** Fetched automatically via CMake.
*   **OptimizedKernelsForRaspberryPi5:** Fetched automatically via CMake.
*   **Vulkan SDK:** Required for Vulkan acceleration (optional, runtime check).

## Building

```bash
mkdir build
cd build
cmake ..
make -j4
```

## Running Examples

### EKF (Extended Kalman Filter)
```bash
./EKF/ekf_test
```

### UKF (Unscented Kalman Filter)
```bash
./UKF/ukf_test
```

### PKF (Particle Filter)
```bash
./PKF/pkf_demo
```

### RBPKF (Rao-Blackwellized Particle Filter)
```bash
./RBPKF/example_rbpf_ctrv
```

## Visualizing Results

The examples produce CSV files (e.g., `ekf_results.csv`). You can visualize them using the provided script:

```bash
python3 scripts/plot_optimized.py ekf_results.csv ukf_results.csv ./plots
```
