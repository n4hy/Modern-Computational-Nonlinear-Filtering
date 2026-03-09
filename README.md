# Modern Computational Nonlinear Filtering

<div align="center">

**High-Performance Nonlinear State Estimation for Embedded Systems**

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-ARM%20aarch64%20%2B%20x86__64-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optimization](https://img.shields.io/badge/Optimization-NEON%20%2B%20SVE2%20%2B%20Vulkan-orange.svg)](https://developer.arm.com/Architectures/Neon)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Implemented Filters](#implemented-filters)
- [Benchmark Results](#benchmark-results)
- [Numerical Stability Guide](#numerical-stability-guide)
- [Features](#features)
- [Dependencies](#dependencies)
- [Build Instructions](#build-instructions)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [References](#references)

---

## Overview

This repository provides nonlinear filtering implementations optimized for **ARM aarch64** (Raspberry Pi 5, Orange Pi 5/6) and **x86_64** using **ARM NEON/SVE2 intrinsics** and **Vulkan compute shaders**. All implementations use single-precision floating point (`float`) for maximum SIMD vectorization efficiency.

### What's Included

- **5 Filtering Methods**: EKF, UKF, SRUKF, PKF, RBPKF
- **Fixed-Lag Smoothers**: Rauch-Tung-Striebel (RTS) backward pass and ancestry-based smoothing
- **Comprehensive Benchmarks**: 4 challenging test problems with full metrics (10D coupled oscillators, Van der Pol, bearing-only tracking, reentry vehicle)
- **Hardware Acceleration**: NEON dense linear algebra + Vulkan particle operations via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)

---

## Implemented Filters

### 1. Extended Kalman Filter (EKF)

**Jacobian-based linearization**

- **Method**: First-order Taylor series approximation
- **Requirements**: Explicit Jacobian matrices
- **Smoothing**: RTS fixed-lag backward pass
- **Best For**: Mildly nonlinear systems, fast prototyping
- **Location**: `EKF/`

### 2. Unscented Kalman Filter (UKF)

**Derivative-free sigma point method**

- **Method**: Deterministic sampling via Merwe scaled unscented transform
- **Parameters**: Dimension-adaptive (alpha=1.0, beta=2.0, kappa=3-n)
- **Smoothing**: RTS with cross-covariance tracking
- **Best For**: Highly nonlinear systems where Jacobians are unavailable or expensive
- **Location**: `UKF/`
- **Optimization**: NEON-accelerated GEMM, Cholesky, inverse, SPD solve

### 3. Square Root UKF (SRUKF)

**Numerically stable square root formulation**

- **Method**: Propagates Cholesky factor S where P = S*S^T
- **Algorithm**: QR decomposition + rank-1 Cholesky updates/downdates with safe fallbacks
- **Advantages**:
  - Guaranteed positive-definite covariance
  - Innovation gating prevents catastrophic updates
  - Safe Cholesky downdate with automatic fallback to full recomputation
- **Best For**: Mission-critical, long-duration, weak observability systems
- **Location**: `UKF/include/SRUKF.h`

### 4. Particle Filter (PKF)

**Bootstrap sequential importance resampling**

- **Method**: Monte Carlo approximation with particle ensemble
- **Resampling**: Systematic, stratified
- **Smoothing**: Ancestry-based fixed-lag trajectory reconstruction
- **Best For**: Non-Gaussian noise, multimodal distributions
- **Location**: `PKF/`
- **Optimization**: Vulkan GPU acceleration for N > 100 particles, OpenMP parallel propagation

### 5. Rao-Blackwellized Particle Filter (RBPKF)

**Hybrid particle-Kalman filter**

- **Method**: Marginalize linear substructure analytically via conditional Kalman filters
- **Structure**: Nonlinear particles + linear Kalman filters per particle
- **Advantages**: Reduced variance vs standard particle filter
- **Best For**: Systems with conditionally linear subspace (e.g., CTRV models)
- **Location**: `RBPKF/`

---

## Benchmark Results

Four challenging problems tested with UKF, SRUKF, and fixed-lag smoothers. All benchmarks run on ARM aarch64 with NEON acceleration.

### Coupled Oscillators (10D State, 5D Observation)

| Filter | RMSE | Smoothed RMSE | NEES | Avg Step Time | Divergences |
|--------|------|---------------|------|---------------|-------------|
| **UKF** | 1.457 | — | 10.58 ± 4.80 | 0.021 ms | 0 |
| **SRUKF** | 1.456 | — | 10.59 ± 4.81 | 0.019 ms | 0 |
| **UKF+Smoother** | 1.457 | **1.148** | 10.58 ± 4.80 | 0.112 ms | 0 |
| **SRUKF+Smoother** | 1.456 | **1.148** | 10.59 ± 4.81 | 0.165 ms | 0 |

Smoothing improves RMSE by **21%** on this 10-dimensional problem.

### Van der Pol Oscillator (2D State, 1D Observation)

| Filter | RMSE | Smoothed RMSE | Avg Step Time | Divergences |
|--------|------|---------------|---------------|-------------|
| **UKF** | 0.468 | — | 0.001 ms | 0 |
| **SRUKF** | 0.466 | — | 0.0006 ms | 0 |
| **SRUKF+Smoother** | 0.466 | **0.429** | 0.016 ms | 0 |

Smoothing improves RMSE by **8%**. SRUKF is slightly faster than UKF.

### Bearing-Only Tracking (4D State, 1D Observation)

| Filter | RMSE | Smoothed RMSE | NEES | Divergences |
|--------|------|---------------|------|-------------|
| **UKF** | 42.84 | — | 1.59 ± 1.12 | 171 |
| **SRUKF** | 43.00 | — | 1.57 ± 1.10 | 173 |
| **SRUKF+Smoother** | 43.00 | **36.95** | 1.57 ± 1.10 | 173 |

Weak observability problem (bearing-only). Smoothing improves RMSE by **14%**.

### Reentry Vehicle (6D State, 3D Observation)

| Filter | RMSE | Smoothed RMSE | NEES | Divergences |
|--------|------|---------------|------|-------------|
| **UKF** | 286.8 | — | 10.99 ± 1.90 | 200 |
| **SRUKF** | **277.8** | — | 11.23 ± 1.97 | 200 |
| **SRUKF+Smoother** | 277.8 | 278.8 | 11.23 ± 1.97 | 200 |

Extremely challenging problem with large state magnitudes (Earth-centered coordinates). SRUKF provides 3% better filtering than UKF.

### Filter & Smoother Test Results

| Test | Filter RMSE | Smoother RMSE | Improvement |
|------|-------------|---------------|-------------|
| EKF (Nonlinear Oscillator, 2D) | 0.060 | 0.052 | 13% |
| UKF (Drag Ball, 4D) | 0.228 | 0.119 | 48% |
| SRUKF (Drag Ball, 4D) | 0.355 | 0.166 | 53% |
| PKF (Lorenz-63, 3D) | 0.806 | 0.595 | 26% |

---

## Numerical Stability Guide

This section documents real numerical issues encountered during development.

### Issue #1: Sigma Point Weight Explosion

**Problem**: With alpha=1e-3, the central sigma point weight becomes extremely negative.

**Root Cause**: When alpha^2*(n+kappa) is approximately equal to n, the denominator n+lambda approaches zero.

**Solution**: Use alpha=1.0 and kappa=3-n, with protection against degenerate spread:
```cpp
float n_lambda = n + lambda;
if (n_lambda < 0.5f) {
    kappa = n / (alpha * alpha) - n;
    lambda = alpha * alpha * (n + kappa) - n;
    n_lambda = n + lambda;
}
```

### Issue #2: Cholesky Downdate Instability

**Problem**: Cholesky downdate can produce negative diagonal elements when the update magnitude exceeds the current factor.

**Solution**: Safe downdate with detection and fallback:
```cpp
float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
if (r_sq <= 0) {
    // Downdate failed — recompute P directly and take fresh Cholesky
    return false;
}
```

### Issue #3: Innovation Gating for SRUKF

**Problem**: Large outlier measurements cause catastrophic state updates.

**Solution**: Mahalanobis distance gating scales down corrections:
```cpp
float mahal_dist_sq = temp_innov.squaredNorm();
if (mahal_dist_sq > gate_threshold) {
    scale = std::sqrt(gate_threshold / mahal_dist_sq);
}
State correction = scale * (K * innovation);
```

### Numerical Health Checklist

Before deploying any Kalman filter, verify:

- Sigma point weights sum to 1 for Wm
- Central weight Wc(0) is reasonable (not exploding)
- Covariance diagonal elements are positive
- Innovation covariance >= measurement noise R
- Kalman gain is bounded
- State estimates don't diverge

---

## Features

### Hardware Optimization

- **ARM NEON Dense Linear Algebra**: Cholesky, matrix inverse, GEMM, mat-vec multiply, SPD solve via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
- **SVE2 Support**: Scalable Vector Extension 2 with FCMA and I8MM on supported platforms
- **Vulkan Compute**: Particle operations parallelized on GPU
- **Graceful Fallback**: NEON -> NEON+jitter -> Eigen LLT/LDLT for numerical robustness
- **Single Precision**: Consistent use of `float` for SIMD vectorization

### Software Quality

- **C++20**: Modern features (concepts, ranges, template constraints)
- **Type Safety**: Template metaprogramming for compile-time dimension checking
- **Joseph Form**: Numerically stable covariance update throughout
- **OpenMP**: Parallel particle propagation and weight updates

---

## Dependencies

### Required

- **C++20 Compiler**: GCC 10+, Clang 11+
- **Eigen3**: Linear algebra (3.4+, fetched automatically if not found)
- **CMake**: Build system (3.14+)
- **[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)**: NEON/SVE2/Vulkan acceleration (cloned to `$HOME`, fetched via FetchContent)

### Optional

- **ARM NEON/SVE2**: Automatic on ARM aarch64 platforms
- **Vulkan SDK**: For GPU-accelerated particle filter (1.3+)
- **OpenMP**: For parallel particle filter
- **Python 3 + Matplotlib**: For visualization scripts

### Installation (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake libeigen3-dev
sudo apt install python3 python3-matplotlib  # Optional: for plots

# Clone the OptimizedKernels dependency
cd ~
git clone https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA.git
```

---

## Build Instructions

```bash
# Clone repository
git clone https://github.com/n4hy/Modern-Computational-Nonlinear-Filtering.git
cd Modern-Computational-Nonlinear-Filtering

# Create build directory
mkdir -p build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
./EKF/ekf_test
./UKF/ukf_test
./UKF/srukf_test
./PKF/pkf_test
./RBPKF/test_rbpf_basic

# Run benchmarks
./Benchmarks/run_benchmarks
```

### Build Outputs

| Target | Description |
|--------|-------------|
| `EKF/ekf_test` | EKF with fixed-lag smoother on nonlinear oscillator |
| `UKF/ukf_test` | UKF with fixed-lag smoother on drag ball model |
| `UKF/srukf_test` | SRUKF with fixed-lag smoother on drag ball model |
| `PKF/pkf_test` | Particle filter unit tests |
| `PKF/pkf_example` | Particle filter on Lorenz-63 attractor |
| `RBPKF/test_rbpf_basic` | RBPF unit tests |
| `RBPKF/example_rbpf_ctrv` | RBPF on CTRV vehicle model |
| `Benchmarks/run_benchmarks` | Full benchmark suite (4 problems, 4 filters) |

---

## Usage Examples

### Basic SRUKF Usage

```cpp
#include "SRUKF.h"
#include "MyModel.h"  // Your state-space model

int main() {
    // Define model (must inherit from StateSpaceModel<NX, NY>)
    MyModel<4, 2> model;  // 4 states, 2 observations

    // Create SRUKF filter
    UKFCore::SRUKF<4, 2> filter(model);

    // Initialize
    Eigen::Vector4f x0;
    x0 << 0, 0, 10, 15;
    Eigen::Matrix4f P0 = Eigen::Matrix4f::Identity();
    filter.initialize(x0, P0);

    // Process measurements
    Eigen::Vector4f u = Eigen::Vector4f::Zero();
    for (int k = 0; k < num_steps; ++k) {
        filter.predict(time[k], u);
        filter.update(time[k], measurements[k]);

        auto x_est = filter.getState();
        auto P_est = filter.getCovariance();  // Reconstructs P = S*S^T
    }
}
```

### SRUKF with Fixed-Lag Smoother

```cpp
#include "SRUKFFixedLagSmoother.h"
#include "MyModel.h"

int main() {
    MyModel<4, 2> model;
    int lag = 50;

    UKFCore::SRUKFFixedLagSmoother<4, 2> smoother(model, lag);
    smoother.initialize(x0, P0);

    Eigen::Vector4f u = Eigen::Vector4f::Zero();
    for (int k = 0; k < num_steps; ++k) {
        smoother.step(time[k], measurements[k], u);

        auto x_filt = smoother.get_filtered_state();
        auto x_smooth = smoother.get_smoothed_state(lag);  // Lag steps back
    }
}
```

### Custom Model Implementation

```cpp
#include "StateSpaceModel.h"

template<int NX = 4, int NY = 1>
class BearingOnlyTracking : public UKFModel::StateSpaceModel<NX, NY> {
public:
    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.1f;

    // Process model: constant velocity
    State f(const State& x, float t,
            const Eigen::Ref<const State>& u) const override {
        State x_next;
        x_next(0) = x(0) + dt * x(2);  // px += vx * dt
        x_next(1) = x(1) + dt * x(3);  // py += vy * dt
        x_next(2) = x(2);
        x_next(3) = x(3);
        return x_next;
    }

    // Observation model: bearing angle
    Observation h(const State& x, float t) const override {
        Observation y;
        y(0) = std::atan2(x(1), x(0));
        return y;
    }

    StateMat Q(float t) const override {
        StateMat q = StateMat::Zero();
        q(2, 2) = 0.1f;
        q(3, 3) = 0.1f;
        return q;
    }

    ObsMat R(float t) const override {
        ObsMat r;
        r(0, 0) = 0.01f;
        return r;
    }
};
```

---

## Architecture

```
Modern-Computational-Nonlinear-Filtering/
├── Common/                     # Shared interfaces
│   └── include/
│       ├── StateSpaceModel.h   # Base model for UKF/SRUKF
│       ├── SystemModel.h       # Base model for EKF
│       └── FileUtils.h         # File I/O utilities
│
├── EKF/                        # Extended Kalman Filter
│   ├── include/
│   │   ├── EKF.h
│   │   ├── EKFFixedLag.h
│   │   ├── BallTossModel.h
│   │   └── NonlinearOscillator.h
│   ├── src/
│   │   ├── EKF.cpp
│   │   └── EKFFixedLag.cpp
│   └── main.cpp
│
├── UKF/                        # Unscented Kalman Filters
│   ├── include/
│   │   ├── UKF.h               # Standard UKF
│   │   ├── SRUKF.h             # Square Root UKF
│   │   ├── SigmaPoints.h       # Sigma point generation
│   │   ├── UnscentedFixedLagSmoother.h
│   │   ├── SRUKFFixedLagSmoother.h
│   │   └── DragBallModel.h     # Example model
│   ├── main.cpp                # UKF + smoother test
│   └── main_srukf.cpp          # SRUKF + smoother test
│
├── PKF/                        # Particle Filter
│   ├── include/
│   │   ├── particle_filter.hpp
│   │   ├── particle_fixed_lag.hpp
│   │   ├── resampling.hpp
│   │   ├── state_space_model.hpp
│   │   ├── noise_models.hpp
│   │   └── lorenz63_model.hpp
│   ├── src/
│   │   └── example_main.cpp
│   └── tests/
│       └── test_particle.cpp
│
├── RBPKF/                      # Rao-Blackwellized Particle Filter
│   ├── include/rbpf/
│   │   ├── rbpf_core.hpp
│   │   ├── rbpf_config.hpp
│   │   ├── kalman_filter.hpp
│   │   ├── resampling.hpp
│   │   ├── state_space_models.hpp
│   │   └── types.hpp
│   ├── src/
│   │   └── resampling.cpp
│   ├── examples/
│   │   └── example_rbpf_ctrv.cpp
│   └── tests/
│       └── test_rbpf_basic.cpp
│
├── Benchmarks/                 # Comprehensive test suite
│   ├── include/
│   │   ├── BenchmarkProblems.h # 4 test problems
│   │   └── BenchmarkRunner.h   # Metrics framework
│   └── src/
│       └── run_benchmarks.cpp
│
├── scripts/                    # Visualization
│   ├── simple_plot_benchmarks.py
│   ├── plot_benchmarks.py
│   ├── pkf_plot_results.py
│   └── ukf_plot_results.py
│
├── docs/images/                # Generated benchmark plots
├── CMakeLists.txt              # Top-level build
└── LICENSE
```

---

## Contributing

Contributions welcome. Areas of interest:

1. **Additional Test Problems**: More challenging benchmark scenarios
2. **GPU Optimization**: Vulkan/CUDA shaders for UKF/SRUKF sigma point propagation
3. **Adaptive Methods**: Automatic parameter tuning (adaptive Q/R)
4. **Multi-Sensor Fusion**: Asynchronous measurement handling
5. **Extended Benchmarks**: Monte Carlo consistency analysis, filter divergence studies

Please:
- Follow C++20 style guidelines
- Add tests for new features
- Document numerical considerations

---

## References

### Square Root Filtering

1. Bierman, G.J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
2. Van der Merwe, R., Wan, E. (2001). "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation". IEEE ICASSP.

### Unscented Transform

3. Julier, S.J., Uhlmann, J.K. (1997). "New Extension of the Kalman Filter to Nonlinear Systems". SPIE AeroSense.
4. Wan, E.A., van der Merwe, R. (2000). "The Unscented Kalman Filter for Nonlinear Estimation". IEEE Adaptive Systems for Signal Processing.

### Particle Filtering

5. Doucet, A., de Freitas, N., Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
6. Schon, T., Gustafsson, F., Nordlund, P.J. (2005). "Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models". IEEE TSP.

### Numerical Stability

7. Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
8. Golub, G.H., Van Loan, C.F. (2013). *Matrix Computations*. Johns Hopkins University Press.

---

## License

MIT License - see LICENSE file for details.

---

**Version**: 2.4.0
**Last Updated**: March 2026
**Platform**: ARM aarch64 (Raspberry Pi 5, Orange Pi 5/6) + x86_64
