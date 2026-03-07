# Modern Computational Nonlinear Filtering

<div align="center">

**High-Performance Nonlinear State Estimation for Embedded Systems**

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20%2B%20x86__64-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optimization](https://img.shields.io/badge/Optimization-NEON%20%2B%20Vulkan-orange.svg)](https://developer.arm.com/Architectures/Neon)

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

This repository provides nonlinear filtering implementations optimized for **Raspberry Pi 5** and **x86_64** using **ARM NEON intrinsics** and **Vulkan compute shaders**. All implementations use single-precision floating point (`float`) for maximum SIMD vectorization efficiency.

### What's Included

- **5 Filtering Methods**: EKF, UKF, SRUKF, PKF, RBPKF
- **Fixed-Lag Smoothers**: Rauch-Tung-Striebel (RTS) and ancestry-based smoothing
- **Comprehensive Benchmarks**: Multiple challenging test problems with full metrics
- **Hardware Acceleration**: NEON (matrix operations) + Vulkan (particle operations)
- **Iridium Satellite Tracking**: Complete UKF/SRUKF implementation for AOA+Doppler tracking

### Current Status (March 2026)

**Production Ready**:
- EKF, UKF, and SRUKF for dimensions NX <= 5, NY <= 3
- SRUKF shows 98.6% RMSE improvement on bearing-only tracking vs UKF
- EKF vs UKF vs SRUKF benchmarks completed for Iridium satellite tracking

**Known Limitations**:
- High-dimensional SRUKF (>5D) requires specialized tuning (see `SRUKF_STATUS.md`)

See `COMPARISON_RESULTS.md` for detailed benchmark analysis.

---

## Implemented Filters

### 1. Extended Kalman Filter (EKF)

**Jacobian-based linearization**

- **Method**: First-order Taylor series approximation
- **Requirements**: Explicit Jacobian matrices (can use numerical differentiation)
- **Best For**: Mildly nonlinear systems, fast prototyping
- **Smoothing**: RTS fixed-lag backward pass
- **Location**: `EKF/`

### 2. Unscented Kalman Filter (UKF)

**Derivative-free sigma point method**

- **Method**: Deterministic sampling via unscented transform
- **Parameters**: alpha=1.0, beta=2.0, kappa=3-n for numerical stability
- **Best For**: Highly nonlinear systems where Jacobians are unavailable
- **Smoothing**: RTS with cross-covariance tracking
- **Location**: `UKF/`
- **Optimization**: NEON-accelerated sigma point propagation

### 3. Square Root UKF (SRUKF)

**Numerically stable square root formulation**

- **Method**: Propagates Cholesky factor S where P = S*S^T
- **Algorithm**: QR decomposition + rank-1 Cholesky updates
- **Advantages**:
  - Guaranteed positive-definite covariance (mathematically proven)
  - 43% faster than standard UKF (NEON optimized)
  - Better numerical conditioning for weak observability
- **Best For**: Mission-critical, long-duration, weak observability systems
- **Location**: `UKF/include/SRUKF.h`

### 4. Particle Filter (PKF)

**Bootstrap sequential importance resampling**

- **Method**: Monte Carlo approximation with particle ensemble
- **Resampling**: Systematic, stratified
- **Smoothing**: Ancestry-based trajectory reconstruction
- **Best For**: Non-Gaussian noise, multimodal distributions
- **Location**: `PKF/`
- **Optimization**: Vulkan GPU acceleration for N > 100 particles

### 5. Rao-Blackwellized Particle Filter (RBPKF)

**Hybrid particle-Kalman filter**

- **Method**: Marginalize linear substructure analytically
- **Structure**: Nonlinear particles + linear Kalman filters
- **Advantages**: Reduced variance vs standard particle filter
- **Best For**: Systems with linear subspace
- **Location**: `RBPKF/`

---

## Benchmark Results

### EKF vs UKF vs SRUKF: Iridium Satellite Tracking

The following results compare filter performance on satellite tracking with extreme initial position errors (~20 deg off, ~600 km altitude error):

| Filter | RMS Position Error | Mean Time/Step | Notes |
|--------|-------------------|----------------|-------|
| **EKF** | 117.6 km | 0.33 ms | Fast but less accurate |
| **UKF** | 107.1 km | 0.76 ms | 9% better than EKF |
| **SRUKF** | 72.5 km | 1.02 ms | 38% better than EKF |

**Key Finding**: SRUKF provides the best accuracy for satellite tracking scenarios with weak observability, though at higher computational cost.

### Standard Benchmark Problems

#### Bearing-Only Tracking (4D State, 1D Observation)

![Bearing-Only UKF](docs/images/bearing_ukf_plot.png)
![Bearing-Only SRUKF](docs/images/bearing_srukf_plot.png)

| Metric | UKF | SRUKF | Improvement |
|--------|-----|-------|-------------|
| **RMSE** | 1229 m | 17 m | 98.6% better |
| **Divergences** | 284 | 182 | 36% reduction |
| **Speed** | 0.022 ms/step | 0.012 ms/step | 43% faster |

#### Van der Pol Oscillator (2D State, 1D Observation)

![Van der Pol UKF](docs/images/vanderpol_ukf_plot.png)
![Van der Pol SRUKF](docs/images/vanderpol_srukf_plot.png)

- Both filters perform well on this problem
- SRUKF is 37% faster (0.0035 ms vs 0.0056 ms per step)

#### Coupled Oscillators (10D State, 5D Observation)

![Coupled Oscillators UKF](docs/images/coupled_osc_ukf_plot.png)
![Coupled Oscillators SRUKF](docs/images/coupled_osc_srukf_plot.png)

- UKF performs well (RMSE: 1.46, 0 divergences)
- SRUKF requires specialized tuning for high-dimensional problems

### Performance Summary

![Performance Comparison](docs/images/performance_comparison.png)
![Summary Table](docs/images/summary_table.png)

---

## Numerical Stability Guide

This section documents real numerical issues encountered during development.

### Issue #1: Sigma Point Weight Explosion

**Problem**: With alpha=1e-3, the central sigma point weight becomes extremely negative.

**Root Cause**: When alpha^2*(n+kappa) is approximately equal to n, the denominator n+lambda approaches zero.

**Solution**: Use alpha=1.0 and kappa=3-n, or add protection:
```cpp
if (std::abs(n + lambda) < 0.1f) {
    kappa = (0.1f / (alpha * alpha)) - n + 1.0f;
    lambda = alpha * alpha * (n + kappa) - n;
}
```

### Issue #2: QR Decomposition for 1D Observations

**Problem**: QR decomposition of a 1xN matrix returns the input unchanged.

**Solution**: Use direct covariance computation for NY==1:
```cpp
if constexpr (NY == 1) {
    float P_yy = 0.0f;
    for (int i = 0; i < NSIG; ++i) {
        float diff_y = Y_pred(0, i) - y_hat(0);
        P_yy += Wc[i] * diff_y * diff_y;
    }
    P_yy += R(0, 0);
    S_yy(0, 0) = sqrt(P_yy);
}
```

### Issue #3: Cholesky Downdate Instability

**Problem**: Cholesky downdate can produce negative diagonal elements.

**Solution**: Clamp to small positive value:
```cpp
float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
if (r_sq <= 0) {
    r_sq = 1e-8f;
}
```

### Numerical Health Checklist

Before deploying any Kalman filter, verify:

- Sigma point weights sum to 1 for Wm
- Central weight Wc(0) is reasonable (not exploding)
- Covariance diagonal elements are positive
- Innovation covariance >= measurement noise R
- Kalman gain is bounded
- State estimates don't explode

---

## Features

### Hardware Optimization

- **ARM NEON Dense Linear Algebra**: Cholesky, matrix inverse, GEMM, SPD solve via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
- **Vulkan Compute**: Particle operations parallelized on GPU
- **Graceful Fallback**: NEON -> NEON+jitter -> Eigen LLT/LDLT for numerical robustness
- **Single Precision**: Consistent use of `float` for SIMD vectorization

### Software Quality

- **C++20**: Modern features (concepts, ranges, template constraints)
- **Type Safety**: Template metaprogramming for compile-time dimension checking
- **Exception Safety**: RAII, no raw pointers, proper resource management
- **Extensive Testing**: Unit tests, integration tests, benchmark validation

---

## Dependencies

### Required

- **C++20 Compiler**: GCC 10+, Clang 11+
- **Eigen3**: Linear algebra (3.4+)
- **CMake**: Build system (3.14+)
- **[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)**: NEON/Vulkan acceleration (fetched automatically)

### Optional

- **ARM NEON**: Automatic on ARM platforms (Raspberry Pi 5)
- **Vulkan SDK**: For GPU-accelerated particle filter (1.3+)
- **Python 3 + Matplotlib**: For visualization scripts
- **OpenMP**: For parallel particle filter

### Installation (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake libeigen3-dev
sudo apt install python3 python3-matplotlib  # For plots
```

---

## Build Instructions

```bash
# Clone repository
git clone https://github.com/n4hy/Modern-Computational-Nonlinear-Filtering.git
cd Modern-Computational-Nonlinear-Filtering

# Create build directory
mkdir -p build && cd build

# Configure
cmake ..

# Build all targets
make -j$(nproc)

# Run benchmarks
./Benchmarks/run_benchmarks

# Generate visualizations
python3 ../scripts/simple_plot_benchmarks.py .
```

### Build Outputs

- `./UKF/ukf_test` - UKF standalone test
- `./UKF/srukf_test` - SRUKF standalone test
- `./UKF/benchmark_ukf_vs_srukf` - EKF vs UKF vs SRUKF comparison
- `./UKF/benchmark_aoa_doppler_comparison` - AOA+Doppler 4-way comparison
- `./EKF/ekf_test` - EKF test
- `./Benchmarks/run_benchmarks` - Full benchmark suite

---

## Usage Examples

### Basic SRUKF Usage

```cpp
#include "SRUKF.h"
#include "MyModel.h"  // Your state-space model

int main() {
    // Define model (must inherit from StateSpaceModel<NX, NY>)
    MyModel<4, 1> model;  // 4 states, 1 observation

    // Create SRUKF filter
    UKFCore::SRUKF<4, 1> filter(model);

    // Initial state and covariance
    Eigen::Vector4f x0;
    x0 << 0, 0, 1, 0;

    Eigen::Matrix4f P0 = 10.0f * Eigen::Matrix4f::Identity();

    // Initialize
    filter.initialize(x0, P0);

    // Process measurements
    for (int k = 0; k < num_steps; ++k) {
        Eigen::Vector4f u = get_control(k);
        Eigen::Vector1f z = get_measurement(k);

        filter.predict(time[k], u);
        filter.update(time[k], z);

        auto x_est = filter.getState();
        auto P_est = filter.getCovariance();
    }

    return 0;
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

    // Process model: x_{k+1} = f(x_k, u_k)
    State f(const State& x, float t,
            const Eigen::Ref<const State>& u) const override {
        State x_next;
        x_next(0) = x(0) + dt * x(2);  // px += vx * dt
        x_next(1) = x(1) + dt * x(3);  // py += vy * dt
        x_next(2) = x(2);              // vx (constant)
        x_next(3) = x(3);              // vy (constant)
        return x_next;
    }

    // Observation model: z_k = h(x_k)
    Observation h(const State& x, float t) const override {
        Observation y;
        Eigen::Vector2f rel_pos(x(0) - obs_x(t), x(1) - obs_y(t));
        y(0) = std::atan2(rel_pos(1), rel_pos(0));
        return y;
    }

    // Process noise covariance
    StateMat Q(float t) const override {
        StateMat q = StateMat::Zero();
        q(2, 2) = 0.1f;
        q(3, 3) = 0.1f;
        return q;
    }

    // Measurement noise covariance
    ObsMat R(float t) const override {
        ObsMat r;
        r(0, 0) = 0.01f;  // 0.1 rad std dev
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
│       ├── StateSpaceModel.h   # Base model interface
│       ├── SystemModel.h       # System model interface
│       └── FileUtils.h         # File I/O utilities
│
├── EKF/                        # Extended Kalman Filter
│   ├── include/
│   │   ├── EKF.h               # EKF implementation
│   │   ├── EKFFixedLag.h       # Fixed-lag EKF
│   │   ├── FixedLagSmoother.h  # RTS smoother
│   │   ├── BallTossModel.h     # Example model
│   │   └── NonlinearOscillator.h
│   ├── src/
│   │   ├── EKF.cpp
│   │   ├── EKFFixedLag.cpp
│   │   └── FixedLagSmoother.cpp
│   └── main.cpp
│
├── UKF/                        # Unscented Kalman Filter
│   ├── include/
│   │   ├── UKF.h               # Standard UKF
│   │   ├── SRUKF.h             # Square Root UKF
│   │   ├── SigmaPoints.h       # Sigma point generation
│   │   ├── UnscentedFixedLagSmoother.h
│   │   ├── SRUKFFixedLagSmoother.h
│   │   ├── DragBallModel.h     # Example model
│   │   ├── IridiumSatelliteModel.h  # Iridium tracking model
│   │   └── SRUKF_IridiumTracker.h   # Iridium SRUKF wrapper
│   ├── main.cpp                # UKF test
│   ├── main_srukf.cpp          # SRUKF test
│   ├── benchmark_ukf_vs_srukf.cpp      # EKF/UKF/SRUKF comparison
│   └── benchmark_aoa_doppler_comparison.cpp  # AOA+Doppler comparison
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
│   │   ├── BenchmarkProblems.h # Test problems
│   │   └── BenchmarkRunner.h   # Metrics framework
│   ├── src/
│   │   └── run_benchmarks.cpp
│   └── README.md
│
├── scripts/                    # Visualization
│   ├── simple_plot_benchmarks.py
│   ├── plot_benchmarks.py
│   ├── pkf_plot_results.py
│   ├── ukf_plot_results.py
│   ├── plot_results.py
│   └── plot_optimized.py
│
├── docs/
│   └── images/                 # Generated plots
│
├── CMakeLists.txt              # Top-level build
├── SRUKF_STATUS.md             # SRUKF implementation status
├── COMPARISON_RESULTS.md       # Detailed benchmark analysis
├── FINAL_AUDIT_SUMMARY.md      # Code audit results
└── LICENSE
```

---

## Contributing

Contributions welcome. Areas of interest:

1. **Additional Test Problems**: More challenging benchmark scenarios
2. **GPU Optimization**: Vulkan shaders for UKF/SRUKF
3. **Adaptive Methods**: Automatic parameter tuning
4. **Multi-Sensor Fusion**: Asynchronous measurement handling
5. **High-Dimensional SRUKF**: Option B implementation (see SRUKF_STATUS.md)

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

### Numerical Stability

5. Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
6. Golub, G.H., Van Loan, C.F. (2013). *Matrix Computations*. Johns Hopkins University Press.

---

## License

MIT License - see LICENSE file for details.

---

**Version**: 2.2.0
**Last Updated**: March 2026
**Platform**: Raspberry Pi 5 + x86_64

