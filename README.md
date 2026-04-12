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
- **Optimization**: SVE2/NEON-accelerated GEMM, Cholesky, SPD solve via FilterMath dispatch

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

Four challenging problems tested with UKF, SRUKF, and fixed-lag smoothers. All benchmarks run on ARM aarch64 with SVE2/NEON acceleration via the FilterMath dispatch layer.

### Coupled Oscillators (10D State, 5D Observation)

| Filter | RMSE | Smoothed RMSE | NEES median | In 95% bounds | Avg Step Time | Divergences |
|--------|------|---------------|-------------|---------------|---------------|-------------|
| **UKF** | 1.457 | — | 9.89 | 94.5% | 0.025 ms | 0 |
| **SRUKF** | 1.457 | — | 9.89 | 94.5% | 0.017 ms | 0 |
| **UKF+Smoother** | 1.457 | **1.148** | 9.89 | 94.5% | 0.141 ms | 0 |
| **SRUKF+Smoother** | 1.457 | **1.148** | 9.89 | 94.5% | 0.187 ms | 0 |

Smoothing improves RMSE by **21%** on this 10-dimensional problem. NEES 94.5% in chi-squared bounds indicates excellent filter consistency.

### Van der Pol Oscillator (2D State, 1D Observation)

| Filter | RMSE | Smoothed RMSE | NEES median | In 95% bounds | Avg Step Time | Divergences |
|--------|------|---------------|-------------|---------------|---------------|-------------|
| **UKF** | 0.468 | — | 1.14 | 95.9% | 0.001 ms | 0 |
| **SRUKF** | 0.466 | — | 1.14 | 96.0% | 0.0005 ms | 0 |
| **SRUKF+Smoother** | 0.466 | **0.430** | 1.14 | 96.0% | 0.022 ms | 0 |

Smoothing improves RMSE by **8%**. SRUKF is 2.9x faster than UKF on this 2D problem.

### Bearing-Only Tracking (4D State, 1D Observation)

| Filter | RMSE | Smoothed RMSE | NEES median | In 95% bounds | Divergences |
|--------|------|---------------|-------------|---------------|-------------|
| **UKF** | 63.81 | — | 3.77 | 99.6% | 176 |
| **SRUKF** | 64.17 | — | 3.77 | 99.6% | 175 |
| **SRUKF+Smoother** | 64.17 | **52.03** | 3.77 | 99.6% | 175 |

Weak observability problem (bearing-only). Smoothing improves RMSE by **19%**. "Divergences" reflect inherently weak observability during early trajectory, not filter instability.

### Reentry Vehicle (6D State, 3D Observation)

| Filter | RMSE | Smoothed RMSE | NEES median | In 95% bounds | Divergences |
|--------|------|---------------|-------------|---------------|-------------|
| **UKF** | 369.0 m | — | 4.99 | 95.9% | 0 |
| **SRUKF** | 369.2 m | — | 4.99 | 95.6% | 0 |
| **SRUKF+Smoother** | 369.2 m | **236.8 m** | 4.99 | 95.6% | 0 |

Realistic spacecraft reentry tracking with altitude-dependent gravity (gravitational parameter μ/r²), exponential atmosphere drag model, and radar on Earth's surface. NEES median 4.99 (expected ~6) with 95.6% in bounds indicates excellent filter consistency. Smoothing improves RMSE by **36%**.

### Filter & Smoother Test Results

| Test | Filter RMSE | Smoother RMSE | Improvement |
|------|-------------|---------------|-------------|
| EKF (Nonlinear Oscillator, 2D) | 0.060 | 0.052 | 13% |
| UKF (Drag Ball, 4D) | 0.228 | 0.119 | 48% |
| SRUKF (Drag Ball, 4D) | 0.354 | 0.165 | 53% |
| PKF (Lorenz-63, 3D) | 0.802 | 0.600 | 25% |

### v3.0.0 Before/After Comparison

Comparison between v2.8.0 (commit `12b8015`, direct NEON calls, `-ffast-math`) and v3.0.0 (commit `2d87c48`, FilterMath SVE2/NEON dispatch, bug fixes). Same hardware, same seeds, same trajectories.

#### Efficiency — Total Benchmark Time (ms)

| Filter + Problem | v2.8.0 | v3.0.0 | Change |
|---|---:|---:|---:|
| UKF — Coupled Osc 10D | 135.1 | **109.9** | **-18.7%** |
| SRUKF — Coupled Osc 10D | 89.3 | 90.6 | +1.4% |
| UKF+Smoother — Coupled Osc 10D | 574.9 | 739.1 | +28.6%† |
| SRUKF+Smoother — Coupled Osc 10D | 821.7 | 974.7 | +18.6%† |
| SRUKF — Van der Pol 2D | 0.94 | 1.17 | +0.23ms‡ |
| SRUKF — Bearing-Only 4D | 0.30 | 0.36 | +0.06ms‡ |
| SRUKF — Reentry 6D | 1.22 | 1.30 | +0.08ms‡ |

> † Smoother slowdown from replacing `neon_inverse()` with numerically stable `solve_spd()` (SPD triangular solve). Correctness over speed.
>
> ‡ Sub-millisecond absolute differences; dominated by measurement noise.
>
> **Key win**: UKF 10D **18.7% faster** — SVE2 cache-blocked GEMM (tuned for A720 12MB L3) paying off on the largest matrix operations.

#### Accuracy — RMSE (identical seeds)

| Filter + Problem | v2.8.0 | v3.0.0 |
|---|---:|---:|
| UKF — Coupled Osc 10D | 1.4566 | 1.4567 |
| SRUKF — Coupled Osc 10D | 1.4566 | 1.4567 |
| UKF — Van der Pol 2D | 0.4681 | 0.4681 |
| SRUKF — Bearing-Only 4D | 43.151 | 43.151 |
| SRUKF — Reentry 6D | 369.21 | 369.18 |

Accuracy preserved to floating-point precision across all problems. NEES consistency unchanged (all pass chi-squared bounds).

#### Robustness — Bug Fixes

| Bug Fixed | Before (silent failure mode) | After |
|---|---|---|
| Global `-ffast-math` | `isfinite()` guards compiled out — NaN propagates undetected | NaN guards work correctly |
| Unsafe `cholupdate_downdate` | Covariance silently corrupted when downdate magnitude too large | Safe version + full-covariance fallback |
| SRUKF S_yy NaN check | Only diagonal checked — off-diagonal NaN/Inf missed | `allFinite()` on entire matrix |
| RBPKF weight NaN | One NaN particle corrupts all weights | `isfinite()` guard + uniform fallback |
| RBPKF ESS div-by-zero | Returns Inf, breaks resampling threshold | Returns N (forces resampling) |
| RBPKF resampling bias | O(N) float rounding — last particles underselected | Kahan compensated summation |
| Cross-platform build | Direct `optmath::neon::*` — fails on x86 | `#if FILTERMATH_ARM64` + Eigen fallback |

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

### Issue #4: Eigen Expression Template Aliasing in SRUKF Mean Computation

**Problem**: SRUKF predict step returned INPUT state instead of PROPAGATED state, causing INS flywheel to fail during GPS outage. Position error grew to 3km in 30 seconds instead of expected ~1km.

**Root Cause**: Eigen's expression templates caused aliasing between `X_pred` (propagated sigma points) and `sigmas.X` (input sigma points) during weighted mean computation. The operation `x_pred_mean += Wm(i) * X_pred.col(i)` was reading from `sigmas.X` memory instead of `X_pred`, even though they were separate stack-allocated matrices.

**Symptoms**:
- State appears unchanged after predict() during measurement outage
- Debug shows X_pred values are correct, but weighted mean equals input
- Double-precision accumulation gives correct result while float loop gives wrong result

**Solution**: Force evaluation with `.eval()` to materialize the expression and break aliasing, combined with `.noalias()` for safe accumulation. This preserves NEON/SVE2 auto-vectorization (unlike the earlier raw C array workaround):
```cpp
// Force evaluation to break Eigen aliasing while keeping SIMD vectorization
typename SigmaPts::SigmaMat X_pred_eval = X_pred.eval();
typename SigmaPts::Weights Wm_eval = sigmas.Wm.eval();

State x_pred_mean = State::Zero();
for (int i = 0; i < SigmaPts::NSIG; ++i) {
    x_pred_mean.noalias() += Wm_eval(i) * X_pred_eval.col(i);
}
```

**Result**: Max error during 30s GPS outage reduced from 3097m to ~1080m.

### Issue #5: Monte Carlo Trial Divergence - RESOLVED

**Problem**: After GPS outage, innovation gating in SRUKF prevented GPS reacquisition.

**Root Causes Identified**:
1. **Innovation gating too aggressive**: After 30s GPS outage, INS drifted ~3km. When GPS returned, the Mahalanobis distance was huge, causing updates to be scaled down to ~0.7% effectiveness
2. **Compiler flags**: `EIGEN_NO_DEBUG` and `-ffast-math` caused numerical instability in edge cases

**Solution Implemented**:
1. Measurement outage recovery: When measurements return after outage with large position error, reinitialize filter around measurements instead of trying to correct with gated updates
2. Disabled `-ffast-math` and `EIGEN_NO_DEBUG` for numerically sensitive targets
3. All Cholesky operations now route through FilterMath dispatch (NEON-accelerated with Eigen fallback)

**Result**: **100% convergence** across Monte Carlo trials.

### Issue #6: Global `-ffast-math` Breaks Filter Stability

**Problem**: CMake root had `-ffast-math` and `EIGEN_FAST_MATH=1` applied globally to all Release builds. This silently caused:
- NaN comparison guards (`isfinite()`, `allFinite()`) to be optimized away
- Cholesky decomposition precision loss from altered floating-point associativity
- Denormal flushing that corrupted small covariance values

**Solution**: Removed global `-ffast-math` and `EIGEN_FAST_MATH=1` from root `CMakeLists.txt`. The NEON/SVE2 intrinsics in OptMathKernels already provide hardware-accelerated fast paths where needed. Numerically sensitive targets should explicitly set `-fno-fast-math` and `EIGEN_FAST_MATH=0`.

### Issue #7: Unsafe Cholesky Downdate in SRUKF Prediction

**Problem**: The SRUKF predict step used the legacy `cholupdate_downdate()` which silently corrupted the square root covariance when the downdate magnitude exceeded the current factor (sets `r_sq = 1e-6 * S(k,k)²` instead of failing).

**Solution**: Replaced with `cholupdate_downdate_safe()` which returns false on failure, plus a full-covariance fallback that recomputes P from all sigma points and takes a fresh Cholesky.

### Issue #8: RBPKF Weight Corruption and Resampling Bias

**Problem**: Three related issues in the Rao-Blackwellized particle filter:
1. `normalize_weights()` did not check `isfinite()` — a single NaN weight corrupted all particles
2. `get_effective_sample_size()` could divide by zero when all weights collapsed
3. Cumulative sum in resampling used naive float addition, accumulating O(N) rounding error that systematically underselected the last particles

**Solution**:
1. Added `isfinite()` guard with uniform-weight fallback for degenerate cases
2. Added `sum_sq <= 0` guard returning `N` (forces resampling)
3. Replaced naive cumulative sum with Kahan compensated summation in both systematic and stratified resampling

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

- **FilterMath Dispatch Layer** (`Common/include/FilterMath.h`): Unified API that automatically selects the best backend at runtime:
  - **GEMM**: SVE2 cache-blocked → NEON blocked → Eigen (SVE2 tuned for A720's 12MB L3)
  - **Cholesky / Inverse / Solve**: NEON accelerated → Eigen LDLT fallback
  - **Kalman Gain**: SPD solve (avoids explicit matrix inverse for O(n²) vs O(n³))
  - **Non-ARM platforms**: All paths fall through to pure Eigen — full cross-platform support
- **ARM NEON Dense Linear Algebra**: Cholesky, GEMM, mat-vec multiply, SPD solve via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
- **ARM SVE2**: Cache-blocked GEMM with FCMA and I8MM on Cortex-A720+ (Orange Pi 5/6)
- **Vulkan Compute**: Particle filter noise addition parallelized on GPU (Mali-G720, VideoCore VII)
- **Graceful Fallback**: Accelerated → jitter + retry → Eigen LLT/LDLT for numerical robustness
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
│       ├── FilterMath.h        # SVE2/NEON/Eigen dispatch layer
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

**Version**: 3.1.0
**Last Updated**: April 2026
**Platform**: ARM aarch64 (Raspberry Pi 5, Orange Pi 5/6) + x86_64 (Eigen fallback)
