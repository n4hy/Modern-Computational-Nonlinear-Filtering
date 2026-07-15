# Modern Computational Nonlinear Filtering

<div align="center">

**High-Performance Nonlinear State Estimation for Embedded Systems**

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Portable](https://img.shields.io/badge/Portable-any%20C%2B%2B20%20target-red.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Optional acceleration](https://img.shields.io/badge/Optional-NEON%20%7C%20SVE2%20%7C%20Vulkan%20%7C%20CUDA-orange.svg)](https://developer.arm.com/Architectures/Neon)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Portability](#portability)
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

This repository provides nonlinear filtering implementations in **portable C++20 + Eigen**, with **optional** SIMD and GPU acceleration (ARM NEON/SVE2 intrinsics, Vulkan compute shaders, NVIDIA CUDA). The filters themselves are architecture-neutral: `Common/include/FilterMath.h` resolves the available backend tiers at compile time and falls back to plain Eigen, so the library builds and produces equivalent results on **any** C++20 target — with no accelerator backend at all. All implementations use single-precision floating point (`float`), which suits SIMD vectorization where it is available.

### What's Included

- **5 Filtering Methods**: EKF, UKF, SRUKF, PKF, RBPKF
- **Smoothers**: fixed-lag RTS, full-interval (batch) RTS with optional iteration (an IEKS for the EKF), and ancestry-based particle smoothing
- **Comprehensive Benchmarks**: 4 test problems in the default suite with full metrics (10D coupled oscillators, Van der Pol, bearing-only tracking, reentry vehicle)
- **Optional Acceleration**: NEON/SVE2 dense linear algebra, Vulkan compute, and CUDA via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA) (OptMathKernels), consumed as a sibling source tree built with `add_subdirectory` (currently **v0.6.3**; no pinned release tag)

---

## Portability

The filters are portable C++; only the backends are architecture-specific, and every
backend is optional. The build detects the **target** architecture
(`CMAKE_SYSTEM_PROCESSOR`, cross-compile aware) and reports it at configure time:

```
-- Target architecture: ARM64
-- SIMD backends: NEON=ON SVE2=ON (SVE2 sources are skipped unless the CPU implements it)
-- Native CPU tuning: -mcpu=native
```

ARM-only backends are never forced onto a non-ARM target; on any other architecture
the same configure step reports `SIMD backends: none for <arch> - filter math uses the
Eigen fallback path` and the build proceeds normally.

### Acceleration Tiers (all optional)

| Backend | Where it applies | Default | Notes |
|---------|------------------|---------|-------|
| **Eigen** (baseline) | every C++20 target | always | The fallback. Correct and complete on its own — no other tier is required. |
| **NEON** | ARM (`aarch64`/`armv8`, `arm`) | ON for ARM targets | Cholesky, GEMM, mat-vec, SPD solve. `-DNLF_ENABLE_NEON=OFF` to disable. |
| **SVE2** | ARM CPUs that implement SVE2 (Cortex-A720+, e.g. Orange Pi 5/6) | ON for ARM targets | Cache-blocked GEMM with FCMA/I8MM. Sources are skipped on ARM CPUs without SVE2 (such as the Pi 5's Cortex-A76), so leaving it ON on a non-SVE2 ARM board is harmless. |
| **Vulkan** | any target with a Vulkan 1.2+ SDK/driver | ON, auto-skipped with no SDK | Compute backend in OptMathKernels. See the note under [Features](#hardware-optimization) about which operations actually reach the GPU. |
| **CUDA** | NVIDIA GPUs, SM 75–120 | auto-detected via `nvcc` | cuBLAS GEMM + GPU particle filter. CUDA 12.x = SM 75–90; 13.x adds Blackwell (SM 100/120). |

### Verified: the library does not need a SIMD backend

A build configured with **every** SIMD tier off and portable codegen —
`-DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF -DNLF_ENABLE_NATIVE_ARCH=OFF`, i.e. what a
brand-new architecture gets on day one — compiles, runs, and passes **30/30** CTest
cases, exactly as the NEON build does. Comparing benchmark RMSE between the two builds:

| Benchmark problem | Eigen-only vs. NEON |
|-------------------|---------------------|
| Coupled Oscillators 10D | bit-identical |
| Van der Pol 2D | bit-identical |
| Bearing-Only 4D | 0.0008% difference |
| Reentry 6D | 0.02% difference |

The two non-zero deltas are pure floating-point association (Eigen's summation order
vs. NEON's), not a behavioural difference. Accuracy is a property of the filters, not
of the backend; **only wall-clock timings change.**

> **Note on CUDA architectures**: CUDA 12.x covers Turing through Hopper (SM 75–90). CUDA 13.x adds Blackwell — including consumer RTX 50-series (**SM 120**, e.g. RTX 5070/5080/5090). To build for your exact GPU, configure with `-DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON`. See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for details.

---

## Implemented Filters

### 1. Extended Kalman Filter (EKF)

**Jacobian-based linearization**

- **Method**: First-order Taylor series approximation
- **Requirements**: Explicit Jacobian matrices
- **Smoothing**: RTS fixed-lag backward pass (`EKF/include/EKFFixedLag.h`) + full-interval iterated RTS / IEKS (`EKF/include/EKFSmoother.h`)
- **Best For**: Mildly nonlinear systems, fast prototyping
- **Location**: `EKF/`

### 2. Unscented Kalman Filter (UKF)

**Derivative-free sigma point method**

- **Method**: Deterministic sampling via Merwe scaled unscented transform
- **Parameters**: Dimension-adaptive (alpha=1.0, beta=2.0, kappa=3-n)
- **Smoothing**: RTS with cross-covariance tracking — fixed-lag (`UKF/include/UnscentedFixedLagSmoother.h`) + full-interval iterated (`UKF/include/UKFSmoother.h`)
- **Best For**: Highly nonlinear systems where Jacobians are unavailable or expensive
- **Location**: `UKF/`
- **Optimization**: GEMM, Cholesky and SPD solve go through the FilterMath dispatch — SVE2/NEON where present, Eigen otherwise

### 3. Square Root UKF (SRUKF)

**Numerically stable square root formulation**

- **Method**: Propagates Cholesky factor S where P = S*S^T
- **Algorithm**: QR decomposition + rank-1 Cholesky updates/downdates with safe fallbacks
- **Advantages**:
  - Guaranteed positive-definite covariance
  - Innovation gating prevents catastrophic updates
  - Safe Cholesky downdate with automatic fallback to full recomputation
- **Best For**: Mission-critical, long-duration, weak observability systems
- **Location**: `UKF/include/SRUKF.h`; smoothers in `UKF/include/SRUKFFixedLagSmoother.h` (fixed-lag) and `UKF/include/SRUKFSmoother.h` (full-interval, iterated)

### 4. Particle Filter (PKF)

**Bootstrap sequential importance resampling**

- **Method**: Monte Carlo approximation with particle ensemble
- **Resampling**: Systematic, stratified
- **Smoothing**: Ancestry-based fixed-lag trajectory reconstruction
- **Best For**: Non-Gaussian noise, multimodal distributions
- **Location**: `PKF/`
- **Optimization**: OpenMP parallel propagation; CUDA particle context auto-enabled at N >= 256 when built with CUDA. A Vulkan noise-addition path exists but rarely engages — see the [Vulkan note](#hardware-optimization) under Features.

### 5. Rao-Blackwellized Particle Filter (RBPKF)

**Hybrid particle-Kalman filter**

- **Method**: Marginalize linear substructure analytically via conditional Kalman filters
- **Structure**: Nonlinear particles + linear Kalman filters per particle
- **Advantages**: Reduced variance vs standard particle filter
- **Best For**: Systems with conditionally linear subspace (e.g., CTRV models)
- **Location**: `RBPKF/`

---

## Benchmark Results

Four problems tested with UKF, SRUKF, and fixed-lag smoothers. Benchmarks run through the FilterMath dispatch layer, which selects whatever backend the build provides (CUDA cuBLAS / SVE2 / NEON / Eigen) and computes the same result either way.

> **How to read these numbers.** **RMSE, NEES and divergence counts are
> backend- and host-independent** — they reproduce on any machine, with or without a
> SIMD/GPU backend (see [Portability](#portability): an Eigen-only build is
> bit-identical on two of the four problems and within 0.02% on the others).
> **Wall-clock timings are host-specific and do not transfer.** Every timing below is
> therefore attributed to the machine that produced it. No single machine is "the"
> reference; the suite is run on several.

**Hosts used below:**

| Host | Configuration | Used for |
|------|---------------|----------|
| **A** — x86_64 | Ubuntu 26.04 LTS, RTX 5070 Ti Laptop GPU (Blackwell, SM 120), CUDA 13.1, Eigen 3.4, Vulkan 1.4 | timings in the summary table (run 8 July 2026) |
| **B** — ARM aarch64 | Raspberry Pi 5 (Cortex-A76, 4 cores), NEON + Vulkan (V3D), no CUDA, no SVE2 | accuracy re-verification, portability comparison (run 14 July 2026) |

Accuracy figures below were produced on host A and re-verified unchanged on host B.
CTest is **30/30** on both (12 filter/benchmark tests registered by this repo + 18
OptimizedKernels GPU/SIMD tests). The exact dependency suite varies with the build:
host A additionally exercises the CUDA kernel tests and selects the discrete GPU for
the Vulkan suites; host B runs the same Vulkan suites on the VideoCore V3D and skips
CUDA.

Reproduce everything below with:

```bash
# from the build directory (see Build Instructions)
./Benchmarks/run_benchmarks                              # writes benchmark_results.csv + trajectory CSVs
source ../.nlfvenv/bin/activate                          # Python venv created by the build
python3 ../scripts/plot_benchmarks.py .                  # writes the PNGs shown here
```

`benchmark_results.csv` columns (one header row):

```
Filter,Problem,RMSE_Overall,RMSE_Smoothed_Overall,Median_NEES,Pct_In_Bounds,NEES_Valid,NEES_Total,Avg_Step_Time_ms,Total_Time_ms,Convergence_Time,Num_Divergences
```

`Convergence_Time` is left **empty** when the filter never met the convergence
threshold — missing data reads as missing, in pandas and in a spreadsheet.

### Summary Across All Problems

**Estimation accuracy** — filtered vs. smoothed RMSE and filter consistency (median NEES):

![RMSE / NEES comparison](docs/images/benchmark_rmse_comparison.png)

**Computational cost** — per-step and total wall-clock time (**host A**; re-run the suite to get your own):

![Timing comparison](docs/images/benchmark_timing_comparison.png)

**Convergence & stability** — time-to-converge and divergence counts:

![Convergence comparison](docs/images/benchmark_convergence_comparison.png)

> **Caveat on the convergence panel**: `compute_convergence_time()`
> (`Benchmarks/include/BenchmarkRunner.h`) uses a fixed **0.5-unit absolute** error
> threshold, which is not meaningful for problems whose error scale is far larger —
> Bearing-Only (~64) and Reentry (~369) legitimately never reach it. Those rows now
> report an **empty** `Convergence_Time` cell and print "did not converge" rather
> than silently emitting the final timestamp as if it were a measurement. Read that
> panel for the Coupled-Oscillator and Van der Pol rows only.

Consolidated metrics (`benchmark_results.csv`). RMSE / NEES / Div are
host-independent; **Avg step** and **Total** are from **host A**:

| Problem | Filter | RMSE | Smoothed RMSE | NEES median | In 95% | Avg step (host A)† | Total (host A)† | Div |
|---------|--------|-----:|--------------:|------------:|-------:|---------:|------:|----:|
| Coupled Osc 10D | UKF | 1.457 | — | 9.89 | 94.5% | 0.0076 ms | 38.2 ms | 0 |
| Coupled Osc 10D | SRUKF | 1.457 | — | 9.89 | 94.5% | 0.0082 ms | 41.5 ms | 0 |
| Coupled Osc 10D | UKF+Smoother | 1.457 | **1.148** | 9.89 | 94.5% | 0.0275 ms | 138.0 ms | 0 |
| Coupled Osc 10D | SRUKF+Smoother | 1.457 | **1.148** | 9.89 | 94.5% | 0.0400 ms | 201.1 ms | 0 |
| Van der Pol 2D | UKF | 0.469 | — | 1.13 | 95.8% | 0.00039 ms | 0.81 ms | 0 |
| Van der Pol 2D | SRUKF | 0.467 | — | 1.13 | 95.9% | 0.00027 ms | 0.57 ms | 0 |
| Van der Pol 2D | SRUKF+Smoother | 0.467 | **0.430** | 1.13 | 95.9% | 0.0050 ms | 10.2 ms | 0 |
| Bearing-Only 4D | UKF | 63.49 | — | 3.71 | 99.6% | 0.00052 ms | 0.16 ms | 0 |
| Bearing-Only 4D | SRUKF | 63.84 | — | 3.72 | 99.6% | 0.00046 ms | 0.15 ms | 0 |
| Bearing-Only 4D | SRUKF+Smoother | 63.84 | **51.68** | 3.72 | 99.6% | 0.0108 ms | 3.25 ms | 0 |
| Reentry 6D | UKF | 366.9 | — | 4.99 | 95.9% | 0.0024 ms | 0.72 ms | 0 |
| Reentry 6D | SRUKF | 367.1 | — | 4.99 | 95.9% | 0.0024 ms | 0.73 ms | 0 |
| Reentry 6D | SRUKF+Smoother | 367.1 | **236.3** | 4.99 | 95.9% | 0.0176 ms | 5.30 ms | 0 |

> † The accuracy columns were re-measured after the timestep-alignment fix
> (commit `3dbbf4c`, which moved every RMSE by up to ~0.6%). The **host A timing
> columns predate that fix and have not been re-measured there** — treat them as
> indicative of relative cost, not as a matched run. Timings are the one thing you
> should measure on your own hardware anyway.

### Coupled Oscillators (10D State, 5D Observation)

![Coupled Oscillators trajectories](docs/images/coupled_osc_srukf_smooth_plot.png)

Ten coupled states tracked over 50 s; True, Filtered, and Smoothed curves are nearly indistinguishable across all dimensions. Smoothing improves RMSE by **21%** (1.457 → 1.148). NEES 94.5% in chi-squared bounds indicates excellent consistency. This is the case that previously failed with NaN before the v3.1.0 audit (resolved via dimension-adaptive sigma-point parameters and a safe Cholesky downdate) — now rock-solid with zero divergences.

### Van der Pol Oscillator (2D State, 1D Observation)

![Van der Pol trajectories](docs/images/vanderpol_srukf_smooth_plot.png)

Classic relaxation oscillator: slow drift in `x0` with sharp limit-cycle transitions in `x1` (the spikes near t≈1.5 s and t≈11 s). The filter captures the fast transitions without lag; the smoother visibly tightens the estimate in noisy stretches. Smoothing improves RMSE by **8%** (0.467 → 0.430). SRUKF is ~1.4× faster than UKF on this small problem (consistent on both hosts).

### Bearing-Only Tracking (4D State, 1D Observation)

![Bearing-only trajectories](docs/images/bearing_srukf_smooth_plot.png)

Weak-observability problem: angle-only measurements barely constrain range, so the range estimate drifts (large steady-state RMSE ~64) even though the filter stays statistically consistent — NEES holds at **99.6% in-bounds** throughout. Smoothing improves RMSE by **19%** (63.84 → 51.68) but cannot fully recover the lost range. (Earlier runs reported a spurious ~175 "divergence" count here; that was a benchmark-metric bug — a fixed 10 m error threshold measured against this problem's ~64 m error scale — corrected in v3.3.0 with a problem-scaled 500 m threshold, so the divergence count is now **0**. The filter never actually diverged; NEES was in-bounds the whole time.)

### Reentry Vehicle (6D State, 3D Observation)

![Reentry vehicle trajectories](docs/images/reentry_srukf_smooth_plot.png)

Realistic spacecraft reentry with altitude-dependent gravity (μ/r²), exponential-atmosphere drag, and a ground-based radar. Position states (`x0`–`x2`) track the truth almost exactly; the noisy velocity states (`x3`, `x5`) show the smoother clearly reducing noise relative to the filter. NEES median 4.99 (expected ~6) with 95.9% in bounds. Smoothing improves RMSE by **36%** (367.1 → 236.3) — the largest gain of the four problems.

> **Note on the trajectory plots**: the Smoothed (red) curve is correctly aligned to the time it estimates and ends one fixed-lag window before the filtered curve (the smoother has no refined estimate for the most recent `lag` steps). Earlier revisions showed a spurious run of leading zeros here; that was a CSV-writer alignment bug (the RMSE numbers were always correct) and is fixed as of the latest commit.

### Filter & Smoother Test Results

Each row is the filtered-vs-smoothed pair reported by that test's own smoother run
(`ekf_test`, `ukf_test`, `srukf_test`, `pkf_example`), re-measured after the
timestep-alignment fix:

| Test | Filter RMSE | Smoother RMSE | Improvement |
|------|-------------|---------------|-------------|
| EKF (Nonlinear Oscillator, 2D) | 0.060 | 0.052 | 13% |
| UKF (Drag Ball, 4D) | 0.228 | 0.119 | 48% |
| SRUKF (Drag Ball, 4D) | 0.311 | 0.167 | 46% |
| PKF (Lorenz-63, 3D) | 0.813 | 0.604 | 26% |

> The PKF row is a Monte-Carlo result on a chaotic system, so it moves slightly
> from host to host (floating-point association changes which particles survive
> resampling); the other three are deterministic and reproduce exactly.

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

This is a **historical v2.8.0 → v3.0.0 regression snapshot**: it shows that the
v3.0 dispatch-layer refactor preserved accuracy to floating-point precision for
the problem configurations *as they were then*. The Bearing-Only scenario has
since changed, so its figure here (43.151) does not match the current **63.84**
in the main results table above — that main table is the authoritative current
result; this row is retained only as a historical refactor-regression check and
is not directly comparable to the current Bearing-Only run. NEES consistency
unchanged (all pass chi-squared bounds).

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

**Problem**: The SRUKF predict step's weighted-mean loop returned the INPUT state instead of the PROPAGATED state, so `predict()` appeared to leave the state unchanged.

**Root Cause**: Eigen's expression templates caused aliasing between `X_pred` (propagated sigma points) and `sigmas.X` (input sigma points) during weighted mean computation. The operation `x_pred_mean += Wm(i) * X_pred.col(i)` was reading from `sigmas.X` memory instead of `X_pred`, even though they were separate stack-allocated matrices.

**Symptoms**:
- State appears unchanged after `predict()`
- Debug shows `X_pred` values are correct, but the weighted mean equals the input
- Double-precision accumulation gives the correct result while the float loop gives the wrong one

**Solution**: Force evaluation with `.eval()` to materialize the expression and break aliasing, combined with `.noalias()` for safe accumulation. This preserves NEON/SVE2 auto-vectorization (unlike the earlier raw C array workaround). The fix lives in `UKF/include/SRUKF.h` (see the `X_pred_eval` / `Wm_eval` block):
```cpp
// Force evaluation to break Eigen aliasing while keeping SIMD vectorization
typename SigmaPts::SigmaMat X_pred_eval = X_pred.eval();
typename SigmaPts::Weights Wm_eval = sigmas.Wm.eval();

State x_pred_mean = State::Zero();
for (int i = 0; i < SigmaPts::NSIG; ++i) {
    x_pred_mean.noalias() += Wm_eval(i) * X_pred_eval.col(i);
}
```

**Result**: `predict()` now propagates the state correctly; the coupled-oscillator and reentry benchmarks pass with zero divergences.

### Issue #5: Global `-ffast-math` Breaks Filter Stability

**Problem**: CMake root had `-ffast-math` and `EIGEN_FAST_MATH=1` applied globally to all Release builds. This silently caused:
- NaN comparison guards (`isfinite()`, `allFinite()`) to be optimized away
- Cholesky decomposition precision loss from altered floating-point associativity
- Denormal flushing that corrupted small covariance values

**Solution**: Removed global `-ffast-math` and `EIGEN_FAST_MATH=1` from root `CMakeLists.txt`. The NEON/SVE2 intrinsics in OptMathKernels already provide hardware-accelerated fast paths where needed. Numerically sensitive targets should explicitly set `-fno-fast-math` and `EIGEN_FAST_MATH=0`.

### Issue #6: Unsafe Cholesky Downdate in SRUKF Prediction

**Problem**: The SRUKF predict step used the legacy `cholupdate_downdate()` which silently corrupted the square root covariance when the downdate magnitude exceeded the current factor (sets `r_sq = 1e-6 * S(k,k)²` instead of failing).

**Solution**: Replaced with `cholupdate_downdate_safe()` which returns false on failure, plus a full-covariance fallback that recomputes P from all sigma points and takes a fresh Cholesky.

### Issue #7: RBPKF Weight Corruption and Resampling Bias

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

- **FilterMath Dispatch Layer** (`Common/include/FilterMath.h`): Unified API that selects the best tier the build provides:
  - **GEMM**: CUDA cuBLAS → SVE2 cache-blocked → NEON blocked → Eigen
  - **Cholesky / Inverse / Solve**: NEON accelerated → Eigen LDLT fallback
  - **Kalman Gain**: SPD solve (avoids explicit matrix inverse for O(n²) vs O(n³))
  - **Any platform with no backend**: falls through to Eigen — always available, always correct
- **Particle Filter GPU** (`PKF/include/particle_filter_gpu.hpp`): CUDA particle filter:
  - **GPUParticleContext\<NX\>**: Manages particles/weights on GPU
  - GPU log-sum-exp weight normalization
  - GPU systematic/stratified resampling
  - Auto-enable for N >= 256 particles (CUDA builds only; a no-CUDA build always uses the CPU path)
- **ARM NEON Dense Linear Algebra**: Cholesky, GEMM, mat-vec multiply, SPD solve via [OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
- **ARM SVE2**: Cache-blocked GEMM with FCMA and I8MM on Cortex-A720+ (e.g. Orange Pi 5/6). Not present on the Cortex-A76 (Raspberry Pi 5), where the SVE2 sources are simply skipped.
- **Vulkan Compute**: Device selection prefers a **discrete GPU with a compute queue** over integrated/CPU fallbacks, and logs the choice at init.
- **Graceful Fallback**: CUDA → SVE2 → NEON → Eigen with jitter + retry for numerical robustness
- **Single Precision**: Consistent use of `float`, which vectorizes well where SIMD is available

> **Vulkan reality check**: the particle filter contains a Vulkan noise-addition path
> (gated at `N > 100` in `PKF/include/particle_filter.hpp`), but OptMathKernels routes
> elementwise operations below **2^20 elements** to the CPU
> (`OPTMATH_VK_ELTWISE_MIN`, `src/vulkan/vulkan_backend.cpp`). The flattened buffer is
> `N*NX` floats, so a 3-state model would need ~350,000 particles before anything
> reaches the GPU. At realistic particle counts this path computes on the CPU. The
> threshold is deliberate: on the tested integrated GPUs, dispatch overhead exceeds the
> work for memory-bound elementwise kernels. Treat Vulkan here as available
> infrastructure, not as an active accelerator for typical particle counts.

> **FilterMathGPU status**: `Common/include/FilterMathGPU.h` (`GPUBufferPool`,
> `GPUSigmaContext<NX>`) is **experimental and not yet wired in** — no filter currently
> includes it, and the UKF/SRUKF paths go through `FilterMath.h`. It ships as a public
> header for downstream experimentation only.

> **CUDA Status**: Active for SM 75–90 (CUDA 12.x) and SM 75–120 incl. Blackwell RTX 50-series (CUDA 13.x). See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

### Software Quality

- **C++20**: Modern features (concepts, ranges, template constraints)
- **Type Safety**: Template metaprogramming for compile-time dimension checking
- **Joseph Form**: Numerically stable covariance update throughout
- **OpenMP**: Parallel particle propagation and weight updates

---

## Dependencies

### Required

- **C++20 Compiler**: GCC 10+, Clang 11+ (MSVC builds use `/O2`; no native-tuning flag is applied)
- **Eigen3**: Linear algebra (3.4+, fetched automatically if not found). This is the only hard numerical dependency — everything below Eigen in the dispatch chain is optional.
- **CMake**: Build system (**3.18+**). The root project declares 3.14, but the required OptMathKernels dependency is built via `add_subdirectory` and itself requires 3.18, so 3.18 is the effective floor.
- **Python 3**: (3.10+) For visualization scripts; a virtual environment is created automatically during the build
- **[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)** (OptMathKernels, currently **v0.6.3**): hosts the optional NEON/SVE2/Vulkan/CUDA kernels. **Required as a sibling directory** next to this repo and built via `add_subdirectory` directly from its **latest `main`** working tree (no pinned release tag). If it is missing, `./bootstrap.sh` offers to clone it over HTTPS and build for you, or configure with `-DAUTO_CLONE_DEPS=ON`. See [Build Instructions](#build-instructions).

### Optional

- **ARM NEON/SVE2**: Enabled by default on ARM targets only (`NLF_ENABLE_NEON` / `NLF_ENABLE_SVE2`); never forced onto a non-ARM target
- **Vulkan SDK + glslang-tools**: For the Vulkan compute backend (**Vulkan 1.2+** — the backend requests `VK_API_VERSION_1_2`); `glslang-tools` provides `glslangValidator` to compile GLSL shaders to SPIR-V. Auto-skipped when no SDK is found.
- **NVIDIA CUDA Toolkit**: For GPU-accelerated GEMM and particle filter (12.x supports SM 75–90; 13.x adds Blackwell, incl. RTX 50-series SM 120). Ensure `nvcc` is on `PATH` (e.g. `export PATH=/usr/local/cuda/bin:$PATH`) so CMake detects it.
- **OpenMP**: For parallel particle filter

### Installation (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake libeigen3-dev
sudo apt install python3 python3-pip python3-venv

# Vulkan shader compiler (required for Vulkan GPU tests)
sudo apt install glslang-tools

# Optional: Vulkan runtime (if not already installed)
sudo apt install vulkan-tools libvulkan-dev

# The OptimizedKernels dependency is REQUIRED as a sibling directory next to this
# repo. bootstrap.sh (see Build Instructions) will offer to clone it for you; to
# provision it manually, clone it beside this repo so both live under the same parent:
cd ~
git clone https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA.git
```

#### Optional: NVIDIA CUDA (GPU acceleration)

CUDA is auto-detected at configure time, but **only if `nvcc` is on your `PATH`** — installing the toolkit alone is not enough. Install the toolkit (12.x for Turing–Hopper, **13.x for Blackwell / RTX 50-series**), then expose it:

```bash
# Install the toolkit (example: CUDA 13.1 on Ubuntu, or use the .run/.deb from
# https://developer.nvidia.com/cuda-downloads)
sudo apt install cuda-toolkit-13-1

# Put nvcc + runtime libs on PATH. Append to ~/.bashrc to persist across shells:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CMake will find it
nvcc --version          # should print the toolkit version
nvidia-smi              # should list your GPU + driver
```

> CUDA 13.x ships `nvcc` that accepts host GCC up to 15. When native tuning is active (`NLF_ENABLE_NATIVE_ARCH=ON`, the default), the build forwards the same probed arch flag to the CUDA host compiler via `-Xcompiler`, so the C++ and CUDA translation units stay ABI-consistent (on x86, mismatched Eigen alignment between them otherwise corrupts the heap).

---

## Build Instructions

```bash
# Clone repository
git clone https://github.com/n4hy/Modern-Computational-Nonlinear-Filtering.git
cd Modern-Computational-Nonlinear-Filtering

# EASIEST: one-shot bootstrap. Checks for the required OptimizedKernels sibling
# directory and, if it is missing, OFFERS to git-clone it (latest main) over
# HTTPS, then configures + builds everything. Extra args pass through to CMake.
./bootstrap.sh
# ./bootstrap.sh -DNLF_BUILD_PYTHON_VENV=OFF     # e.g. skip the plotting venv

# --- or configure manually ---------------------------------------------------
# The build REQUIRES the OptimizedKernels source tree as a sibling directory
# (../OptimizedKernelsForRaspberryPi5_NvidiaCUDA). It is built directly from its
# local working tree (latest main). If it is absent, configuration fails with
# instructions unless you pass -DAUTO_CLONE_DEPS=ON to clone it over HTTPS.
mkdir -p build && cd build

# Configure and build (CUDA auto-detected). CMake also creates a Python venv at
# .nlfvenv/ for the plotting scripts.
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# If the sibling dependency is not present, let CMake clone it (latest main):
cmake .. -DCMAKE_BUILD_TYPE=Release -DAUTO_CLONE_DEPS=ON
make -j$(nproc)

# Optional: build CUDA code for your exact GPU only (faster nvcc, smaller binary).
# On CUDA 13.x the default arch list already includes Blackwell SM 100/120, so this
# is no longer required for RTX 50-series — it's just an optimization.
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
make -j$(nproc)

# Or explicitly disable CUDA if needed
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=""
make -j$(nproc)

# Portable / redistributable binary, or a brand-new architecture: turn off every
# SIMD tier and native-CPU tuning. The filters fall back to Eigen and still pass
# 30/30 ctest (see Portability).
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF -DNLF_ENABLE_NATIVE_ARCH=OFF
make -j$(nproc)

# Cross-compiling: pass your toolchain file. The target arch comes from
# CMAKE_SYSTEM_PROCESSOR, and NLF_ENABLE_NATIVE_ARCH defaults to OFF automatically
# (a "native" flag would otherwise describe the BUILD host, not the target).
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake
make -j$(nproc)

# Offline / CI build: skip the plotting Python venv (no PyPI access needed). Point
# -DOPTMATH_DIR at the sibling clone if it is not at the default ../ location.
cmake .. -DCMAKE_BUILD_TYPE=Release -DNLF_BUILD_PYTHON_VENV=OFF \
         -DOPTMATH_DIR=$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
make -j$(nproc)

# (Optional) system install — installs the reusable filter headers under
# <prefix>/include/nlf/ (plus run_benchmarks) to CMAKE_INSTALL_PREFIX
# (default /usr/local; override with -DCMAKE_INSTALL_PREFIX=/path when configuring).
sudo make install

# Run all tests via CTest
ctest --output-on-failure

# Or run individual tests
./EKF/ekf_test
./EKF/ekf_smoother_test
./UKF/ukf_test
./UKF/ukf_smoother_test
./UKF/srukf_test
./UKF/srukf_angular_wrap_test
./UKF/srukf_smoother_test
./PKF/pkf_test
./PKF/pkf_example
./RBPKF/test_rbpf_basic
./RBPKF/example_rbpf_ctrv

# Run benchmarks
./Benchmarks/run_benchmarks

# Use the Python venv for visualization scripts
source ../.nlfvenv/bin/activate
python3 ../scripts/plot_benchmarks.py .

# Run OptimizedKernels tests (Vulkan, radar, platform). The dependency is now
# built under build/optmath-build/ (add_subdirectory), not _deps/.
./optmath-build/tests/test_vulkan_vector
./optmath-build/tests/test_vulkan_matrix
./optmath-build/tests/test_radar_caf
./optmath-build/tests/test_radar_cfar
```

### Build Options

| Option | Description |
|--------|-------------|
| `-DNLF_ENABLE_NEON=OFF` | Do not build the ARM NEON kernels (default: ON for ARM targets, OFF everywhere else). Filter math falls back to Eigen |
| `-DNLF_ENABLE_SVE2=OFF` | Do not build the ARM SVE2 kernels (default: ON for ARM targets). SVE2 sources are skipped anyway on ARM CPUs that lack SVE2 |
| `-DNLF_ENABLE_VULKAN=OFF` | Do not build the Vulkan compute backend (default: ON; auto-skipped when no SDK is found) |
| `-DNLF_ENABLE_NATIVE_ARCH=OFF` | Do not tune for the building machine's CPU. Default ON, and automatically OFF when cross-compiling. Turn OFF for portable/redistributable binaries. When ON, the build *probes* for `-mcpu=native` (ARM) or `-march=native` (x86) and degrades to portable codegen if neither compiles cleanly; MSVC has no equivalent and uses `/O2` |
| `-DCMAKE_TOOLCHAIN_FILE=<path>` | Cross-compile. Target arch is read from `CMAKE_SYSTEM_PROCESSOR`; native tuning auto-disables |
| `-DCMAKE_CUDA_COMPILER=""` | Disable CUDA (e.g., if toolkit is not installed or causes issues) |
| `-DCMAKE_CUDA_ARCHITECTURES=native` | Build CUDA code for the detected GPU only. Optional on CUDA 13.x (default list already includes Blackwell SM 100/120); useful to shrink build time |
| `-DOPTMATH_CUDA_NATIVE=ON` | Make the OptimizedKernels dependency honor `native` instead of its default multi-arch list |
| `-DOPTMATH_DIR=<path>` | Path to the required OptimizedKernels sibling source tree (default `../OptimizedKernelsForRaspberryPi5_NvidiaCUDA`). Built directly from its local working tree (latest `main`) |
| `-DAUTO_CLONE_DEPS=ON` | If `OPTMATH_DIR` is missing, clone it (latest `main`) over HTTPS instead of failing (default `OFF`). `./bootstrap.sh` does this interactively |
| `-DNLF_BUILD_PYTHON_VENV=OFF` | Skip creating the plotting Python venv (the C++ build then needs no Python/PyPI — for offline/CI) |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build: `-O3` (or `/O2` on MSVC) plus native tuning when `NLF_ENABLE_NATIVE_ARCH` is ON, forwarded to nvcc's host compiler via `-Xcompiler` for ABI consistency. Release flags are applied via `$<CONFIG:Release>` generator expressions, so multi-config generators (Ninja Multi-Config, Visual Studio, Xcode) get them too |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug build with symbols |

### Build Outputs

| Target | Description |
|--------|-------------|
| `EKF/ekf_test` | EKF with fixed-lag smoother on nonlinear oscillator |
| `EKF/ekf_smoother_test` | EKF full-interval RTS / IEKS smoother regression |
| `UKF/ukf_test` | UKF with fixed-lag smoother on drag ball model |
| `UKF/ukf_smoother_test` | UKF full-interval RTS smoother regression |
| `UKF/srukf_test` | SRUKF with fixed-lag smoother on drag ball model |
| `UKF/srukf_angular_wrap_test` | SRUKF angular-observation (wrap) regression |
| `UKF/srukf_smoother_test` | SRUKF full-interval RTS smoother regression |
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

    // Process measurements.
    //
    // Timestep alignment matters: initialize() leaves the estimate AT time[0], and
    // predict(t) evaluates the process model at t = where the state currently IS.
    // So iteration 0 must not propagate, and advancing time[k-1] -> time[k] passes
    // time[k-1] to predict() and time[k] to update().
    Eigen::Vector4f u = Eigen::Vector4f::Zero();

    filter.update(time[0], measurements[0]);   // fuse y0 where the estimate already is

    for (int k = 1; k < num_steps; ++k) {
        filter.predict(time[k - 1], u);        // propagate FROM time[k-1]
        filter.update(time[k], measurements[k]);  // fuse a measurement taken AT time[k]

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
    smoother.initialize(x0, P0);            // estimate now sits AT time[0]

    Eigen::Vector4f u = Eigen::Vector4f::Zero();

    // If you have a measurement at time[0], fuse it where the estimate already is.
    // Calling step() here instead would predict first, landing the estimate at
    // time[1] and folding an observation of x(time[0]) into it. Skip this if your
    // first measurement is at time[1].
    smoother.observe_initial(time[0], measurements[0]);

    for (int k = 1; k < num_steps; ++k) {
        // Two DISTINCT times: propagate from where the estimate is (time[k-1]),
        // then fuse a measurement taken at time[k].
        smoother.step(time[k - 1], time[k], measurements[k], u);

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
│       ├── FilterMath.h        # CUDA/SVE2/NEON/Eigen dispatch layer
│       ├── FilterMathGPU.h     # experimental, not included by any filter
│       ├── StateSpaceModel.h   # Base model for UKF/SRUKF
│       ├── SystemModel.h       # Base model for EKF
│       └── FileUtils.h         # File I/O utilities
│
├── EKF/                        # Extended Kalman Filter
│   ├── include/
│   │   ├── EKF.h
│   │   ├── EKFFixedLag.h       # fixed-lag RTS
│   │   ├── EKFSmoother.h       # full-interval RTS / iterated (IEKS)
│   │   ├── BallTossModel.h
│   │   └── NonlinearOscillator.h
│   ├── src/
│   │   ├── EKF.cpp
│   │   └── EKFFixedLag.cpp
│   ├── tests/
│   │   └── test_ekf_smoother.cpp
│   └── main.cpp
│
├── UKF/                        # Unscented Kalman Filters
│   ├── include/
│   │   ├── UKF.h               # Standard UKF
│   │   ├── SRUKF.h             # Square Root UKF
│   │   ├── SigmaPoints.h       # Sigma point generation
│   │   ├── UnscentedFixedLagSmoother.h
│   │   ├── SRUKFFixedLagSmoother.h
│   │   ├── UKFSmoother.h       # full-interval RTS / iterated
│   │   ├── SRUKFSmoother.h     # full-interval square-root RTS / iterated
│   │   └── DragBallModel.h     # Example model
│   ├── tests/
│   │   ├── test_ukf_smoother.cpp
│   │   ├── test_srukf_smoother.cpp
│   │   └── test_srukf_angular_wrap.cpp
│   ├── main.cpp                # UKF + smoother test
│   └── main_srukf.cpp          # SRUKF + smoother test
│
├── PKF/                        # Particle Filter
│   ├── include/
│   │   ├── particle_filter.hpp
│   │   ├── particle_filter_gpu.hpp  # CUDA GPU particle context
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
│   │   ├── BenchmarkProblems.h # 5 models (4 in the default suite; Lorenz96
│   │   │                       #   is provided but not run)
│   │   └── BenchmarkRunner.h   # Metrics framework
│   └── src/
│       └── run_benchmarks.cpp
│
├── scripts/                    # Visualization
│   ├── simple_plot_benchmarks.py
│   ├── plot_benchmarks.py
│   ├── plot_optimized.py
│   ├── plot_results.py
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
2. **GPU Optimization**: CUDA sigma point propagation, cuSOLVER integration (CUDA 13.x, Blackwell SM 120)
3. **Adaptive Methods**: Automatic parameter tuning (adaptive Q/R)
4. **Multi-Sensor Fusion**: Asynchronous measurement handling
5. **Extended Benchmarks**: Monte Carlo consistency analysis, filter divergence studies
6. **UD Factorization**: Alternative to Cholesky for extreme ill-conditioning

Please:
- Follow C++20 style guidelines
- Add tests for new features
- Document numerical considerations
- See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for current restrictions

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

**Version**: 3.4.0
**Last Updated**: 14 July 2026
**OptMathKernels**: sibling source tree built via `add_subdirectory` (no pinned tag); currently v0.6.3
**Portability**: portable C++20 + Eigen on any target; optional NEON/SVE2 (ARM), Vulkan (1.2+), and CUDA (SM 75–120 via CUDA 12.x/13.x) backends
