# Final Comprehensive Audit Summary
## Modern Computational Nonlinear Filtering

**Date**: July 8, 2026 (v3.2.3)
**Status**: Production-ready with CUDA/SVE2/NEON/Vulkan acceleration and cross-platform Eigen fallback. Verified on Ubuntu 26.04 LTS x86_64, RTX 5070 Ti Laptop GPU (Blackwell, SM 120), CUDA 13.1.115, Vulkan 1.4.341, Eigen 3.4.0. CUDA active for SM 75–120 including Blackwell RTX 50-series. Compute kernels from OptMathKernels pinned at release tag **v0.5.17**.

---

## Audit History

### Phase 1 (Feb 2026): Initial SRUKF Implementation
- Fixed 5 critical numerical bugs (sigma point weights, QR loop count, 1D QR, singular Q, Cholesky division-by-zero)
- Dimension-adaptive parameters (alpha, kappa)
- Replaced QR-based S_yy with direct P_yy computation for robustness

### Phase 2 (Mar 2026): Numerical Stability & Error Handling
- Fixed innovation gating, Eigen expression template aliasing, Monte Carlo convergence
- Disabled `-ffast-math` for numerically sensitive targets
- Measurement outage recovery detection

### Phase 3 (Mar 31, 2026): FilterMath Dispatch & Full Optimization

**Created `Common/include/FilterMath.h`** — unified dispatch layer:
- **GEMM**: SVE2 cache-blocked (A720 12MB L3) → NEON blocked → Eigen
- **Cholesky / Inverse / Solve**: NEON accelerated → Eigen LDLT fallback
- **Kalman Gain**: SPD solve (avoids explicit inverse, O(n^2) vs O(n^3))
- **Non-ARM**: All paths fall through to pure Eigen

**All filter code updated to use FilterMath dispatch**:
- EKF, EKFFixedLag, UKF, SRUKF, SigmaPoints
- UnscentedFixedLagSmoother, SRUKFFixedLagSmoother
- RBPKF (kalman_filter, rbpf_core), Benchmarks

**Bug fixes in this phase**:
1. Removed global `-ffast-math` / `EIGEN_FAST_MATH=1` (broke NaN guards, Cholesky precision)
2. SRUKF predict: replaced unsafe `cholupdate_downdate` with safe version + full-covariance fallback
3. SRUKF update: added `allFinite()` check on S_yy
4. RBPKF `normalize_weights`: added `isfinite()` guard against NaN/Inf weight corruption
5. RBPKF `get_effective_sample_size`: guarded against division by zero
6. RBPKF resampling: Kahan compensated summation for cumulative weights (fixes systematic bias)
7. PKF: added `#if PKF_HAS_VULKAN` guards for cross-platform compilation

### Phase 4 (Apr 2, 2026): CUDA GPU Acceleration

**Created CUDA acceleration layer** (commit 397b2d9):
- **FilterMath.h**: Extended with CUDA backend dispatch (CUDA > SVE2 > NEON > Eigen)
- **FilterMathGPU.h**: GPU buffer management
  - `GPUBufferPool`: Reusable device allocations to minimize PCIe overhead
  - `GPUSigmaContext<NX>`: GPU-accelerated sigma point operations for UKF/SRUKF
- **particle_filter_gpu.hpp**: CUDA particle filter
  - `GPUParticleContext<NX>`: Manages particles/weights on GPU
  - GPU log-sum-exp weight normalization
  - GPU systematic/stratified resampling
  - Auto-enable for N >= 256 particles
- cuBLAS GEMM for matrices >= 32x32
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

**CUDA architecture support** (CMakeLists.txt):
- SM 75: Turing (RTX 2080/2070/2060)
- SM 80/86: Ampere (RTX 3090/3080/3070)
- SM 89: Ada Lovelace (RTX 4090/4080/4070)
- SM 90: Hopper (H100)
- SM 100: Blackwell (RTX 5090/5080) — **requires CUDA 13+**

### Phase 5 (Apr 3–13, 2026): CUDA Compatibility & Activation

**Identified CUDA 12.0.140 (Ubuntu 24.04) incompatibilities**:
1. `nvcc fatal: Unsupported gpu architecture 'compute_100'` — Blackwell not supported in CUDA 12.x

**Resolution**: CMakeLists.txt updated to exclude SM 100 from architecture list. CUDA 12.x active for SM 75–90 (Turing through Hopper). Blackwell (SM 100) pending CUDA 13+.

### Phase 6 (Apr 12, 2026): Correctness Audit & ARM Optimization

Full codebase audit targeting both architecture-independent correctness and
Orange Pi (aarch64 A720/SVE2/NEON/Mali-G720) optimization.

**Correctness fixes**:
1. `FilterMath.h` `solve_spd_mat`: restructured to try LDLT with `isPositive()` check
   first (was missing), then column-wise NEON fallback; eliminates uninitialized matrix
   risk from partial column failure path
2. `BenchmarkProblems.h` `ReentryVehicle::h()`: added range guard and `std::clamp` on
   `asin` argument to prevent NaN from division-by-zero or floating-point overshoot
3. `ReentryVehicle::dynamics()`: eliminated redundant `pos.norm()` call; simplified drag
   formula from `speed*speed/BC*(vel/speed)` to `speed/BC*vel` (removes division)
4. `noise_models.hpp`: added `llt.info()` validation in all 4 LLT decomposition sites
   (student_t_logpdf, student_t_sample, gaussian_logpdf, gaussian_sample)
5. `SRUKFFixedLagSmoother.h`: added `llt.info()` check in ultimate Cholesky fallback;
   falls back to filtered S if decomposition fails instead of using garbage
6. `run_benchmarks.cpp`: added LLT error check in Cholesky fallback for noise generation;
   uses diagonal fallback if both filtermath and Eigen Cholesky fail
7. `UnscentedFixedLagSmoother.h`: added covariance symmetrization after smoothing update
   (was missing, causing accumulated round-off asymmetry); added NaN guard matching
   SRUKFFixedLagSmoother
8. `particle_fixed_lag.hpp`: cleaned up redundant conditionals in `get_smoothed_mean()`
   and `get_smoothed_covariance()` (both branches were identical); added empty-history guard

**Performance optimizations**:
1. `SRUKF.h` predict mean computation: replaced raw C array workaround (27 lines) with
   Eigen `.eval()` + `.noalias()` (5 lines) — preserves aliasing safety while enabling
   NEON/SVE2 auto-vectorization. Previously, copying to/from C arrays defeated SIMD.
2. `resampling.hpp` (PKF): moved `std::uniform_real_distribution` construction outside
   inner loop in both float and double stratified resampling overloads
3. `BenchmarkRunner.h`: replaced O(n log n) full sort with O(n) `std::nth_element` for
   median NEES computation

### Phase 7 (May 25, 2026): CUDA 13 / Blackwell Activation & Kernel Release Pinning

- Activated CUDA 13.x: full Blackwell support, **verified on RTX 5070 Ti (SM 120)**
  with CUDA 13.1. CMakeLists forwards `-march=native` to nvcc's host compiler via
  `-Xcompiler` to keep C++/CUDA Eigen alignment ABI-consistent.
- **Adopted the OptMathKernels tag/release format**: pinned the dependency via
  `OPTMATH_RELEASE_TAG` (now **v0.5.15**) instead of tracking `main`. Audited the
  upstream v0.5.13 → v0.5.15 diff — only behavioral change is Vulkan discrete-GPU
  preference (now selects the RTX 5070 Ti); no public-API change. See
  DEVELOPMENT_NOTES.md → "OptMathKernels Dependency — Release Audit & Pinning Policy".
- Created the repository's first annotated release tag (`v3.2.0`).
- Rebuilt, **24/24 CTest pass**, benchmarks rerun (RMSE/NEES unchanged), plots regenerated.

### Phase 8 (July 8, 2026): Kernel bump v0.5.17, audit fixes & dispatch fast-path

- **Adopted OptMathKernels v0.5.15 → v0.5.17** (audited per the pinning policy).
  Upstream diff is docs + a NEON unit-test-tolerance fix only — no public-API,
  compute-backend, or MPI change. Tagged parent `v3.2.1`. See DEVELOPMENT_NOTES.md
  → "Audit: v0.5.15 → v0.5.17".
- **Fix — `count_divergences()` harness bug (v3.2.2):** Bearing-Only used the
  default 10.0 error threshold against a ~64 m error scale, mis-reporting ~176/175
  "divergences" for a filter that was actually consistent (NEES 99.6% in-bounds).
  Gave it a problem-scaled 500 m threshold (analogous to reentry's 5 km); count is
  now **0** for every problem.
- **Fix — RBPF OpenMP data race (v3.2.2):** the per-particle dynamics/observation
  matrices `A,B,Q,H,R` were declared outside the `#pragma omp parallel for` and
  shared across threads. Moved inside the loop. ThreadSanitizer: ~24 worker-vs-worker
  races → 0.
- **Fix — SRUKF innovation-gate consistency (v3.2.2):** a gated/rejected outlier
  previously downdated the covariance by the full `K·S_yy·S_yyᵀ·Kᵀ` (scale applied
  only to the state) → false certainty. Downdate now uses `U = scale·(K·S_yy)`
  (scale² reduction; no-op when rejected).
- **Optimization (v3.2.3):** fixed-size Eigen fast-path for `filtermath::gemm` /
  `mat_vec_mul` (no heap temporaries / dispatch branch for small compile-time-sized
  operands). UKF 10D ~3.6% faster; RMSE/NEES bit-identical.
- Rebuilt, **24/24 CTest pass** (also under `OMP_NUM_THREADS=24`), benchmarks rerun
  (RMSE/NEES unchanged), plots regenerated. Tagged parent `v3.2.3`.

---

## Current Benchmark Results (July 8, 2026 — OptMathKernels v0.5.17)

| Problem | Filter | RMSE | Smoother RMSE | NEES median | In 95% bounds | Divergences |
|---------|--------|------|---------------|-------------|---------------|-------------|
| Coupled Osc (10D) | UKF | 1.457 | 1.148 | 9.89 | 94.5% | 0 |
| Coupled Osc (10D) | SRUKF | 1.457 | 1.148 | 9.89 | 94.5% | 0 |
| Van der Pol (2D) | UKF | 0.468 | -- | 1.14 | 95.9% | 0 |
| Van der Pol (2D) | SRUKF | 0.466 | 0.430 | 1.14 | 96.0% | 0 |
| Bearing-Only (4D) | UKF | 63.81 | -- | 3.77 | 99.6% | 0 |
| Bearing-Only (4D) | SRUKF | 64.17 | 52.03 | 3.77 | 99.6% | 0 |
| Reentry (6D) | UKF | 369.1 | -- | 5.00 | 95.9% | 0 |
| Reentry (6D) | SRUKF | 369.2 | 236.8 | 4.99 | 95.6% | 0 |

All 24 CTest targets pass (8 filter tests/demos + 16 OptimizedKernels tests including CUDA and Vulkan).

---

## Architecture

```
                         ┌─────────────────────────┐
                         │     FilterMath.h         │
                         │  (dispatch layer)        │
                         └──┬──────────┬─────────┬──┘
                            │          │         │
              ┌─────────────▼──┐  ┌───▼────┐  ┌──▼─────────────┐
              │  CUDA (cuBLAS) │  │ SVE2   │  │  Eigen         │
              │  (GEMM ≥32x32) │  │ (GEMM) │  │  (fallback)    │
              │  [SM 75-120]   │  └───┬────┘  └────────────────┘
              └────────────────┘      │
                                 ┌────▼──────────┐
                                 │  NEON          │
                                 │  (GEMM,Cholesky│
                                 │   Solve,Inverse│
                                 └───────┬────────┘
                                         │
    ┌──────────┬──────────┬──────────┬───┴──────┬──────────────────┐
    │ EKF      │ UKF      │ SRUKF    │ RBPKF    │ PKF              │
    │ +Smoother│ +Smoother│ +Smoother│          │ +GPU (CUDA)      │
    │          │          │          │          │ +Vulkan          │
    └──────────┴──────────┴──────────┴──────────┴──────────────────┘
```

---

## Remaining Known Limitations (non-critical)

- FilterMath dispatch: SVE2 only used for GEMM; Cholesky/Inverse/Solve stay on
  NEON > Eigen. OptMathKernels v0.5.10+ now exposes cuSOLVER `cuda_cholesky` /
  `cuda_solve`, but these are intentionally unused on the filter hot paths —
  covariances are small (≤~10×10), so a PCIe round-trip costs more than the
  factorization. Reserved for a future large-matrix dispatch.
- Bearing-Only is genuinely weakly observable (angle-only), so the **range**
  estimate carries a large steady-state RMSE (~64 m) — this is a physics/observability
  limit, not filter instability (NEES stays 99.6% in-bounds). Note: earlier summaries
  called this "divergences"; that count was a `count_divergences()` harness bug
  (threshold 10.0 vs ~64 m error scale), fixed in v3.2.2 — the count is now 0.
- **Reproducibility under OpenMP**: the RBPF/PKF `thread_local` RNG is seeded from
  `std::random_device`, so multithreaded runs are not reproducible w.r.t. a configured
  seed (estimates remain statistically valid; only exact bit-reproducibility is lost).
- All filters are float32-only; no double-precision template support in FilterMath
- CUDA Blackwell requires CUDA 13.x and a `native`/SM 120 build (default arch list
  stops at SM 90) — see Phase 7 and DEVELOPMENT_NOTES.md

## Status: PRODUCTION READY

All critical issues resolved. Production-ready across all filter types and dimensions.

**Active acceleration**: CUDA (SM 75–120, incl. Blackwell) + Vulkan + OpenMP + Eigen (x86_64), NEON + SVE2 + Vulkan (ARM)

**Compute kernels**: OptMathKernels pinned at release tag **v0.5.17**
