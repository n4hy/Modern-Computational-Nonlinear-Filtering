# Computational Architecture

**Analysis Date:** 2026-07-18
**Project:** Modern-Computational-Nonlinear-Filtering (NLF), version 3.4.0
**Language:** C++20
**Focus:** Computation

## Computational Pipeline

The project is a library of nonlinear Bayesian filters and smoothers for discrete-time
state-space models

```
x_k = f(x_{k-1}, t_k, u_k) + w_k,   w_k ~ N(0, Q_k)  [or non-Gaussian for PKF]
y_k = h(x_k, t_k)          + v_k,   v_k ~ N(0, R_k)  [or non-Gaussian for PKF]
```

together with benchmark drivers, plotting scripts, and a CMake / GoogleTest / CTest
harness. There are no interactive notebooks; every entry point is a compiled C++
binary that emits CSV data consumed by the Python plotting scripts.

**End-to-end pipeline (per benchmark or example):**

1. **Model instantiation.** A concrete subclass of a state-space interface is
   constructed with model parameters. Interfaces:
   - `UKFModel::StateSpaceModel<NX, NY>` (`Common/include/StateSpaceModel.h`) --
     used by UKF, SRUKF, and Benchmark problems. Templated on compile-time state
     and observation dimensions; virtual `f`, `h`, `Q`, `R`, `isAngularState`,
     `isAngularObservation`.
   - `SystemModel` (`Common/include/SystemModel.h`) -- used by EKF. Dynamic
     `Eigen::VectorXf` / `Eigen::MatrixXf` API adding Jacobians `F`, `H`.
   - `PKF::StateSpaceModel<NX, NY>` (`PKF/include/state_space_model.hpp`) --
     Templated non-Gaussian model with `propagate`, `observe`,
     `sample_process_noise`, `sample_observation_noise`, `observation_loglik`.
   - `rbpf::NonlinearModel<Types>` +
     `rbpf::ConditionalLinearGaussianModel<Types>`
     (`RBPKF/include/rbpf/state_space_models.hpp`) -- Rao-Blackwell decomposition.
2. **Ground-truth generation.** A driver seeds `std::mt19937` (typically seed 42),
   Cholesky-factors `Q` and `R` via `filtermath::cholesky`, and propagates a truth
   trajectory of `N = T_final / dt` steps while emitting noisy measurements.
   Reference implementation: `Benchmarks/src/run_benchmarks.cpp::generate_trajectory`.
3. **Filter construction and initialization.** A filter is constructed with a
   reference to the model, then `initialize(x0, P0)` sets the prior. SRUKF
   validates `P0` (finite entries + symmetry within relative tolerance 1e-4)
   and computes its Cholesky factor via a three-tier ladder (accelerated
   Cholesky -> jitter + Cholesky -> LDLT reconstruction) before storing `S0` such
   that `P0 = S0 S0^T`. See `UKF/include/SRUKF.h` lines 84--141.
4. **Sequential predict-update loop.** For each timestep k:
   - `predict(t_k, u_k)` propagates the mean and covariance (or its square root)
     through `f` using sigma-point sampling (UKF/SRUKF), analytic Jacobians (EKF),
     empirical samples (PKF), or per-particle conditional Kalman filters (RBPKF).
     Returns the cross-covariance `P_{x_k, x_{k+1}}` used by the smoother.
   - `update(t_k, y_k)` fuses the measurement, computes the Kalman gain via SPD
     solve (never explicit inverse) using `filtermath::kalman_gain`, and applies
     a Joseph-form symmetric covariance update with a covariance-recovery ladder.
5. **Optional smoothing pass.** Fixed-lag or full-interval smoothers execute a
   backward Rauch-Tung-Striebel (RTS) recursion using stored cross-covariances.
6. **CSV emission.** Each driver writes truth, filtered, and smoothed
   trajectories to CSV files at the repository root (see Data Flow below).
7. **Metrics / plotting.** `run_benchmarks` emits `benchmark_results.csv` with
   RMSE, NEES statistics, timings, and divergence counts; `scripts/*.py` render
   comparison plots into `docs/images/`.

## Filter Class Hierarchy

| Filter | Primary Header | Namespace | State Storage | Notes |
|--------|----------------|-----------|---------------|-------|
| EKF | `EKF/include/EKF.h` (`.cpp` in `EKF/src/EKF.cpp`) | (global) | dynamic `x_`, `P_` | Base EKF with Joseph-form update, retains `x_pred_`, `P_pred_` for smoothing. |
| EKF fixed-lag | `EKF/include/EKFFixedLag.h` (`.cpp`) | (global) | ring buffer | RTS smoothing over a fixed lag window. |
| EKF full-interval smoother | `EKF/include/EKFSmoother.h` | (global) | vector of history entries | Batch RTS; `smooth(n>0)` = Gauss-Newton IEKS (exact affine solve per iteration, `Common/include/AffineRTS.h`). |
| UKF | `UKF/include/UKF.h` | `UKFCore` | fixed-size `x_`, `P_` (template `NX`, `NY`) | Julier-Uhlmann / Merwe scaled sigma points; recovery ladder on `P`. |
| SRUKF | `UKF/include/SRUKF.h` (696 lines) | `UKFCore` | fixed-size `x_`, Cholesky factor `S_` (`P = S Sᵀ`) | Square-root form using QR + cholupdate/downdate; angular-state wrap; chi-squared innovation gating. |
| UKF fixed-lag smoother | `UKF/include/UnscentedFixedLagSmoother.h` | `UKFCore` | `std::deque` of `UKFHistoryEntry<NX>` | Two-time step API (`t_prev`, `t_k`). |
| UKF full-interval smoother | `UKF/include/UKFSmoother.h` | `UKFCore` | `std::vector<UKFHistoryEntry<NX>>` | URTS; `smooth(n>0)` = iterated posterior linearisation (IPLS, `UKF/include/PosteriorLinearization.h`). |
| SRUKF fixed-lag smoother | `UKF/include/SRUKFFixedLagSmoother.h` | `UKFCore` | ring buffer | Square-root variant of the fixed-lag RTS. |
| SRUKF full-interval smoother | `UKF/include/SRUKFSmoother.h` | `UKFCore` | vector | Batch RTS on the square-root filter. |
| PKF (bootstrap PF) | `PKF/include/particle_filter.hpp` (437 lines) | `PKF` | `std::vector<State>` particles + `log_weights_` | Systematic/stratified resampling; OpenMP parallel propagation with per-thread RNGs; CPU only (GPU context removed 2026-11-16). |
| PKF fixed-lag | `PKF/include/particle_fixed_lag.hpp` | `PKF` | ancestry buffers | Backward-simulation smoother. |
| RBPKF (Rao-Blackwellized PF) | `RBPKF/include/rbpf/rbpf_core.hpp` (387 lines) | `rbpf` | `std::vector<RbpfParticle>` (each = nonlinear state + linear KF) | Per-particle `LinearKalmanFilter<Types>` in `RBPKF/include/rbpf/kalman_filter.hpp`; OpenMP over particles; optional fixed-lag ancestry. |

All UKF-family filters share the templated model interface
`UKFModel::StateSpaceModel<NX, NY>` and internally allocate `SigmaPoints<NX>` with
`NSIG = 2*NX + 1`. All work in single precision (`float` / `Eigen::MatrixXf` /
`Eigen::Matrix<float, NX, NY>`) throughout; there is no `double` numerical path.

## Sigma-Point Implementation

**Algorithm:** Merwe scaled symmetric sigma-point set with tuning parameters
`(alpha, beta, kappa)`, defined by `lambda = alpha^2 (n + kappa) - n`.

**Location:** `UKF/include/SigmaPoints.h`

**Data structure:** `template<int NX> struct SigmaPoints`
- `SigmaMat X` -- `NX x NSIG` matrix, columns are sigma points (`NSIG = 2*NX + 1`)
- `Weights Wm`, `Weights Wc` -- fixed-size mean and covariance weight vectors
- `float lambda` -- stored scaling parameter (for reproducibility)

**Weights (Merwe convention):**

```
Wm(0) = lambda / (n + lambda)
Wc(0) = lambda / (n + lambda) + (1 - alpha^2 + beta)
Wm(i) = Wc(i) = 1 / (2 (n + lambda))   for i = 1..2n
```

**Degenerate-scaling guard:** `SigmaPoints.h` line 50 detects `n + lambda < 0.5`
(which collapses the spread) and re-solves for `kappa` such that
`n + lambda = 1`, keeping the sigma set well spread.

**Two sigma-point generators:**
1. `generate_sigma_points<NX>(x, P, alpha, beta, kappa, out)` -- Cholesky-factors
   `P` via `filtermath::cholesky` with a Cholesky -> jitter -> LDLT -> diagonal
   fallback chain.
2. `generate_sigma_points_from_sqrt<NX>(x, S, alpha, beta, kappa, out)` -- takes
   the pre-computed square-root `S` directly, used by SRUKF to avoid re-factoring
   the covariance every step.

**Dimension-adaptive tuning (SRUKF only):** In `SRUKF::initialize` (lines
70--79 of `UKF/include/SRUKF.h`), when the user has not overridden `kappa`, the
filter sets:
- `NX <= 5`: `alpha = 1.0, beta = 2.0, kappa = 3 - NX` (classical `n + kappa = 3`).
- `NX > 5`: `alpha = 1.0, beta = 2.0, kappa = 0`.

The historical `alpha = 1e-3` was removed in the audit-remediation pass because
it made `Wm(0)`, `Wc(0)` hugely negative and destabilized the 10D coupled-oscillator
case. See in-source comment at lines 66--69.

**Angular states.** Sigma-point differences on angular states are wrapped to
`[-pi, pi]` via `std::remainder(diff, 2*pi)` before covariance accumulation, and
means over angular components are computed as `atan2(sum(Wm_i sin xi_i),
sum(Wm_i cos xi_i))` (circular mean). See `SRUKF.h::predict` lines 173--225.

## Cholesky / QR Factorization Strategy

**SRUKF square-root propagation** (`UKF/include/SRUKF.h`):

1. **Predict.** Build a `(3*NX) x NX` matrix whose first `2*NX` rows are
   `sqrt(|Wc_i|) * (X_pred[:,i] - x_pred_mean)^T` (with an explicit sign for
   possibly-negative weights) and whose last `NX` rows are `S_Q^T` (Cholesky
   factor of process noise `Q`). Run a Householder QR (`Eigen::HouseholderQR`)
   and take `S_pred = R^T` as the lower-triangular Cholesky factor.
2. **Rank-1 update/downdate for i=0 term.** The zeroth sigma point's contribution
   is applied via a `cholupdate` (positive weight) or `cholupdate_downdate_safe`
   (negative weight) rank-1 modification of `S_pred`. On downdate failure the
   code falls back to a full-covariance reconstruction + Cholesky.
3. **Update.** The innovation covariance factor is similarly maintained by QR +
   cholupdate.

**Recovery ladder for full-covariance updates** (`UKF/include/UKF.h`
lines 150--178 and `UKF/include/SigmaPoints.h` lines 68--98):

```
Level 1: filtermath::cholesky(P)                     [accelerated: NEON -> Eigen]
Level 2: filtermath::cholesky(P + jitter * I)        [jitter = max(1e-6, 1e-8 * trace(P)/N)]
Level 3: Eigen::LDLT reconstruction with sqrt(max(D_i, 1e-10))
Level 4: diagonal clamp -- diag(P).cwiseMax(1e-10).asDiagonal()  [emits stderr warning]
```

The `1e-6` absolute floor plus `1e-8 * trace(P)/N` relative jitter keeps
small-covariance states from being artificially inflated.

## Linear-Algebra Backend

**Primary backend:** Eigen 3.4 (fetched via `FetchContent` if not
system-installed; see `CMakeLists.txt` lines 305--324).

**Dispatch layer:** `Common/include/FilterMath.h` (504 lines). Provides a
`filtermath::` namespace with hardware-dispatched primitives:

| Primitive | Dispatch order | Location |
|-----------|----------------|----------|
| `gemm(A, B)` | CUDA (min_dim >= 32) > SVE2 > NEON > Eigen | lines 83--104 |
| `gemm<fixed>` | Compile-time fixed-size stack-allocated path | lines 125--136 |
| `mat_vec_mul(A, v)` | CUDA (n >= 32) > NEON > Eigen | lines 145--162 |
| `mat_vec_mul<fixed>` | Compile-time fixed-size path | lines 172--182 |
| `cholesky(A)` | NEON > Eigen (`Eigen::LLT`) -- returns empty on failure | lines 199--213 |
| `inverse(A)` | NEON > Eigen (`Eigen::FullPivLU`) | lines 226--239 |
| `solve_spd(A, b)` | NEON > Eigen (`Eigen::LDLT`) | lines 252--264 |
| `solve_spd<fixed>` | Compile-time LDLT | lines 277--285 |
| `solve_spd_mat(A, B)` | Batch LDLT with column-wise fallback | lines 295--310 |
| `trsv_lower`, `trsv_upper` | NEON > Eigen triangular solve | lines 318--338 |
| `kalman_gain(PHt, S)` | `solve_spd_mat` -> `inverse` -> Jacobi-SVD pseudoinverse | lines 349--365 |
| `reduce_sum`, `reduce_max` | CUDA (v.size >= 32^2) > Eigen | lines 395--416 |
| `vec_exp`, `vec_log` | CUDA (v.size >= 32^2) > Eigen | lines 422+ |
| `gpu_available()`, `gpu_sync()` | Runtime query / synchronize | lines 371--389 |

**Backend provisioning:**

- **Eigen 3.4** is a public transitive dependency (exposed across the API
  boundary because filter types are Eigen fixed-size matrices). Required.
- **`OptMathKernels`** (external repository
  `OptimizedKernelsForRaspberryPi5_NvidiaCUDA`) provides the accelerated backends
  (NEON, SVE2, Vulkan, CUDA). It is a required sibling dependency built via
  `add_subdirectory`. Contract in `CMakeLists.txt` lines 336--418; provisioning
  via `bootstrap.sh` or `-DAUTO_CLONE_DEPS=ON`. The installed headers land in
  `<prefix>/include/optmath/` (mirroring the layout in
  `install/include/optmath/{cuda_backend.hpp, neon_kernels.hpp, sve2_kernels.hpp,
  vulkan_backend.hpp, cuda_error.hpp, platform.hpp, radar_kernels.hpp}`).
- **OpenMP** is optional (`find_package(OpenMP)` in `CMakeLists.txt` line 250).
  When present, `PKF::ParticleFilter` and `rbpf::RaoBlackwellizedParticleFilter`
  parallelize the per-particle propagation loop with per-thread RNGs seeded
  deterministically from a base seed.
- **CUDA** is optional and auto-detected via `check_language(CUDA)`
  (`CMakeLists.txt` lines 31--73). When present:
  - `OPTMATH_USE_CUDA` is defined project-wide.
  - `find_package(CUDAToolkit REQUIRED)` and `CUDA::cudart` are linked globally.
  - `CMAKE_CUDA_ARCHITECTURES` defaults to `75 80 86 89 90` (Turing through
    Hopper); CUDA 13+ additionally appends `100 120` (Blackwell SM 100, RTX 50
    series SM 120). Override via `-DCMAKE_CUDA_ARCHITECTURES=native`.
- **Vulkan** is optional. Requested by default (`NLF_ENABLE_VULKAN=ON`) and
  used by PKF for large-N particle vector-add (`particle_filter.hpp` lines
  128--148, guarded on `PKF_HAS_VULKAN && N_ > 100`). Auto-skipped when the SDK
  is absent; `NLF_VULKAN_ACTIVE` records whether it was actually linked in.
- **NEON / SVE2.** ARM-only, gated on `CMAKE_SYSTEM_PROCESSOR` matching aarch64.
  On x86 the SIMD backends are silently disabled and dispatch falls through to
  Eigen. NEON dispatch is compile-time gated on `FILTERMATH_ARM64`; SVE2
  dispatch on `OPTMATH_USE_SVE2` (defined only if OptMath actually built the
  SVE2 sources).

There is no separate BLAS/LAPACK dependency. All dense linear algebra goes
through Eigen (which internally uses SIMD intrinsics) or the OptMathKernels
backend chain.

## State Propagation Abstraction

**Not `std::function`.** The propagation function `f(x, t, u)` and observation
function `h(x, t)` are **virtual member functions** on the model classes, and
the filters hold the model by reference (UKF, SRUKF, EKF) or by template
parameter (RBPKF). Sigma points are propagated by an explicit loop
`for (int i = 0; i < NSIG; ++i) X_pred.col(i) = model_.f(sigmas.X.col(i), t_k, u_k);`
(see `UKF/include/UKF.h` line 53 and `UKF/include/SRUKF.h` lines 156--159).

The RBPKF templates the model directly:
`RaoBlackwellizedParticleFilter<Types, NonlinearModelT, CondLinModelT>`
(`RBPKF/include/rbpf/rbpf_core.hpp` lines 36--38), so per-particle inner loops
can inline `nonlinear_model_.propagate` and `conditional_model_.get_dynamics`.

## Smoother Implementation

**All smoothers are Rauch-Tung-Striebel (RTS) recursions.** No forward-backward
or two-filter alternative is implemented.

**Fixed-lag smoothers** (`UnscentedFixedLagSmoother`, `SRUKFFixedLagSmoother`,
`EKFFixedLag`, `PKF::particle_fixed_lag`): maintain a ring buffer of at most
`lag+1` history entries and run the RTS backward pass whenever the buffer is
full. Two-time-step step API (`step(t_prev, t_k, y_k, u_k)`) so the forward
propagation time (`t_prev`) is distinguished from the measurement time (`t_k`).

**Full-interval smoothers** (`UKFSmoother`, `SRUKFSmoother`, `EKFSmoother`):
retain the ENTIRE forward trajectory and run one backward RTS pass after all
measurements are consumed. `smooth(n_iterations > 0)` then relinearises about
the previous smoothed estimate and solves the affine model exactly from the
ORIGINAL prior, each measurement once (`Common/include/AffineRTS.h`): Jacobians
for the EKF (Gauss-Newton IEKS), sigma-point statistical linear regression for
the UKF/SRUKF (IPLS). On a linear model every iteration returns exact RTS.
(Until 2026-11-16 iteration replayed the forward filter from the smoothed x_0
with covariance P0, which counted the data twice.)

**RTS gain formula** (see `UKFSmoother.h::backward_pass`, lines 98--119):

```
G_j     = P_{cross,j}  * (P_pred_{j+1})^{-1}         [via filtermath::kalman_gain]
x_s[j]  = x_filt[j] + G_j * (x_s[j+1] - x_pred_{j+1})
P_s[j]  = P_filt[j] + G_j * (P_s[j+1] - P_pred_{j+1}) * G_j^T
P_s[j]  = 0.5 (P_s[j] + P_s[j]^T)                    [symmetrize]
```

`P_cross` is returned by every filter's `predict()` and stored in the history
buffer, so no cross-covariance recomputation is needed on the backward pass.

## Particle Filter Design

**Bootstrap Particle Filter** (`PKF/include/particle_filter.hpp`):

- **Proposal:** the state-transition prior (no adapted proposal implemented).
- **Weight update:** `log_weights_[i] += model_->observation_loglik(y_k,
  particles_[i], t_k)`, then normalized in log-space via log-sum-exp to avoid
  underflow.
- **Effective sample size:** `N_eff = 1 / sum(w_i^2)`. Resampling triggers when
  `N_eff < resampling_threshold_ = threshold_frac * N`.
- **Resampling algorithms:** systematic (default) and stratified
  (`PKF/include/resampling.hpp`). Both are O(N) and use a `double`-accumulated
  cumulative sum even when the input weights are `float` (protects against
  cumulative float drift for large N with skewed weights).
- **Parallelization:** OpenMP `parallel for schedule(static)` over particles for
  the propagation and noise-sampling steps (`particle_filter.hpp` lines
  110--126). Each thread uses its own `std::mt19937_64` seeded from
  `base_seed_`; `schedule(static)` fixes the iteration-thread mapping so
  reproducibility is preserved for a given thread count.
- **CPU only.** The CUDA `GPUParticleContext` (`particle_filter_gpu.hpp`) and
  the ARM Vulkan noise-add were removed on 2026-11-16: the model calls are
  virtual host functions, so neither ran the filter on the device, and the
  CUDA path was slower than the CPU path at every N measured.

**Rao-Blackwellized Particle Filter** (RBPKF,
`RBPKF/include/rbpf/rbpf_core.hpp`):

- **Split:** State = `(x_nl, x_lin)`. The nonlinear part `x_nl` is
  particle-represented; the linear-Gaussian part `x_lin` is analytically
  marginalized per particle by a conditional Kalman filter `LinearKalmanFilter<Types>`
  (`RBPKF/include/rbpf/kalman_filter.hpp`). Each particle carries its own
  `(x, P)` linear posterior in addition to its `x_nl` and `log_weight`.
- **Types template:** `rbpf::RbpfTypes<N_NL, N_LIN, N_Y>` defines all matrix
  types with compile-time fixed extents so `filtermath::gemm` / `mat_vec_mul` /
  `solve_spd` bind to the fixed-size fast path -- critical because the KF
  predict/update is called once per particle per step (thousands of times).
- **Weight update:** log-likelihood under the Gaussian innovation of the linear
  Kalman filter (uses `logdet` of `S`, computed stably as `2 * sum(log(diag L))`
  after LLT). The audit-remediation pass introduced `test_rbpf_logdet.cpp` as a
  regression test.
- **Resampling:** systematic (default) or stratified. Ancestry indices are
  stored in a ring buffer when `config.fixed_lag > 0` for fixed-lag smoothing.
- **Parallelization:** OpenMP over particles with per-thread RNGs deterministic
  in `config.seed`.

## Parallelization Strategy

| Layer | Mechanism | Location |
|-------|-----------|----------|
| Fine-grained linear algebra | Eigen SIMD; NEON / SVE2 through OptMathKernels | `Common/include/FilterMath.h` |
| Per-particle propagation | OpenMP `parallel for schedule(static)` with per-thread `std::mt19937_64` | `PKF/include/particle_filter.hpp` lines 110--126; `RBPKF/include/rbpf/rbpf_core.hpp` lines 109--143 |
| Vector reductions / element-wise ops | CUDA (min size = `FILTERMATH_CUDA_MIN_DIM^2` = 1024) | `FilterMath.h` lines 395--432 |
| Large matrix multiply / triangular solve | CUDA (min dim `FILTERMATH_CUDA_MIN_DIM = 32`) | `FilterMath.h` lines 83--104, 145--162 |
| Particle vector-add | Vulkan compute shader (`vec_add.comp.spv`) | `PKF/include/particle_filter.hpp` lines 128--148 |

CUDA dispatch has a runtime toggle:
`filtermath::config::set_cuda_enabled(bool)`. GPU is only used above threshold
sizes to amortize PCIe transfer latency (10--20 us round-trip).

## Benchmark Infrastructure

**Location:** `Benchmarks/`
- `Benchmarks/include/BenchmarkProblems.h` (470 lines): Four canonical problems.
- `Benchmarks/include/BenchmarkRunner.h` (431 lines): `BenchmarkMetrics` struct,
  RMSE aggregation, NEES statistics (median NEES, `pct_in_bounds` against 95%
  chi-squared), convergence-time detection, divergence counting.
- `Benchmarks/src/run_benchmarks.cpp` (766 lines): driver `main()` that runs
  each filter on each problem and emits `benchmark_results.csv`.

**Problems (all in `BenchmarkProblems.h`):**

| Name | State dim | Obs dim | Dynamics | Notes |
|------|-----------|---------|----------|-------|
| `CoupledOscillators` | 10 | 5 | 5 coupled damped oscillators + nonlinear coupling | RK4, dt=0.01, obs positions only via `y_i = pos_i + 0.1 sin(pos_i)` |
| `Lorenz96` | 40 | 10 | Weather-model chaos, F=8 | RK4, obs every 4th variable |
| `VanDerPolDiscontinuous` | 2 | 1 | Van der Pol with discontinuous forcing, mu=5 | RK4, quadratic observation |
| `BearingOnlyTracking` | 4 | 1 or 2 | Constant-velocity target, bearings-only observer | Angular observation |
| `ReentryVehicle` | 6 | 2 | Reentry ballistic vehicle with drag | Nonlinear, high-dynamics |

**Benchmark metrics** (columns in `benchmark_results.csv`, header
`Filter,Problem,RMSE_Overall,RMSE_Position,RMSE_Velocity,RMSE_Smoothed_Overall,
RMSE_Smoothed_Position,RMSE_Smoothed_Velocity,Median_NEES,Pct_In_Bounds,
NEES_Valid,NEES_Total,Avg_Step_Time_ms,Total_Time_ms,Convergence_Time,
Num_Divergences`):

- Per-subvector RMSE (`RMSE_Position` / `RMSE_Velocity`) is present in the
  header but always emitted as `0`; the calculation exists in
  `compute_rmse_indices()` but is intentionally left unwired because "position"
  is model-specific (Reentry's 6D state has a ballistic coefficient that is
  neither position nor velocity). See `BenchmarkRunner.h` lines 27--32.
- NEES uses filtered-covariance NEES with per-step chi-squared upper/lower
  bounds; `NEES_Valid` counts steps where `P` was well-conditioned.
- `Convergence_Time` is `NaN` when the filter never converged; the sentinel of
  "final timestamp" that would print as if it were a measurement was
  deliberately removed (see `BenchmarkRunner.h` lines 50--58).

**Compile flags:** `-fno-fast-math`, `EIGEN_FAST_MATH=0` for the benchmark
executable (`Benchmarks/CMakeLists.txt` line 41--42) so numerical semantics
match a strict IEEE-754 reference.

## Testing Framework

**Framework:** CTest, driven by `add_test(...)` in the root `CMakeLists.txt`
lines 569--586. Assertion macro: `NLF_CHECK(cond, msg)` in
`Common/include/TestCheck.h` -- a plain `if` that survives NDEBUG (regular
`assert()` compiles out under Release, making a test silently pass regardless
of what it computes).

**Registered tests (16 total):**

Primary examples (also serve as smoke tests):
- `EKF_Test` (from `EKF/main.cpp`)
- `UKF_Test`, `SRUKF_Test` (from `UKF/main.cpp`, `UKF/main_srukf.cpp`)
- `PKF_Test`, `PKF_Example` (from `PKF/tests/test_particle.cpp`,
  `PKF/src/example_main.cpp`)
- `RBPF_Basic`, `RBPF_CTRV` (`RBPKF/tests/test_rbpf_basic.cpp`,
  `RBPKF/examples/example_rbpf_ctrv.cpp`)
- `Benchmarks` (from `Benchmarks/src/run_benchmarks.cpp`)

Smoothing regression tests:
- `SRUKF_AngularWrap` (`UKF/tests/test_srukf_angular_wrap.cpp`)
- `SRUKF_Smoother`, `UKF_Smoother`, `EKF_Smoother`
  (`UKF/tests/test_srukf_smoother.cpp`, `test_ukf_smoother.cpp`,
  `EKF/tests/test_ekf_smoother.cpp`)

Audit-remediation regression tests (introduced in commit `2283ca5`):
- `UKF_Numerical` (`UKF/tests/test_ukf_numerical.cpp`)
- `SRUKF_Initialize` (`UKF/tests/test_srukf_initialize.cpp`)
- `PKF_ParticleConst` (`PKF/tests/test_particle_const.cpp`)
- `RBPF_LogDet` (`RBPKF/tests/test_rbpf_logdet.cpp`)

Each executable's `main()` computes deterministic error metrics (fixed seed 42)
and calls `NLF_CHECK` on RMSE ceilings and monotonicity properties (e.g.,
`rmse_smooth < rmse_filt`) before returning.

**Sanitizer lane** (`NLF_ENABLE_SANITIZERS=ON`, `RelWithDebInfo` or `Debug`
only): adds `-fsanitize=address -fsanitize=undefined
-fno-sanitize-recover=undefined -fno-omit-frame-pointer` to C++ TUs (CUDA TUs
excluded). See `CMakeLists.txt` lines 229--247. Enforced in CI by the
`sanitizers` job with `ASAN_OPTIONS=abort_on_error=1:halt_on_error=1` and
`UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`.

**Warning set** (`nlf_add_warning_flags()` in `CMakeLists.txt` lines 203--219):
`-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter` uniformly across
every filter target (pre-audit only PKF and Benchmarks enforced warnings).
GCC additionally silences `-Wstringop-overread` (Eigen AVX-512 LTO false
positive). Deliberately NOT `-Werror`: the library is compiled by many
toolchains and new compiler releases invent warnings that would break user
builds.

## CI Infrastructure

**Location:** `.github/workflows/ci.yml`

**Two jobs, both on `ubuntu-latest`, both triggered on push / PR to `main`:**

1. **`build-and-test`** (Release lane). No CUDA (routes through the CPU / Eigen
   fallback). Installs `build-essential cmake ninja-build libeigen3-dev
   libomp-dev git python3 python3-venv python3-pip`. Pre-clones OptMathKernels
   at pinned tag `v0.5.15` to `$HOME` for reproducibility, then configures with
   `-GNinja -DCMAKE_BUILD_TYPE=Release
   -DOPTMATH_DIR=$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
   -DNLF_BUILD_PYTHON_VENV=OFF`. Runs `ctest --output-on-failure`.
2. **`sanitizers`** (`RelWithDebInfo` + ASan + UBSan lane). Same setup with
   `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DNLF_ENABLE_SANITIZERS=ON`. Tests run
   with `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1`
   (detect_leaks=0 avoids third-party static-init false-positives) and
   `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`.

## MCP Servers

**File:** `.mcp.json` (repo root; git-ignored via `.gitignore` line 26).

Configures eight MCP servers from a local GPD (`get-physics-done`) install at
`/home/n4hy/.gpd/venv/bin/python`:

| Server | Module | Purpose |
|--------|--------|---------|
| `gpd-conventions` | `gpd.mcp.servers.conventions_server` | Convention tracking |
| `gpd-errors` | `gpd.mcp.servers.errors_mcp` | Error catalog |
| `gpd-patterns` | `gpd.mcp.servers.patterns_server` | Pattern library |
| `gpd-protocols` | `gpd.mcp.servers.protocols_server` | Shared protocols |
| `gpd-skills` | `gpd.mcp.servers.skills_server` | Skills registry |
| `gpd-state` | `gpd.mcp.servers.state_server` | Workflow state |
| `gpd-verification` | `gpd.mcp.servers.verification_server` | Verification harness |
| `gpd-arxiv` | `arxiv_mcp_server` | ArXiv search |

There are no simulation MCP servers; the MCP surface is entirely GPD
tooling for research-agent workflows.

## Data Flow

**Input:** Model parameters are baked into the C++ example programs; there is
no external configuration file. The only runtime input is the RNG seed (fixed
at 42 in most examples).

**Output CSV convention (repo root):**

Filename pattern: `<problem>_<filter>[_smooth].csv`.

Files observed at the repo root:

| File | Columns | Written by |
|------|---------|------------|
| `bearing_ukf.csv`, `bearing_srukf.csv` | `time,true_x0..true_x3,filt_x0..filt_x3` | Benchmark BearingOnly problem (`run_benchmarks` UKF/SRUKF path) |
| `bearing_srukf_smooth.csv` | `time,true_*,filt_*,smooth_*` | BearingOnly + SRUKF smoother |
| `vanderpol_ukf.csv`, `vanderpol_srukf.csv` | `time,true_x0,true_x1,filt_x0,filt_x1` | Van der Pol UKF/SRUKF |
| `vanderpol_srukf_smooth.csv` | as above + smoothed cols | Van der Pol SRUKF smoother |
| `reentry_ukf.csv`, `reentry_srukf.csv` | `time,true_x0..x5,filt_x0..x5` | Reentry vehicle 6D |
| `reentry_srukf_smooth.csv` | + smoothed | Reentry smoother |
| `coupled_osc_ukf.csv`, `coupled_osc_srukf.csv` | `time,true_x0..x9,filt_x0..x9` | 10D coupled oscillators |
| `coupled_osc_ukf_smooth.csv`, `coupled_osc_srukf_smooth.csv` | + smoothed | Coupled-osc smoothers |
| `benchmark_results.csv` | Metrics summary (16 columns; see Benchmark Infrastructure section) | `Benchmarks/src/run_benchmarks.cpp` main routine |

Legacy files `ekf_results.csv`, `ukf_results.csv`, `srukf_results.csv`,
`srukf_smoother_results.csv` are produced by the example drivers
(`EKF/main.cpp`, `UKF/main.cpp`, `UKF/main_srukf.cpp`) when run standalone. They
share the same `time,true_*,filt_*[,smooth_*]` convention. CSVs are all
`.gitignore`d except those already committed (`.gitignore` line 12: `*.csv`;
`benchmark_results.csv` and the model output CSVs are committed anyway --
`.gitignore` uses no `!` allowlist for them, so they were `git add`'d
explicitly).

**Plotting:** `scripts/plot_benchmarks.py` (uses pandas + matplotlib) reads
`benchmark_results.csv` and produces grouped bar charts of RMSE, NEES, and
timing across filter/problem pairs. Other `scripts/*.py` (`plot_results.py`,
`plot_optimized.py`, `pkf_plot_results.py`, `ukf_plot_results.py`,
`simple_plot_benchmarks.py`) render individual problems. Output images land in
`docs/images/` (allowlisted in `.gitignore` lines 13--15).

**Python environment:** `.nlfvenv/` is auto-created by CMake if
`NLF_BUILD_PYTHON_VENV=ON` (default). CMake runs `python3 -m venv
--system-site-packages .nlfvenv` and `pip install -r requirements.txt`
(`requirements.txt` pins `pandas>=2.0.0 matplotlib>=3.7.0 numpy>=1.24.0`).
Legacy `nlfvenv/` still exists in the tree but is superseded by `.nlfvenv/`.
CI disables venv creation via `-DNLF_BUILD_PYTHON_VENV=OFF`.

## Performance Characteristics

Measured on the reference build (see `benchmark_results.csv`):

| Filter | Problem | Overall RMSE | Median NEES | Pct In Bounds | Avg step (ms) | Divergences |
|--------|---------|--------------|-------------|---------------|---------------|-------------|
| UKF | CoupledOscillators10D | 1.457 | 9.89 | 94.5% | 0.0085 | 0 |
| SRUKF | CoupledOscillators10D | 1.457 | 9.89 | 94.5% | 0.0078 | 0 |
| UKF+Smoother | CoupledOscillators10D | 1.457 (smooth 1.148) | 9.89 | 94.5% | 0.027 | 0 |
| SRUKF+Smoother | CoupledOscillators10D | 1.457 (smooth 1.148) | 9.89 | 94.5% | 0.041 | 0 |
| UKF | VanDerPol2D | 0.468 | 1.14 | 95.9% | 0.00039 | 0 |
| SRUKF | VanDerPol2D | 0.466 | 1.14 | 96.0% | 0.00029 | 0 |
| SRUKF+Smoother | VanDerPol2D | 0.466 (smooth 0.430) | 1.14 | 96.0% | 0.0057 | 0 |
| UKF | BearingOnly4D | 63.81 | 3.77 | 99.6% | 0.00056 | 176 |
| SRUKF | BearingOnly4D | 64.17 | 3.77 | 99.6% | 0.00044 | 175 |
| SRUKF+Smoother | BearingOnly4D | 64.17 (smooth 52.03) | 3.77 | 99.6% | 0.012 | 175 |
| UKF | ReentryVehicle6D | 369.1 | 5.00 | 95.9% | 0.0022 | 0 |
| SRUKF | ReentryVehicle6D | 369.2 | 4.99 | 95.6% | 0.0023 | 0 |
| SRUKF+Smoother | ReentryVehicle6D | 369.2 (smooth 236.9) | 4.99 | 95.6% | 0.018 | 0 |

Step times scale roughly with `NX^2` for UKF/SRUKF (dominated by sigma-point
covariance reconstruction) and by particle count for PKF/RBPKF. The high
divergence count on BearingOnly4D reflects observability collapse in
bearings-only geometry, not filter defects.

---

_Architecture analysis: 2026-07-18. Reference build: NLF v3.4.0, Eigen 3.4,
C++20, OptMathKernels v0.5.15 (CI pin). CUDA optional (auto-detected, arch
75--90 or 75--120 with CUDA 13+)._
