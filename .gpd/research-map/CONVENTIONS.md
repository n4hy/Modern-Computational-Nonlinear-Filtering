# Notation, Unit, and Numerical Conventions

**Analysis Date:** 2026-07-18
**Project version:** v3.4.2 (post-Phase 8 audit-remediation)
**Focus:** methodology / conventions (research-mapper report — not the notation-coordinator lock)

This document REPORTS the conventions actually in use across the C++ code and CSV outputs. It is not a prescription (that would belong in `.gpd/CONVENTIONS.md` if a notation-coordinator run creates one). Downstream planner/executor agents should follow what is documented here when extending existing filters or benchmark problems.

---

## 1. State-Space Notation

### 1.1 Symbol conventions (as they appear in code and comments)

| Symbol | Meaning | Where seen |
|--------|---------|-----------|
| `x` (`State`) | State vector, `Eigen::Matrix<float, NX, 1>` | `Common/include/StateSpaceModel.h` (l. 20), `Common/include/SystemModel.h` |
| `y` (`Observation`, also `z` in some Kalman literature) | Measurement vector, `Eigen::Matrix<float, NY, 1>`. **This codebase uses `y`, not `z`.** | `Common/include/StateSpaceModel.h` (l. 21); `UKF.h::update()` takes `Observation& y_k` |
| `u` | Control input, same size as State | `Common/include/StateSpaceModel.h` (l. 33), `EKF::predict(u, t)` |
| `w_k` | Process noise (implicit via `Q`) | Documented in `Common/include/StateSpaceModel.h` (l. 28) but not passed as an argument — process noise is characterised only by its covariance `Q(t)` |
| `v_k` | Measurement noise (implicit via `R`) | `Common/include/StateSpaceModel.h` (l. 36) |
| `Q` (function `model.Q(t)`) | Process noise covariance, `StateMat` (NX×NX) | `Common/include/StateSpaceModel.h::Q(float t)` |
| `R` (function `model.R(t)`) | Measurement noise covariance, `ObsMat` (NY×NY) | `Common/include/StateSpaceModel.h::R(float t)` |
| `P` (`StateMat`) | State covariance, `Eigen::Matrix<float, NX, NX>` | `UKF/include/UKF.h` (member `P_`), `EKF/include/EKF.h` (member `P_`) |
| `S` (SRUKF square root) | Cholesky factor of P such that `P = S · Sᵀ` | `UKF/include/SRUKF.h` (member `S_`, l. 550) — **not to be confused with innovation covariance below** |
| `S_yy` (or `S`) | Innovation covariance in UKF/EKF (`S = H·P·Hᵀ + R` or `Pyy + R`) | `EKF.cpp` (l. 66), `UKF.h` (l. 119) |
| `S_yy` (as Cholesky factor) | In SRUKF, Cholesky factor of the innovation covariance | `SRUKF.h` (l. 337) |
| `Pxy` (`CrossMat`) | State/measurement cross-covariance, `Eigen::Matrix<float, NX, NY>` | `UKF/include/UKF.h` (l. 20), `SRUKF.h` (l. 396) |
| `P_cross` | State/state cross-covariance across a time step (for smoothing) | `UKF.h::predict()` returns this; `SRUKF.h::predict()` returns this |
| `K` | Kalman gain, `Eigen::Matrix<float, NX, NY>` | `UKF.h` (l. 124), `SRUKF.h` (l. 403), `EKF.cpp` (l. 70) |
| `F` (function `model.F(x,u,t)`) | Jacobian of `f` in EKF, ∂f/∂x | `Common/include/SystemModel.h` (l. 33) — **only for EKF via `SystemModel`; UKF-family uses derivative-free sigma points** |
| `H` (function `model.H(x,t)`) | Jacobian of `h` in EKF, ∂h/∂x | `Common/include/SystemModel.h` (l. 39) |
| `f` (function `model.f(x, t, u)`) | Nonlinear state transition | `StateSpaceModel::f`, `SystemModel::f` — **argument order differs, see §1.4** |
| `h` (function `model.h(x, t)`) | Nonlinear observation | `StateSpaceModel::h`, `SystemModel::h` |
| `NX` | State dimension (template parameter) | Used throughout as `template<int NX, int NY>` |
| `NY` | Observation dimension (template parameter) | Same |
| `NSIG` | Number of sigma points, `= 2·NX + 1` | `SigmaPoints<NX>::NSIG`, l. 14 |

### 1.2 State/observation vector orientation

Column vectors: `State = Eigen::Matrix<float, NX, 1>`, `Observation = Eigen::Matrix<float, NY, 1>`. All matrix products are computed in the mathematically standard right-multiplied form (`K · innov`, `H · P · Hᵀ`, etc.).

### 1.3 Sigma-point matrix layout

`SigmaMat = Eigen::Matrix<float, NX, NSIG>` — sigma points are stored as COLUMNS, not rows. `SigmaPoints<NX>::X.col(i)` is the i-th sigma point. See `UKF/include/SigmaPoints.h` (l. 17, l. 102-106).

### 1.4 CRITICAL: `f()` argument-order divergence between filter families

Two incompatible base classes exist. This is a REAL convention conflict downstream code must respect:

| Base class | File | Signature |
|------------|------|-----------|
| `UKFModel::StateSpaceModel<NX,NY>` | `Common/include/StateSpaceModel.h` (l. 31-33) | `State f(const State& x_prev, float t_k, const Eigen::Ref<const State>& u_k)` — **(x, t, u)** |
| `SystemModel` (EKF base) | `Common/include/SystemModel.h` (l. 21) | `Eigen::VectorXf f(const Eigen::VectorXf& x, const Eigen::VectorXf& u, float t)` — **(x, u, t)** |
| `PKF::StateSpaceModel<NX,NY>` | `PKF/include/state_space_model.hpp` (l. 37-39) | `State propagate(const State& x_prev, float t_k, const Eigen::Ref<const State>& u_k)` — **(x, t, u)** and named `propagate`, not `f` |

The UKF/SRUKF/PKF families use `(x, t, u)`; EKF uses `(x, u, t)`. New models must inherit from the base for the filter family they target and use the corresponding argument order.

### 1.5 Model-dimension size limit (PKF only)

`PKF::StateSpaceModel<NX, NY>` has `static_assert(NX <= 20, ...)` and `static_assert(NY <= 20, ...)` — see `PKF/include/state_space_model.hpp` (l. 20-21). No such assertion exists for the UKF/SRUKF/EKF/RBPF families.

---

## 2. Time Indexing and Predict/Update Convention

### 2.1 Timestep alignment (project-critical, applies to every benchmark and test)

The convention is documented at length in `Benchmarks/src/run_benchmarks.cpp` (l. 27-53) — the code enforces it everywhere. Summary:

| Quantity | Meaning |
|----------|---------|
| `true_states[i]` | State at `times[i]` |
| `measurements[i]` | `y_i = h(x_i, times[i]) + v` observes `x_i` AT `t_i` |
| `true_states[i+1]` | `f(x_i, times[i]) + w` — **f is evaluated at the SOURCE time (start of interval), not the target** |
| `initialize(x0, P0)` | Places the estimate AT `times[0]` — iteration 0 must NOT call `predict()`, only `update()` |
| `predict(t, u)` | The argument `t` is the time the state **currently IS**, not where it is going. Advancing from `t_{i−1}` to `t_i` therefore passes `times[i−1]` |
| `update(t, y)` | Fuses measurement taken AT time `t` |

**Practical loop** (from `run_benchmarks.cpp` l. 157-172):
```cpp
for (size_t i = 0; i < data.times.size(); ++i) {
    if (i > 0) filter.predict(data.times[i - 1], u);
    filter.update(data.times[i], data.measurements[i]);
}
```

The smoother variant uses `observe_initial(t0, y0)` for iteration 0 and `step(t_prev, t_k, y_k, u)` — two distinct times — for subsequent iterations. See `run_benchmarks.cpp` (l. 283-284).

Violating this alignment biases every RMSE by up to ~0.6% — this was fixed in commit `3dbbf4c`.

### 2.2 Notation for predict vs update covariances (EKF only)

The EKF stores both:
- `P_pred_`: `P_{k|k−1}` — predicted covariance after `predict()`
- `P_`: `P_{k|k}` after `update()` (also holds `P_{k−1|k−1}` at the start of `predict()`)

Comments in `EKF/include/EKF.h` (l. 39, 42) use the standard `P_{k|k-1}` / `P_{k|k}` notation. UKF/SRUKF do NOT expose separate predicted covariance; they mutate `P_` (or `S_`) in place.

### 2.3 IMPORTANT: `update()` first, then `predict()` — this is a "predict-with-source-time" convention

Note that the initialization pattern is the reverse of the classical textbook "predict → update". Because `initialize()` leaves the estimate AT `times[0]`, iteration 0 fuses `y_0` where the estimate already sits, and only subsequent iterations propagate. The order of operations INSIDE each subsequent iteration is still the classical `predict → update`.

---

## 3. Sigma-Point (Merwe scaled) Convention

### 3.1 Parameter conventions

Parameters (`alpha`, `beta`, `kappa`) and derived `lambda` follow the standard Julier–Uhlmann–van der Merwe parameterisation. See `UKF/include/SigmaPoints.h::generate_sigma_points<NX>()` (l. 37-107).

Formula: `lambda = alpha² · (n + kappa) − n`, where `n = NX`.

Weights:
- `Wm(0) = lambda / (n + lambda)` — central mean weight
- `Wc(0) = lambda / (n + lambda) + (1 − alpha² + beta)` — central covariance weight
- `Wm(i) = Wc(i) = 1 / (2·(n + lambda))` for `i = 1 .. 2n`

Sigma-point generation: `X.col(0) = x`, `X.col(i+1) = x + sqrt(n+lambda) · L.col(i)`, `X.col(i+1+n) = x − sqrt(n+lambda) · L.col(i)`, where `L = chol(P)` (see §4.1 for triangular convention).

### 3.2 Default parameter values (differ between UKF and SRUKF)

| Filter | alpha | beta | kappa | Location |
|--------|-------|------|-------|----------|
| **UKF** (`UKF/include/UKF.h`, l. 24-26) | 1e-3 | 2.0 | 0.0 | Fixed defaults; not dimension-adaptive |
| **SRUKF** (`UKF/include/SRUKF.h`, l. 40-42 + l. 70-79) | 1.0 | 2.0 | 3−NX if NX≤5, else 0 | **Dimension-adaptive**, set in `initialize()` |

The SRUKF defaults are the outcome of Bug #1 in `COMPARISON_RESULTS.md`: `alpha = 1e-3` with `n = 10` produced `Wc(0) ≈ −986,894` and destabilised the 10D coupled-oscillator problem. The dimension-adaptive rule keeps `Wc(0)` finite and moderate.

**Caveat (planner/executor beware):** the plain `UKF<NX,NY>` class **retains the old defaults** (`alpha=1e-3, kappa=0`). It relies on the `SigmaPoints.h` guard `if (n_lambda < 0.5f) { kappa = n/alpha² − n; ... }` (l. 50-54) to avoid catastrophic collapse of the spread. New callers should either use SRUKF or explicitly set `alpha=1.0f, kappa=3.0f-NX` before initializing.

### 3.3 Cholesky/matrix-square-root convention for sigma points

- **Cholesky lower-triangular convention**: `L = chol(P)` such that `P = L · Lᵀ`. This is what `filtermath::cholesky(P)` returns and what `SigmaPoints.h` assumes.
- SRUKF stores `S_` such that `P = S · Sᵀ` — same convention (lower-triangular).
- QR-decomposition path (`SRUKF.h` l. 230): `chi_diff_T` is `3·NX × NX`, factored via `HouseholderQR`. The upper-triangular `R` factor is transposed to give the lower-triangular Cholesky factor `S_pred = Rᵀ`.
- Fallback chain when Cholesky fails: (1) accelerated `filtermath::cholesky` → (2) add relative jitter `max(1e-6, 1e-8 · trace(P)/NX)` and retry → (3) `Eigen::LDLT` reconstruction (`L · diag(sqrt(D))`) → (4) diagonal-only fallback (`sqrt(max(P(i,i), 1e-8))`). See `SigmaPoints.h` (l. 68-99) and `SRUKF.h` (l. 103-141).

**Do not** use `Eigen::SelfAdjointEigenSolver` for square roots in the hot path — the codebase only uses it inside NEES / test PSD checks (`BenchmarkRunner.h` l. 60, `test_ukf_smoother.cpp` l. 60).

---

## 4. Covariance Update Conventions

### 4.1 Joseph form (used everywhere)

Every filter uses a symmetric (Joseph or Joseph-style) covariance update as the numerical-stability default:

| Filter | Form | Location |
|--------|------|----------|
| EKF | `P = (I − KH)·P·(I − KH)ᵀ + K·R·Kᵀ` | `EKF/src/EKF.cpp` (l. 76-86) |
| UKF | `P = P − K·Pxyᵀ − Pxy·Kᵀ + K·S·Kᵀ` (Joseph-style four-term symmetric form, algebraically identical to `P − K·S·Kᵀ`) | `UKF/include/UKF.h` (l. 132-144) |
| SRUKF | Rank-1 Cholesky downdates on `S_`, with **Joseph partial-update scaling** `downdate_scale = sqrt(2s − s²)` when the innovation gate scales the correction by `s` | `SRUKF.h` (l. 456-525) |
| RBPKF (per-particle KF) | Joseph form via `filtermath::kalman_gain` and SPD solve | `RBPKF/include/rbpf/kalman_filter.hpp` |

**Never** use the shorter `P = P − K·S·Kᵀ` form without symmetrization — the codebase already has the Joseph form wired everywhere for a reason (the `-ffast-math` incident, Issue #5 in README.md).

### 4.2 Covariance recovery ladder (LLT → jittered LLT → LDLT → diagonal)

`UKF.h::update()` and `SRUKF.h::update()` share an identical recovery ladder (l. 150-178 in UKF.h; l. 495-521 in SRUKF.h). If Cholesky (LLT) of a candidate covariance fails, the sequence is:

1. LLT of `P_new` — accept if `Eigen::Success`
2. Add relative jitter: `jitter = max(1e-6, 1e-8 · max(trace(P)/NX, 0))`, add `jitter · I`, retry LLT
3. `LDLT` reconstruction: `L·diag(sqrt(max(D, 1e-10)))`
4. Last resort: clamp diagonal to `≥ 1e-10`, discard off-diagonals

The rescue is jitter-scale-aware (relative to the trace of `P`) rather than an unconditional `1e-6 · I` add — small-covariance states are not artificially inflated.

### 4.3 SPD solve, not explicit inverse

Kalman gain is computed via `filtermath::kalman_gain(PHt, S)` which calls `solve_spd_mat` (a Cholesky/LDLT solve), avoiding explicit `S⁻¹`. See `Common/include/FilterMath.h` (l. 349-353). This is O(n²) vs O(n³) and numerically better-behaved.

**Do not** use `.inverse()` in filter hot paths. `filtermath::inverse` exists (`FilterMath.h` l. 223) but is documented as "Prefer solve_spd() or kalman_gain() when possible".

### 4.4 Symmetrization after every covariance update

Every `predict()` and `update()` symmetrizes `P` with `P = 0.5f · (P + P.transpose())` to shed floating-point asymmetry (`EKF.cpp` l. 40, 89; `UKF.h` l. 78, 145; `SRUKF.h` in the P_new fallback; RBPF `kalman_filter.hpp` l. 56).

---

## 5. Innovation Gating and NIS

Applies only to SRUKF (R33 audit item). See `SRUKF.h` (l. 411-451).

- Chi-squared gate threshold: `innovation_gate_chi2_ = 25.0f` (member; settable via `setInnovationGateChi2()`). Default 25 ≈ 3-sigma for NY=3.
- **NIS** (Normalized Innovation Squared): `NIS = innovᵀ · (S_yy · S_yyᵀ)⁻¹ · innov`, computed via a triangular solve on `S_yy` — never via an explicit inverse. Exposed as `getLastNIS()`.
- Gating action: if `NIS > threshold`, either down-scale correction by `sqrt(threshold / NIS)` (default) or reject entirely (`setRejectOutliers(true)`).
- **Joseph partial-update consistency**: when the gate scales the state correction by `s`, the covariance downdate uses `sqrt(2s − s²)` to keep the Joseph form self-consistent. `s=1` → full update; `s=0` (rejected outlier) → zero downdate. See `SRUKF.h` (l. 461-474) and Phase 8 audit item.

Gated events emit a single line to `std::clog` (only the first firing per filter instance). Count is exposed via `getGatedCount()`.

---

## 6. Angular States and Observations

Two virtual predicates on `StateSpaceModel<NX,NY>` (default `false`):

- `isAngularState(int i)` — l. 57 in `StateSpaceModel.h`. When `true`, SRUKF wraps state differences to [−π, π] using `std::remainder(diff(j), 2π)` and computes the mean of angular states via the **circular mean** (`atan2(sum(Wm·sin), sum(Wm·cos))`), not the arithmetic mean. See `SRUKF.h` (l. 173-183, 218-223).
- `isAngularObservation(int i)` — l. 64. Wraps innovation `y − ŷ` to [−π, π] in the update step (R32). Without this, an interferometer-style innovation straddling the ±π branch cut would inject a ~2π catastrophic correction. See `SRUKF.h` (l. 383-386, 416-420).

**Convention:** angles are in **radians throughout** (see §7.4). Wrapping is to [−π, π].

**Coverage:** currently only `AngleModel` in `UKF/tests/test_srukf_angular_wrap.cpp` overrides `isAngularObservation`. None of the default benchmark problems in `Benchmarks/include/BenchmarkProblems.h` use these predicates — including `BearingOnlyTracking` and `ReentryVehicle`, both of which contain angular observations (`atan2`, `asin`). Bearing-only's steady-state trajectory keeps innovations well inside [−π, π] so the omission has not surfaced as a bug there, but it is a latent risk for callers whose observer geometry changes.

---

## 7. Units, Coordinate Systems, and Physical Conventions

### 7.1 Numeric precision (project-wide)

**Single precision `float` everywhere.** No double-precision path in FilterMath, no template `Scalar` parameter. Documented in DEVELOPMENT_NOTES.md ("All filters are float32-only; no double-precision template support in FilterMath"). All `Eigen::Matrix<...>` in filter code are `float`-scalar. This is a **hard constraint** — do not introduce doubles into a filter hot path.

### 7.2 Unit systems (per benchmark)

The library is unit-agnostic — the filter code carries no unit tags — but the benchmark problems use these units:

| Problem | Units |
|---------|-------|
| **Van der Pol** (`BenchmarkProblems.h` l. 189-253) | Dimensionless (natural units). `mu = 5.0`, no explicit length/time scale |
| **Coupled Oscillators (10D)** (l. 17-111) | Dimensionless. `omega = 2.0` (angular freq), `dt = 0.01` |
| **Bearing-Only** (l. 405-466) | Length in **meters**; time in **seconds**; angle in **radians**. Observer platform on a 100 m circle turning at 0.1 rad/s. `R = 0.01 rad²` bearing noise |
| **Reentry Vehicle** (l. 262-396) | **SI throughout**: position m, velocity m/s, mass in the ballistic coefficient (500 kg/m²), gravitational parameter `mu = 3.986004418e14 m³/s²` (Earth), scale height 9000 m, Earth radius 6371000 m, sea-level density 1.225 kg/m³. Observation: range in m, azimuth in rad, elevation in rad |
| **Drag Ball** (`UKF/include/DragBallModel.h`) | Position m, velocity m/s, gravity 9.81 m/s². Time in seconds via `dt=0.1` |
| **Nonlinear Oscillator** (`EKF/include/NonlinearOscillator.h`) | Dimensionless. `omega² = 2`, damping 0.5, dt 0.01 |
| **Lorenz-63** (in `PKF/include/lorenz63_model.hpp`) | Dimensionless (chaotic attractor coordinates) |
| **CTRV** (`RBPKF/examples/example_rbpf_ctrv.cpp`) | Constant-turn-rate-velocity vehicle model. Length m, time s, angle rad |

### 7.3 Coordinate frames

- **Reentry Vehicle** (`BenchmarkProblems.h` l. 262-396): ECEF-like inertial frame. Origin at Earth center; radar located at `(R_earth, 0, 0) = (6371000, 0, 0) m` on the +x-axis surface. Observations `[range, azimuth, elevation]` are relative to the radar:
  - `range = |pos − radar_pos|`
  - `azimuth = atan2(rel_y, rel_x)` — in-plane angle from radar +x axis
  - `elevation = asin(rel_z / range)` — with `std::clamp(arg, -1, 1)` guard against FP overshoot
  - **NOT** a downrange-altitude parametrisation
- **Bearing-Only Tracking** (l. 405-466): 2D Cartesian scene frame; observer platform position `(100 cos(0.1 t), 100 sin(0.1 t))`. Bearing measurement is `atan2(target_y − obs_y, target_x − obs_x)`.
- **Drag Ball**: 2D vertical-plane Cartesian, position `(x, y)`, velocity `(vx, vy)`, gravity acts on `vy`.

### 7.4 Angular conventions

- All angles in **radians**. No degrees anywhere in the filter code or benchmark models.
- Angle wrap window: **[−π, π]** using `std::remainder(x, 2·π)` where `π = std::numbers::pi_v<float>` (C++20 `<numbers>`). See `SRUKF.h` (l. 220).
- `atan2(y, x)` returns in [−π, π] per the C++ standard; project relies on this.
- `asin` arguments always clamped to [−1, 1] before use — see `ReentryVehicle::h()` l. 320 for the pattern.

---

## 8. Output CSV File Conventions

### 8.1 Trajectory CSV format (per benchmark)

Format written by `save_trajectory_csv()` in `Benchmarks/include/BenchmarkRunner.h` (l. 370-410).

Column layout: `time, true_x0, ..., true_x{NX-1}, filt_x0, ..., filt_x{NX-1}[, smooth_x0, ..., smooth_x{NX-1}]`. Smoothed columns are emitted only when the run includes a smoother.

Example headers (from actual CSVs on disk):

| File | Header |
|------|--------|
| `vanderpol_ukf.csv` | `time,true_x0,true_x1,filt_x0,filt_x1` |
| `bearing_srukf_smooth.csv` | `time,true_x0,true_x1,true_x2,true_x3,filt_x0,filt_x1,filt_x2,filt_x3,smooth_x0,smooth_x1,smooth_x2,smooth_x3` |
| `reentry_srukf.csv` | `time,true_x0,...,true_x5,filt_x0,...,filt_x5` |
| `coupled_osc_srukf_smooth.csv` | `time,true_x0..true_x9,filt_x0..filt_x9,smooth_x0..smooth_x9` |

### 8.2 File-name convention

`<problem_prefix>_<filter>[_smooth].csv` where:

| Problem prefix | Problem |
|----------------|---------|
| `coupled_osc_` | Coupled Oscillators 10D |
| `vanderpol_` | Van der Pol 2D |
| `bearing_` | Bearing-Only 4D |
| `reentry_` | Reentry Vehicle 6D |

`<filter>` is one of `ukf`, `srukf`. Suffix `_smooth` denotes a smoother run (fixed-lag). See explicit call sites in `run_benchmarks.cpp` (l. 436, 448, 463, 478, ...).

### 8.3 Summary metrics CSV

`benchmark_results.csv` header (from `BenchmarkRunner.h` l. 110-117):

```
Filter,Problem,RMSE_Overall,RMSE_Smoothed_Overall,Median_NEES,Pct_In_Bounds,NEES_Valid,NEES_Total,Avg_Step_Time_ms,Total_Time_ms,Convergence_Time,Num_Divergences
```

Note: `RMSE_Position` / `RMSE_Velocity` columns were **removed** in v3.4.1 (`c8d2905`) because they were emitted as fabricated `0.0` (never assigned). Any downstream tool that parses this CSV must handle both column sets — old CSVs on disk have those columns (compare `head -1 benchmark_results.csv` shows they are still present in the on-disk artifact), but freshly regenerated CSVs will not. **Convergence_Time is empty (not zero) when the filter never converged**; see `save_to_csv()` l. 104-107.

### 8.4 Smoothed-state time alignment in CSV

`smoothed_states[i]` is the smoothed estimate for time index `i − smoother_lag`. The plotting scripts must offset the smoothed column by `smoother_lag` steps to align with the truth. The trajectory CSV writer emits raw indices; alignment is the plotter's job (`scripts/plot_benchmarks.py`).

---

## 9. C++ Code Style / Structural Conventions

### 9.1 Namespaces

**Not** `optmath::`. The optmath namespace belongs to the external OptimizedKernels dependency (kernels only). This library uses:

| Namespace | Content | Location |
|-----------|---------|----------|
| `UKFCore` | UKF, SRUKF, sigma points, unscented smoothers | `UKF/include/UKF.h`, `SRUKF.h`, `UKFSmoother.h`, `SRUKFSmoother.h`, `SRUKFFixedLagSmoother.h`, `UnscentedFixedLagSmoother.h`, `SigmaPoints.h` |
| `UKFModel` | Base classes for UKF/SRUKF models | `Common/include/StateSpaceModel.h` |
| `PKF` | Particle filter, its models, resampling | `PKF/include/*.hpp` |
| `PKF::Resampling` | Systematic/stratified resamplers | `PKF/include/resampling.hpp` |
| `rbpf` | Rao-Blackwellized particle filter | `RBPKF/include/rbpf/*.hpp` |
| `filtermath` | Dispatch layer (GEMM, Cholesky, kalman_gain) | `Common/include/FilterMath.h` |
| `filtermath::config` | Runtime enable toggle for CUDA | `Common/include/FilterMath.h` l. 63-71 |
| `Benchmark` | All benchmark problem classes | `Benchmarks/include/BenchmarkProblems.h` |
| `optmath::neon`, `optmath::sve2`, `optmath::cuda` | External kernels only (not owned by this repo) | via includes |
| **EKF class** — no namespace (`class EKF { ... }` at file scope) | `EKF/include/EKF.h` | Note: this is inconsistent with the UKF family; a future refactor may want to place EKF into a namespace |

### 9.2 File-naming conventions (inconsistent between subprojects — historical)

Downstream code must be aware of this. The convention within each subdirectory is enforced:

| Subdir | Header extension | Class-name style | Filename style |
|--------|------------------|------------------|----------------|
| `Common/include/` | `.h` | `PascalCase` | `PascalCase.h` (e.g. `FilterMath.h`, `StateSpaceModel.h`) |
| `EKF/include/` | `.h` | `PascalCase` | `PascalCase.h` |
| `UKF/include/` | `.h` | `PascalCase` (UKF, SRUKF, SigmaPoints) | `PascalCase.h` |
| `PKF/include/` | `.hpp` | `snake_case` classes... actually `PascalCase` classes in `snake_case` files | `snake_case.hpp` (e.g. `particle_filter.hpp`) |
| `RBPKF/include/rbpf/` | `.hpp` | `PascalCase` | `snake_case.hpp` within `rbpf/` subdirectory |
| `Benchmarks/include/` | `.h` | `PascalCase` | `PascalCase.h` |

Do not "fix" this. Each subdirectory internally follows its own convention consistently. Introducing `PKF/include/ParticleFilter.h` would break the pattern.

### 9.3 Header include guards

**Both patterns coexist** — do not standardize without a full sweep:

- `#ifndef XXX_H / #define XXX_H / ... / #endif` — most `.h` files under `Common/`, `EKF/`, `UKF/`, `Benchmarks/` use this (e.g. `#ifndef SRUKF_H`, `#ifndef FILTERMATH_H`).
- `#ifndef PKF_XXX_HPP / #define PKF_XXX_HPP` — PKF headers use `PKF_`-prefixed guards.
- `#ifndef RBPF_XXX_HPP / #define RBPF_XXX_HPP` — RBPKF headers use `RBPF_`-prefixed guards.

`#pragma once` is **not** used anywhere in the filter subdirectories. It IS the convention in the OptimizedKernels external dependency, but not here.

### 9.4 Template parameter names

`template<int NX, int NY>` throughout — never `N, M` or `StateDim, ObsDim` at the template-argument site. Types-struct pattern in RBPKF uses `Types::Nlin`, `Types::Ny` (see `RBPKF/include/rbpf/kalman_filter.hpp` l. 40-51).

Template classes expose static constexpr aliases as documentation:
- `static constexpr int StateDim = NX;`
- `static constexpr int ObsDim = NY;`
- `static constexpr int NSIG = 2·NX + 1;` (SigmaPoints)

See `StateSpaceModel.h` l. 17-18, `BenchmarkProblems.h` l. 20-21.

### 9.5 Type aliases (per class)

Standardized within a class template:

```cpp
using State       = Eigen::Matrix<float, NX, 1>;
using Observation = Eigen::Matrix<float, NY, 1>;
using StateMat    = Eigen::Matrix<float, NX, NX>;
using ObsMat      = Eigen::Matrix<float, NY, NY>;
using CrossMat    = Eigen::Matrix<float, NX, NY>;   // UKF/SRUKF only
using SigmaPts    = SigmaPoints<NX>;                // UKF/SRUKF only
```

Any new filter class in this codebase should adopt the same alias set.

### 9.6 C++ language standard

**C++20 minimum.** `<numbers>` header (`std::numbers::pi_v<float>`) used in `SRUKF.h`. Concepts / ranges available but sparingly used. See `CMakeLists.txt` for the requirement (declared globally).

---

## 10. Test-Case Parameter Conventions

### 10.1 Initial states (declared in `run_benchmarks.cpp`)

| Problem | `x0` | `P0` |
|---------|------|------|
| Coupled Osc 10D | `[1, 0, -0.5, 0, 0.8, 0, -0.3, 0, 0.6, 0]` (positions and zero velocities) | Identity 10×10 |
| Van der Pol 2D | `[1, 0]` | Identity 2×2 |
| Bearing-Only 4D | `[200, 0, 10, 5]` (position + velocity) | `10 · I` (large initial uncertainty) |
| Reentry 6D | `[R0 + 0.7·80000, 0.5·80000, 0.5·80000, -500, 2000, -300]` (80 km altitude, three-axis offset for observability, three-component velocity) | Block-diagonal: `P0[0:3,0:3] = 100000·I` (~316 m std dev on position), `P0[3:6,3:6] = 10000·I` (~100 m/s std dev on velocity) |

### 10.2 Simulation lengths and timesteps (per benchmark)

| Problem | `T_final` (s) | `dt` (s) | Seed |
|---------|--------------|----------|------|
| Coupled Osc 10D | 50.0 | 0.01 | 42 |
| Van der Pol 2D | 20.0 | 0.01 | 43 |
| Bearing-Only 4D | 30.0 | 0.1 | 44 |
| Reentry 6D | 30.0 | 0.1 | 45 |

Seeds are per-problem — do not consolidate them onto a single seed unless you want to invalidate every published RMSE.

### 10.3 Smoother lag (per benchmark)

| Problem | `lag` steps | Wall time |
|---------|-------------|-----------|
| Coupled Osc 10D | 20 | 0.2 s |
| Van der Pol 2D | 20 | 0.2 s |
| Bearing-Only 4D | 30 | 3.0 s |
| Reentry 6D | 20 | 2.0 s |

See `run_benchmarks.cpp` calls to `run_srukf_smoother_benchmark(..., lag, ...)`.

### 10.4 Divergence threshold (per benchmark)

`count_divergences()` takes an absolute error norm threshold. The suite passes problem-scaled thresholds to distinguish "the filter lost the target" from "the filter tracks with a large steady-state RMSE":

| Problem | Threshold (m or dimensionless) | Rationale |
|---------|-------------------------------|-----------|
| Coupled Osc / Van der Pol | 10.0 (default) | State magnitudes ~ O(1) |
| Bearing-Only 4D | **500.0** m | Steady-state RMSE ~64 m; 500 m marks true loss-of-track |
| Reentry 6D | **5000.0** m | Positions ~ O(1e6) m; 5 km = ~0.1% |

Fixed in v3.3.0 audit. See `run_benchmarks.cpp` l. 547-553 and 619-620.

### 10.5 Seed policy for RNG-driven filters (PKF, RBPKF)

- **PKF** (`particle_filter.hpp` l. 74-80): derives a 64-bit seed from two `std::random_device` draws; overridable via `set_seed(uint64_t)`. Both serial `rng_` and per-thread OpenMP RNGs are re-seeded from the same source in `seed_thread_rngs()`.
- **RBPKF** (`RbpfConfig::seed`, default `123456789`): explicit `unsigned long long` field. `initialize()` **reseeds** so a re-run reproduces (`rbpf_core.hpp` l. 79-80).
- **UKF/SRUKF/EKF benchmarks**: `std::mt19937 gen(42)` in `EKF/main.cpp`, `UKF/main.cpp`, `UKF/main_srukf.cpp`. The benchmark trajectory generator uses per-problem seeds (42-45).
- **PKF** internally uses `std::mt19937_64` (64-bit); **UKF/EKF/RBPF** benchmark drivers use `std::mt19937` (32-bit).

---

## 11. Numerical Regularisation Constants

Reference values that appear across the codebase — quote these if a new filter needs to match the existing safety chain:

| Constant | Purpose | Where |
|----------|---------|-------|
| `1e-6f` | Base Cholesky jitter floor (added to P before retry) | `SigmaPoints.h` l. 76, `SRUKF.h` l. 111, `UKF.h` l. 157 |
| `1e-8f` | Relative-trace multiplier: `jitter = max(1e-6, 1e-8·trace(P)/NX)` | `SRUKF.h` l. 111, `UKF.h` l. 157 |
| `1e-10f` | Minimum diagonal after LDLT reconstruction; also `S(k,k)` regularisation threshold in `cholupdate*` | `SigmaPoints.h` l. 87, `SRUKF.h` l. 203, 565 |
| `1e-4f` | Symmetry-check tolerance: `‖P0 − P0ᵀ‖ / ‖P0‖ ≤ 1e-4` in `SRUKF::initialize()` | `SRUKF.h` l. 93 |
| `25.0f` | Default chi-squared innovation-gate threshold (SRUKF) | `SRUKF.h` l. 556 |
| `1e-30f` | Log-det clamp in RBPF's LDLT diagonal-product path | v3.4.2 changelog; `RBPKF/include/rbpf/rbpf_core.hpp` |
| `100e3, 10e3` | Reentry P0 block scales (position m², velocity (m/s)²) | `run_benchmarks.cpp` l. 614-615 |
| `n_lambda < 0.5f` | Guard for degenerate sigma-point spread; triggers kappa reset | `SigmaPoints.h` l. 50 |
| `1e8f` | NEES condition-number cutoff: skip a step with `max(diag)/min(diag) > 1e8` | `BenchmarkRunner.h` l. 239, 263 |

**Do not silently change these**. Each was tuned or added in response to a specific numerical failure; the audit trail is in `AUDIT_2026-07-08.md`, `README.md` "Issues #1-#7", and the DEVELOPMENT_NOTES.md changelog.

---

## 12. Notation Consistency Assessment

### 12.1 Consistent usage (safe to rely on)

- `x` = state everywhere, `y` = measurement everywhere, `u` = control everywhere, `t` (or `t_k`) = time.
- `Q(t)`, `R(t)` = process/measurement noise covariance functions on all base classes.
- `P` = state covariance, `S` (SRUKF) = its Cholesky factor — never overloaded within a single class.
- Sigma-point weights: `Wm` (mean), `Wc` (covariance).
- Angular wrap: always to [−π, π], always via `std::remainder`.

### 12.2 Symbol overloading and conflicts to watch for

- **`S`** is overloaded across contexts:
  - `S_` member of SRUKF: Cholesky factor of state covariance (`P = S·Sᵀ`)
  - `S` local variable in UKF/EKF update: **innovation** covariance (`S = Pyy + R`)
  - `S_yy` local variable in SRUKF update: Cholesky factor of innovation covariance
  
  Downstream code MUST look at scope. When speaking generally, prefer "square root S", "innovation covariance S_yy", "innovation covariance S" (EKF/UKF).

- **`f`** is overloaded across argument orders — see §1.4.

- **`R`** is: the measurement noise covariance function `R(t)` (member function on models); a local matrix inside `predict()`/`update()`; the QR-decomposition R factor in SRUKF (l. 232); and the Earth radius parameter `R0` in `ReentryVehicle`. Scope disambiguates each usage.

### 12.3 No detected drift vs `.gpd/CONVENTIONS.md`

`.gpd/CONVENTIONS.md` does not exist at time of analysis (no notation-coordinator has run). This document is therefore the sole record of what conventions the project uses. If a notation-coordinator lock is created later, cross-check this document against it and flag any divergence.

---

## 13. Dimensional-Analysis Notes on Selected Equations

Applying the dimensional-consistency check for the key filter equations (in the natural SI units of the Reentry problem, since that is the fully-dimensional case):

- **State covariance update `P = F·P·Fᵀ + Q`**:
  - `[F]` dimensionless (Jacobian of position/velocity map)
  - `[P]` and `[Q]` share dimensions (mixed: position blocks m², velocity blocks (m/s)², cross m²/s)
  - Verified consistent: RHS both terms are covariance matrices of the same physical state ✓

- **Kalman gain `K = P·Hᵀ·S⁻¹`**:
  - `[H]` maps state to observation: for reentry radar it maps m to {m, rad, rad}, so `[H]` has mixed rows (1, 1/m, 1/m)
  - `[S] = [H·P·Hᵀ + R]` has the units of the observation covariance
  - `[K] = [P·Hᵀ·S⁻¹]` has units mapping observation to state — consistent ✓

- **NEES `= eᵀ·P⁻¹·e`** where `e = x_true − x_est`:
  - `[e]` state error; `[P⁻¹]` inverse-state covariance
  - Result is dimensionless (chi-squared statistic) ✓

- **Merwe `lambda = α²(n + κ) − n`**:
  - `n`, `κ`, `λ` all dimensionless ✓
  - `sqrt(n + λ) · L.col(i)` gives a displacement in state units (since `L` has state units) ✓

No dimensional inconsistencies detected in the core filter equations. The dimensionless benchmarks (Van der Pol, Coupled Oscillators, Nonlinear Oscillator) trivially pass; the dimensional ones (Reentry, Drag Ball, Bearing-Only) have been verified end-to-end.

---

## 14. Approximations Made (in the filter formulations)

Focus is on APPROXIMATIONS, not on approximation-free choices. The catalog below is prescriptive: use the same approximations if extending the code; do not silently introduce different ones.

| Approximation | Where | Expansion parameter | Grade | First-neglected order |
|---------------|-------|--------------------|-------|-----------------------|
| **First-order Taylor linearization** (EKF) | `EKF/include/EKF.h`; `NonlinearOscillator::F/H`, `BallTossModel` | `‖x_true − x_hat‖` — no explicit control | **Weak** (no error estimate published; correctness relies on the linearization neighborhood matching P). Documented as approximate. | Second-order Taylor (curvature) |
| **Unscented transform (Merwe scaled) captures mean/cov to 2nd order for Gaussians, 3rd order for symmetric noise** | UKF/SRUKF | `alpha` scales spread; `beta=2` optimal for Gaussian | **Adequate** — Julier-Uhlmann Ref [3-4] gives the accuracy guarantee | 4th-order moments |
| **RK4 integration of continuous dynamics** | `BenchmarkProblems.h` l. 42-46, 137-141, 208-212, 288-292 | `dt` (0.01 s for oscillators, 0.1 s for tracking) | **Adequate** — 5th-order local truncation | O(dt⁵) local, O(dt⁴) global |
| **Euler integration** for the two non-benchmark models | `NonlinearOscillator::f` l. 37-38, `DragBallModel::f` l. 44-47 | `dt` | **Weak** — no error bound published | O(dt²) local, O(dt) global |
| **Bootstrap particle filter (proposal = prior)** | `PKF/include/particle_filter.hpp` | Weight variance / effective sample size | **Adequate** with systematic resampling | Higher-order optimal proposals available (auxiliary PF, unscented PF) — not implemented |
| **Rao-Blackwellization split** — assumes exact conditional linearity of the linear substate | `RBPKF/include/rbpf/rbpf_core.hpp` | Conditional linearity | **Strong** if the model is genuinely conditionally linear (e.g. CTRV) | Zero — this is exact given the assumption |
| **Circular mean for angular states** (SRUKF only) | `SRUKF.h` l. 173-183 | Concentration parameter of the state distribution | **Adequate** for unimodal wrapped-Gaussian; degrades near uniform | Distribution becomes bimodal |
| **Innovation gating scales rather than rejects by default** | SRUKF `update()`, l. 431-451 | Chi-squared gate value | **Adequate** — Joseph consistency preserved via `sqrt(2s − s²)` factor | Only extreme outliers fully rejected when `setRejectOutliers(true)` |
| **Fixed-lag smoothing lag chosen empirically** | Benchmarks: lag=20-30 | Autocorrelation time of the state | **Weak** — no automated lag selection; hand-tuned per problem | Truncation of information from measurements > lag steps in the future |

**If any of these fails, the effect is documented in `CONCERNS.md`.**

---

_This document is a REPORT of conventions present in the codebase, not a prescription. If a notation-coordinator run creates `.gpd/CONVENTIONS.md`, that file supersedes this one for prescriptive use. Convention conflicts against a future lock must be recorded in `CONCERNS.md`._
