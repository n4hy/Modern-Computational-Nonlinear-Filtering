# VALIDATION — Benchmarks, Consistency Checks, Test Coverage

> Focus: **methodology**. What has actually been checked, how, and where the
> evidence lives. Every claim points at a code path or a status doc.

## 1. Known-Limit Checks

### 1.1 Linear-Gaussian limit (UKF → KF, EKF → KF)
- Not exercised as a dedicated CTest. Implicit in the fact that the same
  `Common/include/FilterMath.h` dispatch layer serves EKF, UKF, and SRUKF —
  bit-identical Eigen fallback verified across NEON and Eigen-only builds
  (`DEVELOPMENT_NOTES.md` §Backend portability, RMSE/NEES identical to reported
  precision on 8 of 10 rows).
- **Gap:** no explicit "reduce to KF on a linear model" test is registered.

### 1.2 Gaussian noise limit
- All four benchmarks assume additive Gaussian process/measurement noise per
  `Common/include/StateSpaceModel.h`. Non-Gaussian regimes are not part of the
  registered suite.

### 1.3 Small-noise / deterministic limit
- Not exercised. The smoother regression uses `seed 12345` for reproducibility
  (`DEVELOPMENT_NOTES.md` §v3.4.1 "Smoother test hardening") with fixed noise
  amplitude.

---

## 2. Benchmark Comparisons

### 2.1 Current suite (v3.4.x, backend-independent)

Values from `benchmark_results.csv` and `DEVELOPMENT_NOTES.md` "Backend
portability". `RMSE_Position` / `RMSE_Velocity` columns are `0` because they were
**stripped in v3.4.1** for being unassigned (`DEVELOPMENT_NOTES.md` §"Stopped
publishing fabricated metrics") — the retained `RMSE_Overall` is the honest
figure.

| Problem | Filter | RMSE | Median NEES | % in 95% bounds | Divergences | Smoothed RMSE |
|---|---|---:|---:|---:|---:|---:|
| CoupledOscillators10D | UKF | 1.45665 | 9.89 | 94.49% | 0 | — |
| CoupledOscillators10D | SRUKF | 1.45666 | 9.89 | 94.49% | 0 | — |
| CoupledOscillators10D | UKF+Smoother | — | — | — | — | 1.14767 |
| CoupledOscillators10D | SRUKF+Smoother | — | — | — | — | 1.14767 |
| VanDerPol2D | UKF | 0.468053 | 1.14 | 95.94% | 0 | — |
| VanDerPol2D | SRUKF | 0.46626 | 1.14 | 96.00% | 0 | — |
| VanDerPol2D | SRUKF+Smoother | — | — | — | — | 0.429705 |
| BearingOnly4D | UKF | 63.8086 | 3.77 | 99.63% | 176 (artifact) | — |
| BearingOnly4D | SRUKF | 64.1727 | 3.77 | 99.63% | 175 (artifact) | — |
| BearingOnly4D | SRUKF+Smoother | — | — | — | — | 52.0285 |
| ReentryVehicle6D | UKF | 369.115 | 5.00 | 95.93% | 0 | — |
| ReentryVehicle6D | SRUKF | 369.165 | 4.99 | 95.56% | 0 | — |
| ReentryVehicle6D | SRUKF+Smoother | — | — | — | — | 236.858 |

BearingOnly4D "176/175 divergence" figures in the CSV come from a stale
`count_divergences()` gate threshold of 10.0 m against a ~64 m error scale. The
v3.3.0 problem-scaled 500 m threshold reduces the count to 0 — the filter is
statistically consistent throughout (`DEVELOPMENT_NOTES.md` §v3.3.0).

### 2.2 Smoother regression (v3.4.0, CTest only)

Moving target, mildly nonlinear observation, off prior. Ceilings sit ~20% above
the clean deterministic values at seed 12345:

| Filter | filtered → smoothed → iterated RMSE | interior cov trace | Ceiling (seed 12345) |
|---|---|---|---|
| EKF | 1.21 → 0.72 → 0.64 m | 0.86 → 0.13 | RMSE 1.207/0.717 |
| UKF | 1.22 → 0.68 → 0.66 m | 0.89 → 0.13 | RMSE 1.218/0.680 |
| SRUKF | 1.22 → 0.68 → 0.66 m | 0.89 → 0.13 | RMSE 1.218/0.680 |

Registered as CTest cases `SRUKF_Smoother`, `UKF_Smoother`, `EKF_Smoother`.
Assertions: absolute RMSE ceilings on **both** filtered and smoothed, min
eigenvalue ≥ −1e-4 on smoothed covariance at every timestep, two-sided interior
trace bound. Injected defects (× 1.05, × 1.20, × 0.80) all fail with exit=1
(`DEVELOPMENT_NOTES.md` §v3.4.1 "Smoother test hardening").

### 2.3 Backend portability (aarch64 Raspberry Pi 5, no CUDA, 2026-07-14)

Same host, two builds compared:

| Problem / filter | NEON | Eigen-only | Δ |
|---|---:|---:|---|
| CoupledOsc 10D (UKF & SRUKF) | 1.45666 | 1.45666 | identical |
| CoupledOsc 10D smoothed | 1.14767 | 1.14767 | identical |
| VanDerPol 2D (UKF) | 0.468053 | 0.468053 | identical |
| VanDerPol 2D (SRUKF) | 0.46626 | 0.46626 | identical |
| VanDerPol 2D smoothed | 0.429705 | 0.429705 | identical |
| Bearing-Only 4D (UKF) | 63.8082 | 63.8084 | 0.0003% |
| Bearing-Only 4D (SRUKF) | 64.1723 | 64.1723 | identical |
| Reentry 6D (UKF) | 369.043 | 369.117 | 0.02% |
| Reentry 6D (SRUKF) | 369.192 | 369.189 | 0.0008% |
| Reentry 6D smoothed | 236.785 | 236.817 | 0.014% |

Residual deltas ≤ 0.02% are float reassociation, not behavioral. CTest:
**30/30 NEON, 29/29 Eigen-only** (delta is `test_neon_fp16`, a dependency test).

### 2.4 Before/After validation (July 8, 2026, x86_64 + RTX 5070 Ti)

`DEVELOPMENT_NOTES.md` §"Before/After Validation Record". `Before` = commit
`7609df4`; `After` = commit `2c7dccf`. RMSE/NEES/divergence figures reproduce on
any host; timings do not.

- CTest: 24/24 unchanged before/after, also under `OMP_NUM_THREADS=24`.
- RMSE identical to ≥ 4 sig figs on every problem.
- Bearing-Only divergences: **176/175 → 0** (metric fix, not filter behavior).
- RBPF OpenMP TSan worker races: **~24 reports → 0** (Phase 3 fix).
- SRUKF rejected-outlier trace behavior: 5.2547 → 5.2367 (before, false
  certainty) vs 5.2547 → 5.2747 (after, +Q correctly reflected).
- UKF 10D fixed-size dispatch: 38.27 ms → 36.91 ms (−3.6%), RMSE bit-identical.

---

## 3. Consistency Checks

### 3.1 Covariance positive-definiteness / symmetry
- SRUKF update uses safe `cholupdate_downdate` + full-covariance fallback
  (`AUDIT_2026-07-08.md` §Phase 3, item 2).
- UKF update uses Joseph symmetric form (`P − K·Pxyᵀ − Pxy·Kᵀ + K·S·Kᵀ`) with
  LLT/LDLT recovery ladder and **relative** jitter
  `max(1e-6, 1e-8·trace(P)/NX)` (`UKF/include/UKF.h`, `DEVELOPMENT_NOTES.md`
  §v3.4.2).
- SRUKF `initialize()` throws `std::runtime_error` on NaN / Inf / asymmetric /
  non-PSD `P0` including condition number in the message
  (`UKF/include/SRUKF.h`, v3.4.2).
- SigmaPoints diagonal fallback on covariance decomposition failure
  (`UKF/include/SigmaPoints.h:92`) — logs to stderr, filter continues.
- RBPKF `normalize_weights` guarded against NaN/Inf; `get_effective_sample_size`
  guarded against divide-by-zero (`AUDIT_2026-07-08.md` §Phase 3, items 4–5).
- RBPKF log-det computed via LDLT diagonal-product with `1e-30f` clamp
  (`RBPKF/include/rbpf/rbpf_core.hpp`, v3.4.2) — direct `det(S)` on float32
  underflows silently.
- **Smoother min-eigenvalue guard** ≥ −1e-4 asserted at **every timestep** in
  the v3.4.0 CTest cases (registered).
- Smoother `sqrt_spd` (`UKF/include/SRUKFSmoother.h:113-122`) silently returns
  `Identity()` on Cholesky failure — see `CONCERNS.md` §1.1 (HIGH open concern).

### 3.2 Innovation whiteness / NIS
- SRUKF exposes `getLastNIS()` (Normalized Innovation Squared) and a χ² gate
  (`UKF/include/SRUKF.h`, v3.3.0 R33). Default threshold (since 2026-11-18): the
  χ²_NY upper quantile at tail probability 1e-4 (`Common/include/ChiSquared.h`),
  previously a fixed `25.0f`. Set via `setInnovationGateTailProbability()` or
  `setInnovationGateChi2()`; read via `getInnovationGateChi2()`.
- The first gate firing in the process logs a single line to `std::clog`
  (per process since 2026-11-16). Per-filter count via `getGatedCount()`.
- Outlier policy toggle `setRejectOutliers()` — reject vs down-scale.
- Under scaling by `s < 1`, the covariance downdate is scaled by
  `sqrt(2s − s²)` (Joseph-consistent for gain `sK`) so the reported covariance
  is not over-confident (`AUDIT_2026-07-08.md` §Phase 3 SRUKF gate/covariance).

### 3.3 Angular observation wrap
- SRUKF wraps angular observation innovations to [−π, π] at
  `UKF/include/SRUKF.h:383,417` (v3.3.0 R32).
- Coverage restricted to SRUKF; UKF, EKF, PKF, RBPKF, and every smoother lack
  it. Default `StateSpaceModel::isAngularObservation = false` — no default
  benchmark exercises the path (see `CONCERNS.md` §1.2).
- CTest `SRUKF_AngularWrap` verifies the path independently
  (`UKF/tests/test_srukf_angular_wrap.cpp`).

### 3.4 Reproducibility (seed handling)
- RBPF and PKF `OpenMP` RNG previously used `thread_local std::random_device`,
  ignoring `config.seed`. Replaced by per-thread `mt19937_64` seeded via
  SplitMix64 mix of the base seed with `schedule(static)` — reproducibility
  confirmed at 4 threads (`AUDIT_2026-07-08.md` §Phase-8 OpenMP seed fix,
  `DEVELOPMENT_NOTES.md` §v3.2.1).
- Smoother CTest is deterministic at `seed 12345`; measured drift between
  `-O1` and `-O3 -march=native` is zero (`DEVELOPMENT_NOTES.md` §v3.4.1).

### 3.5 Metric discipline
- v3.4.1 stripped `RMSE_Position` / `RMSE_Velocity` (and smoothed twins) from
  the metrics struct + CSV — they were being emitted as `0.0` for every row and
  published as data. `compute_rmse_indices()` retained but uncalled.
- v3.4.1 fixed `compute_convergence_time()` — now returns NaN + prints
  `did not converge` + writes empty CSV cell when the filter never converges
  (was returning `times.back()`, a sentinel indistinguishable from
  last-step convergence for 6 of 13 rows).
- v3.3.0 fixed `count_divergences()` — problem-scaled 500 m threshold for the
  bearings-only ~64 m error scale (was fixed 10.0 m).

---

## 4. Regression Test Suite

### 4.1 Registered CTest cases

12 project-registered CTests (root `CMakeLists.txt:569-586`) + dependency tests
from OptMathKernels. Current counts:

- **aarch64 RPi5, OptMath v0.6.3, no CUDA**: 30/30 NEON, 29/29 Eigen-only.
- **x86_64 + RTX 5070 Ti / CUDA 13.1**: 25/25 (July 8, 2026 snapshot; grew to
  28/28 by v3.4.2 with the four new regression tests).

Registered tests (project side):

| CTest name | Source | Purpose |
|---|---|---|
| `EKF_Test` | `EKF/tests/test_ekf.cpp` | EKF baseline |
| `EKF_Smoother` | `EKF/tests/test_ekf_smoother.cpp` | v3.4.0 full-interval RTS |
| `UKF_Test` | `UKF/tests/test_ukf.cpp` | UKF baseline |
| `UKF_Smoother` | `UKF/tests/test_ukf_smoother.cpp` | v3.4.0 full-interval URTSS |
| `SRUKF_Test` | `UKF/tests/test_srukf.cpp` | SRUKF baseline |
| `SRUKF_AngularWrap` | `UKF/tests/test_srukf_angular_wrap.cpp` | v3.3.0 R32 wrap coverage |
| `SRUKF_Smoother` | `UKF/tests/test_srukf_smoother.cpp` | v3.4.0 full-interval SR URTSS |
| `PKF_Test` | `PKF/tests/test_particle.cpp` | Particle filter baseline |
| `PKF_Example` | `PKF/examples/pkf_example.cpp` | End-to-end demo |
| `RBPF_Basic` | `RBPKF/tests/test_rbpf_basic.cpp` | Rao-Blackwellized baseline |
| `RBPF_CTRV` | `RBPKF/examples/example_rbpf_ctrv.cpp` | RBPF CTRV benchmark |
| `Benchmarks` | `Benchmarks/src/run_benchmarks.cpp` | Aggregate benchmark runner |

Plus the v3.4.2 audit-remediation regressions:

| CTest name | Source | Purpose |
|---|---|---|
| `test_ukf_numerical` | `UKF/tests/test_ukf_numerical.cpp` | rank-deficient `P0` + covariance contraction |
| `test_srukf_initialize` | `UKF/tests/test_srukf_initialize.cpp` | throw on NaN / Inf / asymmetric / non-PSD `P0` |
| `test_particle_const` | `PKF/tests/test_particle_const.cpp` | SFINAE `static_assert` on const-ness of `get_mean` / `get_covariance` / `compute_mean_cpu` |
| `test_rbpf_logdet` | `RBPKF/tests/test_rbpf_logdet.cpp` | LDLT log-det vs SVD across cond ∈ {1e2, 1e6, 1e12} |

### 4.2 Historical demo-program problem
The July 14 broader audit found **6 of 12 CTest registrations were demo programs
that could not fail** (`AUDIT_2026-07-08.md` header). `PKF_Example`, `RBPF_CTRV`,
`Benchmarks`, and similar are still in the list — treat their "PASS" as
"program ran cleanly," not as a numeric assertion. Real numeric gates live in
the CTest cases with `test_*` prefixes plus the smoother-test hardening from
v3.4.1.

### 4.3 Sanitizer coverage
- Root `CMakeLists.txt` gained `nlf_add_warning_flags(<target>)` in v3.4.2:
  `-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter` + GCC-only
  `-Wno-stringop-overread` at compile and link (works around Eigen AVX-512 LTO
  false positive).
- Root `CMakeLists.txt` gained `NLF_ENABLE_SANITIZERS`: when `ON` and build type
  is `Debug` or `RelWithDebInfo`, ASan + UBSan are attached to C++ TUs (CUDA
  TUs skipped — nvcc does not consistently forward `-fsanitize=…`).
- `.github/workflows/ci.yml` — two-job matrix: Release + ctest;
  RelWithDebInfo + sanitizers.
- **ThreadSanitizer** was used offline as a definitive check on the RBPF race
  fix (`DEVELOPMENT_NOTES.md` §Fix #1 RBPF OpenMP race): 24 worker-vs-worker
  reports → 0. Residual reports are known GCC-libgomp barrier false positives.

### 4.4 Warning discipline
- Uniform warning flag application via `nlf_add_warning_flags()` in root
  `CMakeLists.txt`. Applied to every executable across EKF / UKF / PKF / RBPKF /
  Benchmarks.
- `-Werror` **not** enforced globally (deliberate — Eigen/AVX-512 LTO diagnostic
  quirks would break the build on some toolchains).

---

## 5. CI Validation

`.github/workflows/ci.yml` — two-job matrix (from `DEVELOPMENT_NOTES.md`
§v3.4.2):

1. **Release build + ctest.**
2. **RelWithDebInfo + sanitizers** (ASan + UBSan).

Both jobs pre-clone OptMathKernels **v0.5.15** to
`$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA` and pass
`-DOPTMATH_DIR=$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA`
`-DNLF_BUILD_PYTHON_VENV=OFF` to CMake. Matches the sibling-directory
provisioning contract in the root `CMakeLists.txt`.

- **Concern:** CI pins **v0.5.15** while working-tree runs **v0.6.3**
  (`DEVELOPMENT_NOTES.md:132`). Divergence between CI and developer builds. See
  `CONCERNS.md` §6.
- **Portability matrix suggested but not committed** (`DEVELOPMENT_NOTES.md`
  §v3.4.1 "Recommended follow-up: CI"): x86_64 runner with NEON/SVE2 off, plus
  Eigen-only job (`-DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF
  -DNLF_ENABLE_NATIVE_ARCH=OFF`).

---

## 6. Audit History

Recorded phases, in chronological order:

| Phase | Date | Focus | Locator |
|---|---|---|---|
| Phase 1 | Feb 2026 | Initial SRUKF — 5 critical numerical bugs (sigma-point weights, QR loop count, 1D QR, singular Q, Cholesky div-by-zero); dimension-adaptive parameters; replaced QR-based `S_yy` with direct `P_yy`. | `AUDIT_2026-07-08.md` §Phase 1 |
| Phase 2 | Mar 2026 | Numerical stability, error handling — innovation gating, Eigen aliasing, Monte Carlo convergence, `-ffast-math` disabled, measurement outage recovery. | `AUDIT_2026-07-08.md` §Phase 2 |
| Phase 3 | Mar 31, 2026 | FilterMath dispatch + full optimization — GEMM/Cholesky/Kalman-gain tiered dispatch, safe `cholupdate_downdate`, `allFinite()` gates, RBPKF NaN/div-by-zero guards. | `AUDIT_2026-07-08.md` §Phase 3 |
| Phase 4 | Apr 2026 | Particle filter memory (v3.4.2 completion) | `AUDIT_2026-07-08.md` §Phase 4 |
| Phase 5 | Apr 2026 | Multi-arch build refactor (initial) | `AUDIT_2026-07-08.md` §Phase 5 |
| Phase 6 | Apr 12, 2026 | Correctness fix list (11 items, re-verified 2026-07-14 as still live). | `AUDIT_2026-07-08.md` §Phase 6 (preserved only in this doc) |
| Phase 7 | May 2026 | v3.2.0 — kernel adoption v0.5.15, CUDA 13.x + Blackwell SM 100/120 unblocked, Vulkan discrete-GPU preference. | `DEVELOPMENT_NOTES.md` §v3.2.0 |
| Phase 8 | Jul 2026 | v3.2.1 → v3.4.2 audit-remediation — RBPF OpenMP race, seeded RNG, Joseph UKF/SRUKF forms, fixed-size dispatch, smoother hardening, warning/sanitizer CI. | `DEVELOPMENT_NOTES.md` §v3.2.1 – §v3.4.2 |

Two subsequent audit passes are recorded:

- **2026-07-10** — dependency provisioning refactor: `OPTMATH_RELEASE_TAG`
  mechanism removed (commit `168dafd`); `OPTMATH_DIR` sibling-tree convention
  adopted.
- **2026-07-14** — broader audit finding: 6 of 12 CTest registrations were demo
  programs that could not fail; smoother tests gated on relative ratios only.
  Both fixes landed via v3.4.1 "Smoother test hardening" and the four new
  numeric regressions in v3.4.2.

---

## 7. Coverage Gaps

Deliberately mirrored from `CONCERNS.md` §4 so the validation view is
self-consistent:

- **Angular-wrap fix untested in benchmark suite** — CTest-only.
- **Full-interval RTS smoother not in benchmark suite** — CTest-only.
- **UKF/EKF/PKF/RBPKF wrap coverage** — none.
- **NIS-gate + reject-outliers policy comparison** — no benchmark row.
- **Sigma-point α/β/κ sweep** — undocumented.
- **Particle-count sweep** — none for PKF/RBPKF.
- **High-μ Van der Pol / regime-crossing reentry** — no stress benchmarks.
- **Long-run PSD stress** (10⁴–10⁵ steps) on smoother backward pass — not on
  record.
- **Linear-Gaussian reduction test** — not registered.

---

## 8. Sources

- `benchmark_results.csv` — 14 rows, current tree.
- `DEVELOPMENT_NOTES.md` (734 L) — authoritative live engineering doc. Sections
  cited: Backend portability; Before/After Validation Record; Build
  Verification; Future Work; Changelog v3.2.0–v3.4.2.
- `SRUKF_STATUS.md` (120 L) — Numerical Safety Chain, dimension-adaptive
  parameters, current benchmark table.
- `AUDIT_2026-07-08.md` (352 L) — audit-phase log (superseded but preserved for
  Phase 6 history).
- `COMPARISON_RESULTS.md` (480 L) — historical UKF-vs-SRUKF; performance claims
  **RETRACTED**.
- `README.md` — user-facing entry point; build recipes; CTest layout.
- `.github/workflows/ci.yml` — CI configuration.
- Root `CMakeLists.txt` — CTest registration list, sanitizer/warning flags.
- Test sources: `EKF/tests/`, `UKF/tests/`, `PKF/tests/`, `RBPKF/tests/`,
  `Benchmarks/src/`.

Private contract-work files (per user global `CLAUDE.md`) are excluded.
