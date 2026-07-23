# REFERENCES — Active Anchor Registry

> Focus: **theory**. Anchor registry for the Modern Computational Nonlinear
> Filtering library. Every row carries a stable `Anchor ID` and a concrete
> `Source / Locator`. `Carry Forward To` names workflow stages. When exact
> contract claim/deliverable IDs become known, put them in
> `Contract Subject IDs`, not in `Carry Forward To`.

## Legend

- **Anchor ID** — stable slug used across the research map and future GPD
  artifacts. Do not rename.
- **Source / Locator** — canonical citation, file path, or URL. Every anchor has
  one.
- **Kind** — `paper` / `book` / `thesis` / `benchmark` / `artifact` / `code` /
  `dataset` / `open_question`.
- **Carry Forward To** — workflow stages that should re-consult the anchor.
- **Contract Subject IDs** — reserved. Populated once `project_contract` claims /
  deliverables exist.

---

## A. Foundational Literature

### A.1 Bayesian nonlinear filtering — algorithm foundations

| Anchor ID | Kind | Source / Locator | Carry Forward To | Contract Subject IDs |
|---|---|---|---|---|
| `ref/kalman-1960` | paper | R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, J. Basic Engineering **82**(1) (1960) 35–45. | plan-phase, execute, verify | — |
| `ref/rts-1965` | paper | H. E. Rauch, F. Tung, C. T. Striebel, *Maximum Likelihood Estimates of Linear Dynamic Systems*, AIAA Journal **3**(8) (1965) 1445–1450. | plan-phase (smoothers), verify | — |
| `ref/julier-uhlmann-1997` | paper | S. J. Julier & J. K. Uhlmann, *A New Extension of the Kalman Filter to Nonlinear Systems*, Proc. SPIE AeroSense (1997). | plan-phase (UKF), verify (sigma-point choice) | — |
| `ref/wan-vandermerwe-2000` | paper | E. A. Wan & R. van der Merwe, *The Unscented Kalman Filter for Nonlinear Estimation*, IEEE AS-SPCC (2000). | plan-phase (UKF, scaled UT), verify (α/β/κ) | — |
| `ref/vandermerwe-thesis-2004` | thesis | R. van der Merwe, *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models*, PhD thesis, OGI/OHSU, 2004. | plan-phase (SRUKF, Cholesky updates), verify | — |
| `ref/sarkka-book-2013` | book | S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge Univ. Press, 2013 (open PDF). | plan-phase (any filter/smoother), verify, write-paper | — |
| `ref/sarkka-svensson-2023` | book | S. Särkkä & L. Svensson, *Bayesian Filtering and Smoothing*, 2nd ed., 2023. | plan-phase (iterated smoothers), write-paper | — |
| `ref/doucet-2001-smc` | book | A. Doucet, N. de Freitas, N. Gordon (eds.), *Sequential Monte Carlo Methods in Practice*, Springer, 2001. | plan-phase (PKF, resampling), verify | — |
| `ref/doucet-2009-sms` | paper | A. Doucet & A. M. Johansen, *A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later*, Oxford Handbook of Nonlinear Filtering (2009). | plan-phase (PKF, RBPKF), verify | — |
| `ref/chen-liu-2000-rbpf` | paper | R. Chen & J. S. Liu, *Mixture Kalman Filters*, J. Royal Stat. Soc. B **62**(3) (2000) 493–508. | plan-phase (RBPKF), verify | — |
| `ref/schoen-gustafsson-2005-rbpf` | paper | T. Schön, F. Gustafsson, P. Nordlund, *Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models*, IEEE T-SP **53**(7) (2005) 2279–2289. | plan-phase (RBPKF marginalization), verify | — |
| `ref/kotecha-djuric-2003-gpf` | paper | J. Kotecha & P. M. Djurić, *Gaussian Particle Filtering*, IEEE T-SP **51**(10) (2003) 2592–2601. | plan-phase (PKF Gaussian mixture), verify | — |
| `ref/sarkka-2008-urtss` | paper | S. Särkkä, *Unscented Rauch–Tung–Striebel Smoother*, IEEE T-AC **53**(3) (2008) 845–849. | plan-phase (URTSS), verify | — |
| `ref/bell-cathey-1993-iekf` | paper | B. M. Bell & F. W. Cathey, *The Iterated Kalman Filter Update as a Gauss–Newton Method*, IEEE T-AC **38**(2) (1993). | plan-phase (iterated EKF/SRUKF smoother), verify | — |
| `ref/bierman-1977-book` | book | G. J. Bierman, *Factorization Methods for Discrete Sequential Estimation*, Academic Press, 1977. | plan-phase (square-root filtering), verify | — |
| `ref/joseph-form` | note | Bucy–Joseph symmetric covariance-update form, standard result; cross-check any Kalman textbook (Bar-Shalom §5, Simon 2006, Grewal-Andrews 2015). | plan-phase (UKF/SRUKF/EKF update), verify | — |
| `ref/simon-2006-book` | book | D. Simon, *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*, Wiley, 2006. | onboarding, verify, write-paper | — |

### A.2 Benchmark-physics literature

| Anchor ID | Kind | Source / Locator | Carry Forward To | Contract Subject IDs |
|---|---|---|---|---|
| `ref/vanderpol-1927` | paper | B. van der Pol, *Forced Oscillations in a Circuit with Non-Linear Resistance*, Phil. Mag. **3** (1927) 65–80. | verify (VdP benchmark), write-paper | — |
| `ref/aidala-hammel-1983-bearings` | paper | V. J. Aidala & S. E. Hammel, *Utilization of Modified Polar Coordinates for Bearings-Only Tracking*, IEEE T-AC **28**(3) (1983) 283–294. | plan-phase (bearings-only), verify | — |
| `ref/nardone-aidala-1981-observability` | paper | S. C. Nardone & V. J. Aidala, *Observability Criteria for Bearings-Only Target Motion Analysis*, IEEE T-AES **17**(2) (1981). | verify (bearings-only observability), write-paper | — |
| `ref/costa-1994-reentry` | paper | P. J. Costa, *Adaptive Model Architecture and Extended Kalman-Bucy Filters*, IEEE T-AES **30**(2) (1994) 525–533. | plan-phase (reentry), verify | — |
| `ref/austin-1997-reentry` | paper | J. Austin & C. Leondes, *Statistically Linearized Estimation of Reentry Trajectories*, IEEE T-AES **17**(1) (1981). | plan-phase (reentry), verify | — |
| `ref/julier-uhlmann-2004-reentry` | paper | S. J. Julier & J. K. Uhlmann, *Unscented Filtering and Nonlinear Estimation*, Proc. IEEE **92**(3) (2004) — reentry canonical benchmark. | verify (reentry RMSE parity), write-paper | — |
| `ref/lorenz-1963` | paper | E. N. Lorenz, *Deterministic Nonperiodic Flow*, J. Atmos. Sci. **20** (1963) 130–141. | verify (Lorenz-63 sanity if reused) | — |
| `ref/lorenz-1996` | paper | E. N. Lorenz, *Predictability: A Problem Partly Solved*, ECMWF Seminar (1996) — Lorenz-96 model. | plan-phase (only if Lorenz96 default is re-enabled) | — |
| `ref/kim-park-2013-ctrv` | paper | Standard CTRV motion model — see Y. Bar-Shalom et al. *Estimation with Applications to Tracking and Navigation*, Wiley 2001, §11 for the constant-turn variants used by RBPKF. | plan-phase (RBPKF CTRV example), verify | — |

### A.3 Numerical / infrastructure references

| Anchor ID | Kind | Source / Locator | Carry Forward To | Contract Subject IDs |
|---|---|---|---|---|
| `ref/eigen-3-4` | code | Eigen 3.4.0, https://eigen.tuxfamily.org — dense linear algebra used throughout. | plan-phase, execute | — |
| `ref/optmath-kernels` | code | `OptimizedKernelsForRaspberryPi5_NvidiaCUDA` sibling checkout (`OPTMATH_DIR`), currently working-tree v0.6.3 per `DEVELOPMENT_NOTES.md:132`. | plan-phase (kernel dispatch), execute | — |
| `ref/optmath-cusolver` | code | OptMathKernels v0.5.10+ ships cuSOLVER Cholesky / triangular solve — **not yet wired into `Common/include/FilterMath.h` Cholesky dispatch** (`DEVELOPMENT_NOTES.md` §Future Work). | plan-phase (GPU Cholesky), execute | — |

---

## B. Benchmark Values (backend-independent)

Anchor rows below are decisive numeric baselines; use them as regression targets
in any future verify step. Values are from
`benchmark_results.csv` (seed 12345, current tree) and cross-check against the
"Backend portability" and "Before/After Validation" tables in
`DEVELOPMENT_NOTES.md`. RMSE / NEES reproduce across hosts; timings do not.

| Anchor ID | Filter | Problem | RMSE | Median NEES | % in 95% bounds | Divergences | Locator |
|---|---|---|---|---|---|---|---|
| `bench/coupled-osc-10d/ukf` | UKF | CoupledOscillators10D | 1.45665 | 9.892 | 94.49% | 0 | `benchmark_results.csv` row 2 |
| `bench/coupled-osc-10d/srukf` | SRUKF | CoupledOscillators10D | 1.45666 | 9.892 | 94.49% | 0 | `benchmark_results.csv` row 3 |
| `bench/coupled-osc-10d/ukf-smooth` | UKF+Smoother | CoupledOscillators10D | 1.14767 (smoothed) | — | — | 0 | `benchmark_results.csv` row 4 |
| `bench/vanderpol-2d/ukf` | UKF | VanDerPol2D | 0.468053 | 1.136 | 95.94% | 0 | `benchmark_results.csv` row 6 |
| `bench/vanderpol-2d/srukf` | SRUKF | VanDerPol2D | 0.46626 | 1.137 | 96.00% | 0 | `benchmark_results.csv` row 7 |
| `bench/vanderpol-2d/srukf-smooth` | SRUKF+Smoother | VanDerPol2D | 0.429705 (smoothed) | — | — | 0 | `benchmark_results.csv` row 8 |
| `bench/bearing-only-4d/ukf` | UKF | BearingOnly4D | 63.8086 | 3.772 | 99.63% | 176 (metric artifact) | `benchmark_results.csv` row 9; **note: divergence count is a pre-fix threshold artifact** — see `DEVELOPMENT_NOTES.md` §v3.3.0 Bearing-Only fix |
| `bench/bearing-only-4d/srukf` | SRUKF | BearingOnly4D | 64.1727 | 3.774 | 99.63% | 175 (metric artifact) | `benchmark_results.csv` row 10 |
| `bench/bearing-only-4d/srukf-smooth` | SRUKF+Smoother | BearingOnly4D | 52.0285 (smoothed) | — | — | — | `benchmark_results.csv` row 11 |
| `bench/reentry-6d/ukf` | UKF | ReentryVehicle6D | 369.115 | 4.996 | 95.93% | 0 | `benchmark_results.csv` row 12 |
| `bench/reentry-6d/srukf` | SRUKF | ReentryVehicle6D | 369.165 | 4.989 | 95.56% | 0 | `benchmark_results.csv` row 13 |
| `bench/reentry-6d/srukf-smooth` | SRUKF+Smoother | ReentryVehicle6D | 236.858 (smoothed) | — | — | 0 | `benchmark_results.csv` row 14 |
| `bench/smoother-regression/srukf` | SRUKF-Smoother CTest | moving-target, mildly nonlinear obs | filtered 1.22 → smoothed 0.68 → iterated 0.66 m | interior cov trace 0.89 → 0.13 | — | — | `DEVELOPMENT_NOTES.md` §v3.4.0 |
| `bench/smoother-regression/ukf`   | UKF-Smoother  CTest | same | filtered 1.22 → 0.68 → 0.66 m | interior cov trace 0.89 → 0.13 | — | — | `DEVELOPMENT_NOTES.md` §v3.4.0 |
| `bench/smoother-regression/ekf`   | EKF-Smoother  CTest | same | filtered 1.21 → 0.72 → 0.64 m | interior cov trace 0.86 → 0.13 | — | — | `DEVELOPMENT_NOTES.md` §v3.4.0 |
| `bench/backend-portability/rpi5` | (portability) | all four | RMSE/NEES identical to ≤ 0.02% between NEON and Eigen-only on aarch64 RPi5 | — | — | 30/30 CTest NEON; 29/29 Eigen-only | `DEVELOPMENT_NOTES.md` §Backend portability |
| `bench/before-after/rbpf-race` | (ThreadSanitizer) | RBPF | 24 worker-vs-worker TSan reports → 0 after Phase 3 fix | — | — | — | `DEVELOPMENT_NOTES.md` §Fix #1 |
| `bench/before-after/srukf-gate` | (deterministic harness) | SRUKF gated covariance | trace(P) 5.2547 → 5.2367 (before, false certainty) vs 5.2547 → 5.2747 (after, +Q) | — | — | — | `DEVELOPMENT_NOTES.md` §Fix #2 |

---

## C. In-Repo Prior Artifacts

Every prior output the mapper found. All are ground-truthed against the current
source tree; consult these instead of re-deriving.

| Anchor ID | Kind | Path | Note |
|---|---|---|---|
| `art/benchmark-csv` | dataset | `benchmark_results.csv` | Aggregate benchmark table (14 rows including smoothers). Backend-independent RMSE/NEES/div-count. |
| `art/csv/vanderpol-ukf` | dataset | `vanderpol_ukf.csv`, `vanderpol_srukf.csv`, `vanderpol_srukf_smooth.csv` | Per-step trajectories. |
| `art/csv/coupled-osc` | dataset | `coupled_osc_ukf.csv`, `coupled_osc_srukf.csv`, `coupled_osc_ukf_smooth.csv`, `coupled_osc_srukf_smooth.csv` | 10-D trajectories. |
| `art/csv/bearing` | dataset | `bearing_ukf.csv`, `bearing_srukf.csv`, `bearing_srukf_smooth.csv` | 4-D bearings-only trajectories. |
| `art/csv/reentry` | dataset | `reentry_ukf.csv`, `reentry_srukf.csv`, `reentry_srukf_smooth.csv` | 6-D reentry trajectories. |
| `art/audit-2026-07-08` | doc | `AUDIT_2026-07-08.md` (352 L) | Historical audit, superseded — retained for Phase 6 fix history. |
| `art/comparison-results` | doc | `COMPARISON_RESULTS.md` (480 L) | Historical UKF-vs-SRUKF snapshot; performance claims **RETRACTED**. |
| `art/srukf-status` | doc | `SRUKF_STATUS.md` (120 L) | v3.4.0 SRUKF live status. |
| `art/dev-notes` | doc | `DEVELOPMENT_NOTES.md` (734 L) | Authoritative live engineering log (per audit header). |
| `art/readme` | doc | `README.md` (977 L) | User-facing entry point. |
| `art/formalism-map` | doc | `.gpd/research-map/FORMALISM.md` | This mapping run — theoretical framework. |
| `art/architecture-map` | doc | `.gpd/research-map/ARCHITECTURE.md` | This mapping run — computational pipeline. |
| `art/structure-map` | doc | `.gpd/research-map/STRUCTURE.md` | This mapping run — file layout. |
| `art/conventions-map` | doc | `.gpd/research-map/CONVENTIONS.md` | This mapping run — notation / unit conventions. |
| `art/concerns-map` | doc | `.gpd/research-map/CONCERNS.md` | This mapping run — open concerns registry. |
| `art/ci-workflow` | code | `.github/workflows/ci.yml` | Two-job matrix: Release + RelWithDebInfo/sanitizers. |
| `art/bootstrap` | code | `bootstrap.sh` | One-shot configure/build that provisions the OptMath sibling checkout. |

---

## D. Open Questions and Required Carry-Forward Actions

Curated from `SRUKF_STATUS.md`, `DEVELOPMENT_NOTES.md`, `AUDIT_2026-07-08.md`,
and the mapper's own findings.

| Anchor ID | Question / Action | Source | Carry Forward To | Contract Subject IDs |
|---|---|---|---|---|
| `oq/cusolver-cholesky` | Wire cuSOLVER Cholesky into `Common/include/FilterMath.h` dispatch. Available upstream since OptMathKernels v0.5.10 but not consumed. | `DEVELOPMENT_NOTES.md` §Future Work | plan-phase, execute | — |
| `oq/wrap-defaults` | Should `StateSpaceModel::isAngularObservation` default to `true` for the two angular benchmarks (`BearingOnlyTracking`, `ReentryVehicle`)? Current default `false` means R32 wrap logic is dormant in every default benchmark. | `Common/include/StateSpaceModel.h:64`; `Benchmarks/include/BenchmarkProblems.h` | plan-phase, verify | — |
| `oq/wrap-cross-filter` | Extend angular-innovation wrap from SRUKF (`UKF/include/SRUKF.h:383,417`) to UKF, EKF, PKF, RBPKF, and every smoother. | mapper finding (CONCERNS §1.2) | plan-phase, execute | — |
| `oq/smoother-psd-guard` | Replace silent `Identity()` fallback in `sqrt_spd` (`UKF/include/SRUKFSmoother.h:113-122`) with a runtime policy (throw / log / fall back). | mapper finding (CONCERNS §1.1) | plan-phase, execute, verify | — |
| `oq/ci-pin-drift` | CI pins OptMathKernels **v0.5.15** while working tree runs **v0.6.3**. Move CI to v0.6.3 or pin `OPTMATH_DIR` to a specific commit. | `.github/workflows/ci.yml:41-45` vs `DEVELOPMENT_NOTES.md:132` | plan-phase (build), execute | — |
| `oq/version-drift` | README says v3.4.0; changelog reaches v3.4.2. Bump README version banner. | `README.md`, `DEVELOPMENT_NOTES.md` §Changelog | plan-phase (docs) | — |
| `oq/dead-filter-math-gpu` | **CLOSED 2026-07-23** — `Common/include/FilterMathGPU.h` deleted (never `#include`d anywhere; verified by full rebuild + 34/34 CTest). | mapper finding, `DEVELOPMENT_NOTES.md` mentions | — | resolved by removal |
| `oq/full-rts-benchmarks` | Full-interval RTS smoothers are CTest-only; add to `Benchmarks/src/run_benchmarks.cpp` so RMSE/NEES numbers are user-visible. | `SRUKF_STATUS.md`, `DEVELOPMENT_NOTES.md` §v3.4.0 | plan-phase, execute | — |
| `oq/reject-outliers-bench` | `setRejectOutliers()` policy is exposed (R33) but no benchmark row compares on/off. | `SRUKF_STATUS.md` §Numerical Safety Chain | plan-phase, verify | — |
| `oq/sigma-tuning-sweep` | Systematic α/β/κ sweep across the four benchmark dimensions (2, 4, 6, 10) has not been recorded. Dimension-adaptive defaults are heuristic. | `SRUKF_STATUS.md` §Technical Details | plan-phase, verify | — |
| `oq/particle-count-sweep` | Particle-count sensitivity study for PKF and RBPKF is absent. | mapper finding (CONCERNS §1.5) | plan-phase, verify | — |
| `oq/vanderpol-high-mu` | High-μ Van der Pol stiffness stress test is not in the suite. | mapper finding (CONCERNS §1.7) | plan-phase (VdP) | — |
| `oq/reentry-regimes` | Reentry regime-transition (atmospheric / peak-heating / terminal) is not resolved as regime-switching filter behavior. | mapper finding (CONCERNS §1.8) | plan-phase (reentry) | — |
| `oq/long-run-psd` | Long-run (10⁴–10⁵ step) PSD stress on the smoother backward passes. | mapper finding (CONCERNS §4) | plan-phase, verify | — |
| `oq/cmake-min-inconsistency` | `cmake_minimum_required` disagrees across trees (3.10 / 3.14 / 3.20). Unify. | mapper finding (CONCERNS §6) | plan-phase (build) | — |
| `oq/retracted-doc-purge` | Retracted `COMPARISON_RESULTS.md` performance figures still resident in-tree despite banners. Move to `docs/archive/` or delete. | mapper finding (CONCERNS §7) | plan-phase (docs) | — |
| `oq/audit-closure-state` | July 14 broader audit — no explicit "audit closed" statement. Confirm every finding is either fixed or logged as `oq/*`. | `AUDIT_2026-07-08.md` header | plan-phase (audit) | — |

---

## E. External Cross-Check Anchors

Reserved for downstream compare-experiment or peer-review workflows. Populate
when specific real-world datasets or published benchmark numbers are adopted.

| Anchor ID | Kind | Locator | Carry Forward To |
|---|---|---|---|
| `xref/vanderpol-published-nees` | benchmark | Any published UKF study on Van der Pol (e.g. Wan-van der Merwe 2000 §5.1) | verify (VdP parity) |
| `xref/reentry-published` | benchmark | Julier-Uhlmann 2004 reentry Table I | verify (reentry parity) |
| `xref/bearing-published` | benchmark | Aidala-Hammel 1983 tables; Ristic-Arulampalam-Gordon *Beyond the Kalman Filter*, Artech House, 2004 §7 | verify (bearings-only parity) |
| `xref/rbpf-published` | benchmark | Schön-Gustafsson-Nordlund 2005 §V benchmarks | verify (RBPKF parity) |
| `xref/particle-textbook` | benchmark | Doucet-de Freitas-Gordon 2001 §11 | verify (PKF sanity) |

---

## F. Private / Excluded Content

Per user's global `CLAUDE.md`, these files are **not** in scope for the map and
are excluded from every artifact in `.gpd/research-map/`:

- `include/optmath/ukf_aoa_tracking.hpp`
- `include/optmath/ukf_aoa_doppler_tracking.hpp`
- `include/optmath/iridium_burst_demodulator.hpp`
- `include/optmath/multi_satellite_tracker.hpp`
- `examples/iridium_aoa_tracking.cpp`
- `examples/compare_aoa_doppler.cpp`
- `examples/iridium_tracking_complete.cpp`
- `tests/test_ukf_aoa.cpp`

These are contract-work implementations of Iridium-Next satellite tracking with a
two-antenna coherent receiver / UKF / AOA+Doppler / multi-satellite tracking. Do
not reference them in any research-map deliverable.
