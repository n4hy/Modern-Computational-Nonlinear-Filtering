# CONCERNS — Known Issues, Open Questions, Fragile Areas

> Focus: **status**. Snapshot of what is not yet solved, what is dormant, what
> disagrees with itself, and where numerical / theoretical / build risk lives in
> the Modern Computational Nonlinear Filtering library at v3.4.x.
>
> This document is source-derived. Every claim points at either a code path
> (`file:line`) or a status doc (`AUDIT_2026-07-08.md`, `SRUKF_STATUS.md`,
> `DEVELOPMENT_NOTES.md`, `COMPARISON_RESULTS.md`, `README.md`). It is meant to
> feed roadmap creation, not to substitute for those docs.

---

## Priority Summary

| Priority | Concern | Locator |
|----------|---------|---------|
| **HIGH** | Full-interval RTS smoothers silently fall back to identity on Cholesky failure | `UKF/include/SRUKFSmoother.h:113-122` (`sqrt_spd`) |
| **HIGH** | Angular-observation wrap flag defaults to `false` and no benchmark overrides it | `Common/include/StateSpaceModel.h:64`; `Benchmarks/include/BenchmarkProblems.h` (BearingOnly, Reentry) |
| **HIGH** | Wrap logic is SRUKF-only — UKF, EKF, PKF, RBPKF, and every smoother lack it | `UKF/include/SRUKF.h:383,417` |
| **HIGH** | CI pins OptMathKernels v0.5.15 while working tree runs v0.6.3 | `.github/workflows/ci.yml:41-45` vs `DEVELOPMENT_NOTES.md` line 132 |
| **MED** | Version drift: `README.md` says v3.4.0, `DEVELOPMENT_NOTES.md` changelog reaches v3.4.2 | root docs |
| **MED** | `cmake_minimum_required` inconsistent across trees (3.10 / 3.14 / 3.20) | `CMakeLists.txt`, `RBPKF/CMakeLists.txt`, `Benchmarks/CMakeLists.txt` |
| **RESOLVED** | `Common/include/FilterMathGPU.h` dead-code header — **deleted 2026-07-23**; no `#include` sites ever existed | flagged repeatedly across status docs; closed by removal |
| **MED** | 6 of 12 historical CTest registrations were demo programs that could not fail | `AUDIT_2026-07-08.md` broader-audit note |
| **MED** | Smoother tests originally gated on *ratios* only; PSD not enforced in production | `UKF/tests/test_ukf_smoother.cpp:76`, `UKF/tests/test_srukf_smoother.cpp:106`, `EKF/tests/test_ekf_smoother.cpp:82` |
| **LOW** | Bearing-only weak observability shows as ~64 m range RMSE (not divergence) | `SRUKF_STATUS.md` — WORKING with 99.6% NEES-in-bounds |
| **LOW** | Deprecated performance claims still resident in `COMPARISON_RESULTS.md` | file explicitly marks them **RETRACTED** but they remain in-tree |

---

## 1. Filter-theory concerns still live in code

### 1.1 Smoother numerical safety (HIGH)
- `UKF/include/SRUKFSmoother.h` (`sqrt_spd` helper): on Cholesky failure returns
  `Identity()` **silently**. No log, no exception, no NaN flag. Downstream users
  cannot tell that the backward pass has effectively been disabled for that step.
- The PSD assertion exists **only in the tests**
  (`UKF/tests/test_ukf_smoother.cpp:76`, `UKF/tests/test_srukf_smoother.cpp:106`,
  `EKF/tests/test_ekf_smoother.cpp:82`); production callers get whatever comes
  out of the identity fallback.
- Recommended fix path: promote the PSD check into a runtime guard with a
  configurable policy (throw / log / fall back), analogous to `setRejectOutliers`
  in SRUKF (R33). Cross-reference `SRUKF_STATUS.md` "Numerical Safety Chain".

### 1.2 Angular-observation wrap coverage (HIGH)
- `Common/include/StateSpaceModel.h:64` — the `isAngularObservation` flag defaults
  to `false`.
- Neither `BearingOnlyTracking` nor `ReentryVehicle` in
  `Benchmarks/include/BenchmarkProblems.h` overrides it, so the fix from v3.3.0
  (R32) is **dormant in every default benchmark that observes angles**.
- Wrap logic exists at only two sites: `UKF/include/SRUKF.h:383` and
  `UKF/include/SRUKF.h:417`. It is absent from UKF, EKF, PKF, RBPKF, and every
  smoother. Any wrap-sensitive scenario using anything other than SRUKF will
  silently accumulate 2π innovations.
- Impact: benchmarks appear to "work" (RMSE numbers reported in `SRUKF_STATUS.md`)
  because the true trajectory never wraps within the run, so the code path is
  never exercised — the guard is a paper guard.

### 1.3 Covariance positive-definiteness across long trajectories
- SRUKF has explicit safety in place: safe `cholupdate_downdate`, full-covariance
  fallback, `allFinite()` gates, RBPKF `normalize_weights` NaN guard
  (`AUDIT_2026-07-08.md` §Phase 3).
- Sigma-points fallback (`UKF/include/SigmaPoints.h:92`) prints a diagonal-fallback
  warning to `std::cerr` but does not flag it in the filter state. In a long
  benchmark run this may fire on many steps invisibly to the RMSE aggregator.

### 1.4 Sigma-point tuning sensitivity
- Dimension-adaptive parameters (α = 1.0, κ = 3 − n for n ≤ 5; κ = 0 for n > 5) are
  the current default (`SRUKF_STATUS.md` §Technical Details).
- Not systematically validated across the four benchmark dimensions (2, 4, 6, 10)
  as a **sweep**; the choice is documented as heuristic. β = 2 (Gaussian optimum)
  is not reported as configurable in the mapper's audit trail.
- Open question: robustness of the `κ = 0` corner for n > 5 vs. published
  guidance (van der Merwe / Wan) that uses scaled UT with α ≪ 1.

### 1.5 Particle-filter degeneracy
- `PKF/include/particle_filter.hpp` and `RBPKF/include/rbpf/rbpf_core.hpp`
  implement resampling but the ESS threshold policy is not surfaced as a
  benchmark-tunable knob. RBPKF `get_effective_sample_size` was guarded against
  divide-by-zero in Phase 3 — the guard is present, the policy question is not.
- No systematic study of particle-count sensitivity across the four benchmarks
  is recorded in `SRUKF_STATUS.md` or `COMPARISON_RESULTS.md`.

### 1.6 Bearings-only observability
- Weak observability of range is expected; current result is RMSE ≈ 64 m with
  NEES 99.6% in 95% bounds (`SRUKF_STATUS.md`). Divergence count is 0 after the
  problem-scaled threshold fix.
- Concern is **historical**: `COMPARISON_RESULTS.md` still contains 175/182/284
  divergence figures marked "superseded" but not deleted. Anyone reading the file
  linearly can be misled by the retracted section.

### 1.7 Van der Pol stiffness at high μ
- Test problem runs at a fixed μ (documented in `Benchmarks/`, not enumerated
  here). No high-μ stiffness sweep on record. Van der Pol is a canonical stiff
  benchmark for large μ; the library ships only the moderate-μ configuration.

### 1.8 Reentry regime transitions
- 6D reentry benchmark passes at RMSE ≈ 369 m, NEES 95.6% in bounds
  (`SRUKF_STATUS.md`).
- No explicit coverage of the atmospheric-entry / peak-heating / terminal
  transition boundary as regime-switching filter behavior.

---

## 2. TODO / FIXME / WARNING markers in the tree

| Marker | Site | Note |
|---|---|---|
| WARNING (comment) | `Common/include/TestCheck.h:13` | "WARNING and then returning 0 is not a test — CTest hides stdout by default" — points to the demo-program CTest problem cited by the July 14 audit. |
| WARNING (comment) | `UKF/main.cpp:160` | Same pattern — a warn-and-return-0 pathway was removed but the comment records the reason. |
| WARNING (log) | `UKF/include/SigmaPoints.h:92` | Diagonal fallback on covariance decomposition failure — silent-to-filter, only stderr. |
| CMake WARNING | `PKF/CMakeLists.txt:113,117` | glslc/glslangValidator not found — PKF Vulkan runtime will fail with missing `vec_add.comp.spv`. Build succeeds; runtime does not. Latent build/runtime mismatch. |
| CMake WARNING | `CMakeLists.txt:245` | Sanitizers requested on non-GCC/Clang compilers are ignored — expected, but users get no other indicator. |
| CRITICAL WARNING (doc) | `COMPARISON_RESULTS.md:212` | SRUKF NEES 7.5e11 flagged as "dangerous for multi-sensor fusion" — this is **superseded** (metric-artifact fix in v3.3.0) but text remains as-is. |

No live `TODO` / `FIXME` / `HACK` / `XXX` markers were found outside the exclusion
paths.

---

## 3. Audit-remediation state

### 3.1 `AUDIT_2026-07-08.md` (v3.3.0 audit) — **fully superseded**
Explicitly marked historical; carries `[SUPERSEDED]` inline flags where v3.4.0,
July 10, and July 14 work has replaced items. Key superseded items:
- Kernel-pinning mechanism `OPTMATH_RELEASE_TAG` **removed** in commit `168dafd`
  (July 10). Now consumed as `add_subdirectory` from `OPTMATH_DIR` sibling tree.
- `-march=native` / unconditional SIMD replaced by `NLF_ENABLE_NEON` /
  `NLF_ENABLE_SVE2` / `NLF_ENABLE_VULKAN` / `NLF_ENABLE_NATIVE_ARCH` toggles
  (commit `c8d2905`).
- Broader July 14 audit found: 6 of 12 CTest registrations were demo programs
  that could not fail; smoother tests gated on ratios only. Both surfaced fixes
  are landed but the demo-program pattern remains an ongoing test-hygiene concern.

### 3.2 Phase 8 audit-remediation (most recent, per git log)
- Commit `70f4adb`: regression tests, uniform warnings, sanitizers, GitHub Actions.
- Commit `2283ca5`: 28/28 tests, CI, sanitizers logged.
- Commit `e4a2ac9`: docs+CI drift fix around `OPTMATH_DIR` provisioning and test
  counts.
- Remaining question: whether **all** items from the July 14 broader audit are
  closed. `DEVELOPMENT_NOTES.md` is the authoritative log per the audit header,
  but no explicit "audit closed" statement is recorded.

### 3.3 `SRUKF_STATUS.md` (v3.4.0)
Reports all four benchmark problems **WORKING** with fixed-lag smoother. Open
items visible in-doc:
- **Full-interval RTS smoother not in the benchmark suite**; covered only by the
  `SRUKF_Smoother` CTest. Benchmark comparison against fixed-lag is not recorded.
- SRUKF **NIS gate + reject-outliers policy** exposed but **not exercised in the
  benchmarks** — only in test coverage.

### 3.4 `DEVELOPMENT_NOTES.md`
- Documents the v0.5.15 → v0.5.17 → v0.6.3 OptMathKernels transition; the last
  audit block explicitly notes the NEON GEMM rewrite is bit-exact against Eigen
  below the `OPTMATH_GEMM_EIGEN_MAX = 80·80·80` cutoff and that NLF's largest
  dynamic GEMM (M·N·K = 216) never crosses that cutoff. Structurally safe.
- Backend-portability check (aarch64 RPi5, no CUDA): NEON build 30/30 CTest,
  Eigen-only 29/29 (delta is `test_neon_fp16`, a dependency test). RMSE/NEES
  agrees to reported precision. Timing is **not** portable and is explicitly
  refused as a claim.

### 3.5 `COMPARISON_RESULTS.md`
- Retains large blocks of **RETRACTED / SUPERSEDED** content: "43–47% faster",
  divergence counts (175/182/284), the "CRITICAL WARNING" NEES box. Marked with
  banners but not physically deleted. Risk of misreading by anyone skimming.

---

## 4. Missing validation / coverage gaps

- **Angular-wrap fix is untested in the benchmark suite** — the flag is `false` at
  the benchmark problem level, so the wrap code in `SRUKF.h:383,417` never fires
  under `run_benchmarks`. Coverage is CTest-only.
- **Full-interval RTS smoother is not in the four-problem benchmark comparison**
  (only fixed-lag numbers are reported in `SRUKF_STATUS.md`).
- **UKF/EKF/PKF/RBPKF wrap coverage is nil.** No test exercises measurement wrap
  outside SRUKF.
- **NIS-gated reject-outliers policy** — no benchmark row demonstrates the
  policy on/off comparison.
- **Sigma-point α/β/κ sweep** — dimension-adaptive defaults are documented but no
  sweep is committed.
- **Particle-count sweep** for PKF/RBPKF — no numbers on record.
- **High-μ Van der Pol / regime-crossing reentry** — no stress benchmarks.
- **Long-run PSD stress** (10⁴ – 10⁵ steps) to detect covariance drift in
  smoothers — not on record.

---

## 5. Fragile areas — code sensitive to environment

- **Build/runtime split for Vulkan**: `PKF/CMakeLists.txt:113-117` warns on
  missing `glslc`/`glslangValidator`; build succeeds, PKF Vulkan will fail at
  runtime because `vec_add.comp.spv` is missing. Users on hosts without the
  SPIR-V toolchain get a delayed failure.
- **`-ffast-math` / `EIGEN_FAST_MATH=1`**: explicitly disabled in Phase 3 because
  it broke NaN guards and Cholesky precision. Any downstream user who re-enables
  either flag (e.g. via `CMAKE_CXX_FLAGS`) silently loses the guards. Not enforced
  by the build.
- **CUDA arch pinning**: Blackwell (SM 100/120) needs
  `-DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON` (`DEVELOPMENT_NOTES.md`
  §CUDA Development Status). Default multi-arch list does not include SM 120.
- **Sanitizer coverage**: only GCC/Clang. Non-GCC/Clang builds silently drop the
  ASan/UBSan safety net.
- **`Common/include/FilterMathGPU.h`** — deleted 2026-07-23. The header carried
  ~460 lines targeting the optmath CUDA API but was never `#include`d anywhere
  across four upstream-audit cycles. Removal changed no compiled code (verified
  by full rebuild + 34/34 CTest post-deletion).

---

## 6. Build / CI concerns

- **Version pin drift**: `.github/workflows/ci.yml:41-45` pins OptMathKernels
  **v0.5.15**; `DEVELOPMENT_NOTES.md` line 132 says the working tree is
  **v0.6.3**. CI is validating an older dependency than developers use locally.
- **`cmake_minimum_required` inconsistency**: root 3.14, `RBPKF/CMakeLists.txt`
  3.10, `Benchmarks/CMakeLists.txt` 3.20. Any target-scoped feature reachable
  only above 3.20 will be silently unavailable to sub-projects.
- **`README.md` version 3.4.0** vs **`DEVELOPMENT_NOTES.md` changelog reaches
  v3.4.2** — user-facing README is stale.
- **`bootstrap.sh`** — not audited for reproducibility (assumes apt package
  availability, no lock file for optional accelerators).
- **Dependency pinning**: `OPTMATH_DIR` is a sibling checkout tracking `main`, no
  hash pin. Bit-exactness across contributor machines depends on discipline.

---

## 7. Documentation drift

- `README.md` (977 lines) vs `DEVELOPMENT_NOTES.md` (734 lines) — README versions
  lag by ≥ two point releases per the changelog comparison.
- `COMPARISON_RESULTS.md` (480 lines) is marked as a historical snapshot but
  contains numeric figures that a reader searching for "SRUKF vs UKF speedup"
  will surface first. Recommend converting the retracted sections to structured
  metadata or archiving the file.
- `AUDIT_2026-07-08.md` (352 lines) — same pattern: authoritative in July 2026,
  superseded in July 10 and July 14 by inline `[SUPERSEDED]` markers. Would
  benefit from a summary "current state" pointer at the top instead of relying on
  users to reach the superseded banners.

---

## 8. Open questions surfaced by the mapping pass

1. **Should the wrap flag default flip to `true` for angular-observation
   benchmarks?** Or should the wrap be moved out of `SRUKF.h` into
   `Common/include/StateSpaceModel.h` and inherited by every filter and
   smoother? (Fixes 1.2 and 1.3 together.)
2. **Should `sqrt_spd`'s identity fallback become an assertion, an exception, or
   a policy knob?** Silent identity is the most dangerous of the three.
3. **Are the retracted `COMPARISON_RESULTS.md` figures archival or actionable?**
   If archival, they should be moved to `docs/archive/`. If actionable, the
   retraction banners are hiding live claims.
4. **What is the closure state of the July 14 broader audit?** The three most
   recent commits look like remediation, but no explicit "audit closed" statement
   exists.
5. **CI dependency version.** Should CI move to v0.6.3 (matches dev tree) or
   should the tree be pinned to a specific `OPTMATH_DIR` hash?

---

## 9. Speculative next-work items surfaced in status docs

- **cuSOLVER Cholesky wire-up**: OptMathKernels v0.5.10+ ships cuSOLVER Cholesky,
  but `FilterMath.h` Cholesky dispatch does not use it yet
  (`DEVELOPMENT_NOTES.md` §CUDA Development Status).
- **Full-interval RTS smoother benchmarks**: promote from CTest-only into the
  `run_benchmarks` suite so RMSE/NEES numbers are user-visible.
- **Wrap-aware filters across UKF/EKF/PKF/RBPKF**: reuse the SRUKF innovation-wrap
  code in a shared header consumed by every filter.
- **NIS-gated outlier rejection benchmarks**: on/off rows in the comparison table
  for each filter that supports it.

---

## 10. Sources

- `AUDIT_2026-07-08.md` (352 lines) — v3.3.0 audit, superseded.
- `SRUKF_STATUS.md` (120 lines) — v3.4.0 status.
- `DEVELOPMENT_NOTES.md` (734 lines) — live engineering doc.
- `COMPARISON_RESULTS.md` (480 lines) — historical Feb–Apr 2026 snapshot.
- `README.md` (977 lines) — user-facing project doc.
- `.github/workflows/ci.yml` — CI pin drift evidence.
- Root and sub-tree `CMakeLists.txt` — build inconsistency evidence.
- `Common/include/StateSpaceModel.h`, `Common/include/FilterMath.h`
  — Common-tier concerns.
- `UKF/include/{SRUKF,UKF,SigmaPoints,SRUKFSmoother,UKFSmoother}.h`,
  `EKF/include/EKFSmoother.h`, `PKF/include/particle_filter.hpp`,
  `RBPKF/include/rbpf/rbpf_core.hpp`, `Benchmarks/include/BenchmarkProblems.h`
  — filter-tier concerns.

Private contract-work files (per user global `CLAUDE.md`) were **not** read and
are not referenced.
