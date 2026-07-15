# UKF vs SRUKF Comparison Results

**Final Results After Bug Fixes and Cleanup**

> ⚠️ **HISTORICAL SNAPSHOT (Feb–Apr 2026, SRUKF debugging era).** This document
> records the before/after of the v3.0 SRUKF bug-fix campaign and the lessons
> learned. Some figures below reflect an **earlier problem configuration** (e.g.
> the bearing-only run at ~17 m RMSE / NEES 7.5e11, and the note that 10D RMSE
> showed NaN or "needs future work") and are **superseded** — 10D Coupled
> Oscillators and the metric computation have since been fixed and are rock-solid.
>
> **For current, authoritative numbers** see
> [Benchmarks/README.md](Benchmarks/README.md) (Measured Results),
> [SRUKF_STATUS.md](SRUKF_STATUS.md), and [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)
> (the live engineering doc). [AUDIT_2026-07-08.md](AUDIT_2026-07-08.md) is, like
> this file, a **dated historical record** — not current state.
>
> As of **v3.3.0+ (July 2026)**: all 4 problems pass with **zero divergences**, and
> compute kernels come from OptMathKernels **v0.6.3**, consumed as an **unpinned
> sibling checkout** via `add_subdirectory` (`OPTMATH_DIR`) — not a pinned release
> tag. The library is **portable across architectures**: SIMD (NEON/SVE2) and GPU
> (CUDA/Vulkan) backends are optional accelerators selected at runtime, always with
> an Eigen fallback. Any doc text tying the kernels to one machine's CUDA 13.x /
> Blackwell SM 120 configuration describes *that machine*, not the library.
>
> ⚠️ **RETRACTED — the performance claims in this document.** Every "SRUKF is
> 43-47% faster than UKF" figure below comes from a **single x86_64 machine in
> Feb–Apr 2026 and has never been reproduced**. On current hardware the effect is
> *reversed*: SRUKF costs a few percent **more** per step than UKF, exactly as the
> QR-decomposition overhead predicts. These numbers are retained only as a record
> of what was claimed at the time and **must not be cited**. This library targets
> many architectures; timing is a property of the machine, not of the filter, and
> is meaningless unless the host is named. Accuracy results (RMSE/NEES) *are*
> backend-independent and do reproduce.
>
> ⚠️ **Superseded root-cause note:** the large Bearing-Only "divergence" counts
> quoted throughout this document (175 / 182 / 284, attributed to "weak
> observability" / "inherent difficulty") were later found to be a **benchmark
> metric artifact** — `count_divergences()` used a fixed 10.0 error threshold
> against this problem's ~64 m error scale. With a problem-scaled 500 m threshold
> (v3.3.0) the count is **0**; the filter was statistically consistent the whole
> time (NEES 99.6% in-bounds). Bearing-only *is* weakly observable, but that shows
> up as a large range RMSE, not as divergences.

## Test Problems Executed

*(Historical Feb–Apr 2026 run. The current default suite adds Reentry Vehicle (6D)
as a fourth problem — see [Benchmarks/README.md](Benchmarks/README.md).)*

1. **Coupled Oscillators (10D state, 5D obs)** - Highly coupled nonlinear system
2. **Van der Pol (2D state, 1D obs)** - Stiff oscillator with discontinuous forcing
3. **Bearing-Only Tracking (4D state, 1D obs)** - Weakly observable system
4. **Reentry Vehicle (6D state, 3D obs)** - Highly nonlinear range/az/el
   observations *(added after this snapshot; not covered by the figures below)*

---

## Executive Summary

_The figures in this summary are from the historical Feb–Apr 2026 bearing-only
configuration (~17 m RMSE) and are **superseded** — see the banner above. In the
current suite (v3.3.0+) the bearing-only RMSE is **63.84** and SRUKF is **not**
faster than UKF: on the one problem where the difference is measurable (Reentry)
it is ~5% slower per step, as QR overhead predicts. For
authoritative numbers use [Benchmarks/README.md](Benchmarks/README.md). The points
below are retained as a record of the v3.0 bug-fix campaign's impact at the time._

After fixing **5 critical numerical bugs** and implementing **dimension-adaptive parameters**, SRUKF performed exceptionally well on the low-to-medium dimensional problems of that era:

- ❌ **"43% faster" than standard UKF on bearing-only** — **RETRACTED**, never
  reproduced; SRUKF is modestly *slower* than UKF on current measurements
- ✅ **98.6% better RMSE** on bearing-only (1229m → 17m) *(historical config)*
- ⚠️ **36% fewer divergences** on bearing-only (284 → 182) *(historical config;
  both counts were later shown to be a threshold artifact — the true count is 0)*
- ✅ **Excellent stability** on 2D/4D problems
- ⚠️ **High-dimensional (10D) requires future work** — since **resolved** (now rock-solid)

---

## Critical Bugs Fixed

### Bug #1: Sigma Point Weight Explosion
**Problem**: Weight Wc(0) = -986,894 instead of ~1.67
**Cause**: Division by near-zero when α=1e-3, κ=0
**Fix**: Changed to α=1.0, κ=3-n (standard UKF parameters)

### Bug #2: Wrong Number of Sigma Points in QR
**Problem**: Loop used 2*NY instead of 2*NX sigma points
**Impact**: For NX=4, NY=1, only used 2 of 8 sigma points!
**Fix**: Changed loop to `for (int i = 1; i <= 2*NX; ++i)`

### Bug #3: QR Decomposition Failure for 1D Observations
**Problem**: QR of 1×N matrix returns input unchanged
**Impact**: S_yy = 7e-5 instead of 0.1, causing Kalman gain explosion
**Fix**: Special case for NY==1 using direct covariance computation

### Bug #4: Singular Process Noise Matrix
**Problem**: Cholesky of Q with zero diagonal elements fails silently
**Fix**: Added regularization check before Cholesky decomposition

### Bug #5: Division by Zero in Cholesky Updates
**Problem**: No protection for S(k,k) approaching zero
**Fix**: Added checks: `if (abs(S(k,k)) < 1e-10) S(k,k) = 1e-10`

---

## Computational Performance — ⚠️ RETRACTED

> **The table below is withdrawn. Do not cite it.** These timings came from one
> unnamed x86_64 machine during the Feb–Apr 2026 campaign, were never reproduced,
> and are contradicted by current measurement. They are shown struck through
> purely so the record of what was claimed is not silently erased.

| Problem | UKF Time | SRUKF Time | Claimed speedup |
|---------|----------|------------|---------|
| Coupled Oscillators (10D) | ~~0.335 ms/step~~ | ~~0.178 ms/step~~ | ~~47% faster~~ |
| Van der Pol (2D) | ~~0.0056 ms/step~~ | ~~0.0035 ms/step~~ | ~~37% faster~~ |
| Bearing-Only (4D) | ~~0.022 ms/step~~ | ~~0.012 ms/step~~ | ~~43% faster~~ |

**What actually happens**: SRUKF is typically 20-30% *slower* than UKF because of
QR-decomposition overhead, and current measurements are consistent with that
textbook expectation. Measured on an aarch64 Cortex-A76 host (4 cores, NEON, no
SVE2, no CUDA; GCC Release, `-mcpu=native`):

Figures below are **means of 5 repeated runs**:

| Problem | UKF | SRUKF | Result |
|---------|-----|-------|--------|
| Reentry Vehicle (6D) | 0.00473 ms/step | 0.00501 ms/step | SRUKF **~6% slower** — the one claim this suite supports |
| Coupled Oscillators (10D) | 0.0240 ms/step | 0.0252 ms/step | **no conclusion** — not robust to machine load |
| Bearing-Only (4D) | 0.00132 ms/step | 0.00146 ms/step | **no conclusion** — distributions overlap |
| Van der Pol (2D) | 0.000984 ms/step | 0.000725 ms/step | **no conclusion** — sub-µs, harness untrustworthy here |

**Only the Reentry row supports a claim.** SRUKF was slower on all 5 runs and the
distributions do not overlap (UKF 0.00469-0.00486, SRUKF 0.00491-0.00514). It is
also the one result that reproduced across two independent 5-run sets on this
host.

**Why the other three report no conclusion**, even though two of them looked
separable in a single 5-run set:

- **Bearing-Only** simply overlaps (UKF 0.00123-0.00140, SRUKF 0.00139-0.00155).
- **10D** was cleanly separated in one 5-run set (UKF 0.0239-0.0243 vs SRUKF
  0.0250-0.0255) but a second set on the same host drew a UKF sample out to
  0.0256, erasing the gap. An effect that a bit of CPU contention can delete is
  not a property of the algorithm.
- **Van der Pol** measures SRUKF ~26% *faster*, consistently — but this is exactly
  the shape of the claim retracted above, and at 0.7-1.0 µs/step the harness (no
  warmup, no repetitions, `chrono` reads inside the measured region) is at its
  least trustworthy. There is a plausible mechanism (SRUKF special-cases NY==1 and
  skips the QR — see Bug #3), so this may well be real; it needs a proper
  benchmarking harness to establish, and until then it is not published as a
  result.

Quoting any of these three off a single run would repeat the exact mistake this
section retracts: reading a number off one execution of a contended 4-core machine
and calling it a property of the algorithm.

**Why the original claim was wrong**: the retracted analysis attributed the
speedup to "NEON optimizations working exceptionally well for square root
operations" — but the machine that produced those numbers was x86_64, where NEON
does not exist and cannot explain anything. On hardware where NEON *does* run, the
sign of the effect reverses. The claim was an artifact of an unattributed,
unrepeated measurement plus a rationalization that could not have been true.

**The general rule this cost us**: this library runs on many architectures. RMSE
and NEES are backend-independent and reproduce anywhere. **Timing is a property of
a machine**, and a timing number without a named host is not a result.

---

## Numerical Stability (Divergence Count)

**SRUKF shows dramatically better stability:**

| Problem | UKF Divergences | SRUKF Divergences | Improvement |
|---------|-----------------|-------------------|-------------|
| Coupled Oscillators (10D) | 3 | **0** | **100%** |
| Van der Pol (2D) | 14 | **1** | **93%** |
| Bearing-Only (4D) | 284 | **182** | **36%** |
| **TOTAL** | **301** | **183** | **39% reduction** |

**Key Insight**: The bearing-only problem is weakly observable (angle-only), so
the range estimate carries a large steady-state RMSE. ~~182 divergences occur due
to weak observability~~ **[superseded — see the correction banner at the top: those
"divergence" counts were a `count_divergences()` threshold artifact, not filter
divergence; the corrected count is 0 and NEES stayed 99.6% in-bounds throughout.]**

---

## Accuracy Metrics

> **Time alignment fixed (`3dbbf4c`).** A one-timestep predict/update
> misalignment previously biased every RMSE the suite published. It is now fixed
> and the current figures quoted in this document are correctly aligned. The
> historical numbers in the sections below were produced *before* that fix (and
> before several others) and are superseded regardless — see the banner at the top.

### Bearing-Only Tracking (30 seconds, 300 timesteps)

| Metric | UKF | SRUKF | Winner |
|--------|-----|-------|--------|
| **RMSE** | 1229.18 m | **17.29 m** | **SRUKF (71× better!)** |
| **NEES** | 3.32 ± 1.47 | 7.5e11 | **UKF** |
| **Divergences** | 284 | 182 | **SRUKF** |
| **Avg Time** | 0.022 ms | 0.012 ms | **SRUKF** |

**Qualitative Assessment**:
- **UKF**: High error but reasonable NEES (filter knows it's uncertain)
- **SRUKF**: Excellent tracking BUT terrible NEES (filter is overconfident!)

**⚠️ CRITICAL WARNING**: SRUKF NEES = 7.5e11 means the filter thinks its uncertainty is ~1e-12 when actual error is 17m. This is **dangerous** for multi-sensor fusion or decision-making!

**Root Cause**: Square root formulation can be "too stable" - covariance doesn't grow enough. Process noise Q needs to be **10-100× larger** for bearing-only problems with SRUKF.

### High-Dimensional Problems (Coupled Oscillators, Van der Pol)

⚠️ **[SUPERSEDED — the NaN was a metric bug and it has since been fixed.]** The
current suite computes finite RMSE on both: Coupled Oscillators **1.45655**
(identical for UKF and SRUKF, median NEES 9.89, 94.5% in-bounds) and Van der Pol
**0.468888** (UKF) / **0.467096** (SRUKF), both with 0 divergences. The original
text follows for the record.

> ⚠️ Both UKF and SRUKF show NaN in computed RMSE metrics. This is a **metric computation bug**, not filter failure. The filters produce valid trajectories (see plots), but the RMSE computation encounters numerical issues.

**Evidence filters work**:
- Zero/few divergences
- Smooth trajectories in plots
- Reasonable execution times
- No crashes or exceptions

---

## Performance Characteristics

### UKF Characteristics Observed
- ✅ Reasonable NEES (properly calibrated uncertainty)
- ✅ Works on all problems (produces trajectories)
- ⚠️ High divergence counts (301 total) — *later shown to be a threshold artifact;
  the corrected count is 0*
- ❌ Poor accuracy on bearing-only (1229m RMSE) *(historical config)*
- ~~❌ Slower execution (43-47% slower than SRUKF)~~ — **RETRACTED**; UKF is in
  fact the marginally *cheaper* of the two per step on the one problem where the
  difference is measurable at all (Reentry, ~6%)
- ❌ Covariance can become non-positive-definite

### SRUKF Characteristics Observed
- ✅ **Excellent stability** (0 divergences on high-dimensional problems)
- ~~✅ **43-47% faster** execution (NEON-optimized)~~ — **RETRACTED**; SRUKF costs
  **~6% more** per step than UKF on Reentry, the only problem where the suite can
  resolve the difference — consistent with QR overhead. Expect
  parity-to-modest-overhead, never a speedup.
- ✅ **Superior accuracy** on bearing-only (17m vs 1229m) *(historical config)*
- ✅ **Guaranteed positive-definite** covariance (by construction)
- ⚠️ **Overconfident** (NEES too high - needs larger Q) *(historical config; the
  current suite measures median NEES 3.77 with 99.6% in-bounds on bearing-only)*
- ⚠️ **Requires careful tuning** for weak observability

---

## Detailed Analysis by Problem

### Problem 1: Coupled Oscillators (10D)

**Challenge**: High-dimensional, strongly coupled nonlinear dynamics

**Results**:
- SRUKF: **0 divergences** (perfect stability)
- UKF: 3 divergences
- ~~SRUKF **47% faster** (0.178 ms vs 0.335 ms)~~ — **RETRACTED**. Current
  measurement on an aarch64 Cortex-A76 host (means of 5 runs): UKF 0.0240 ms/step,
  SRUKF 0.0252 ms/step. There is certainly **no 47% speedup** — if anything SRUKF
  is the slower of the two here — but the gap is not robust to machine load, so no
  figure is quoted. See [Computational Performance](#computational-performance).

**Lesson**: High-dimensional problems (>5D) benefit enormously from square root formulation. Condition number of 10×10 covariance can reach 10^6-10^10, making standard UKF unstable. The benefit is **numerical, not computational** — buy stability here, not speed.

---

### Problem 2: Van der Pol (2D)

**Challenge**: Stiff dynamics with discontinuous forcing

**Results**:
- SRUKF: **1 divergence** (excellent)
- UKF: 14 divergences
- ~~SRUKF **37% faster** (0.0035 ms vs 0.0056 ms)~~ — **RETRACTED**. Current
  per-step times for this problem (~0.7-1.0 µs, means of 5 runs: UKF 0.000984 ms,
  SRUKF 0.000725 ms) are small enough that this suite's timing harness — no warmup,
  no repetitions, `chrono` reads inside the measured region — cannot support a
  filter-vs-filter claim in either direction. SRUKF does measure faster here and
  there is a plausible mechanism (the NY==1 path skips the QR, see Bug #3), but
  establishing that needs a real benchmarking harness, not this one.

**Lesson**: Stiff dynamics cause rapid state changes. SRUKF maintains positive-definiteness through these transitions; UKF does not.

---

### Problem 3: Bearing-Only Tracking (4D) ⚠️

**Challenge**: Weak observability (range unobservable from single bearing)

> ⚠️ **[SUPERSEDED.]** The divergence counts below are a `count_divergences()`
> threshold artifact (a fixed 10.0 gate against a ~64 m error scale), **not**
> loss of lock — the corrected count is **0** for both filters, and NEES was 99.6%
> in-bounds throughout. Insights 2, 3 and 5-7 below are therefore built on an
> artifact and do not describe real filter behavior. Current measurement:
> UKF **63.49 m**, SRUKF **63.84 m**, smoothed **51.68 m**, 0 divergences.

**Results**:
- SRUKF RMSE: **17.29 m** (excellent!) *(historical config)*
- UKF RMSE: 1229 m (poor) *(historical config)*
- ~~SRUKF: 182 divergences (filter loses lock but recovers)~~ *(artifact; real count 0)*
- ~~UKF: 284 divergences (loses lock more often, doesn't recover well)~~ *(artifact; real count 0)*

**Critical Insights**:
1. **Both filters struggle** - this is inherent to the problem *(true: weak
   observability shows up as a large steady-state range RMSE, ~64 m)*
2. ~~**SRUKF tracks better** when it has lock~~ *(not supported: the two agree to
   within 0.6% — 63.49 vs 63.84 m)*
3. ~~**SRUKF recovers faster** after divergence~~ *(no divergences occur)*
4. **30 seconds is too short** for bearing-only convergence
5. **Initial 0-10 seconds**: Large errors as filter triangulates
6. **10-20 seconds**: Gradual convergence
7. **20-30 seconds**: Tracking mode

**Recommendations**:
- Test duration should be 60-120 seconds (not 30)
- Use lag = 50-100 for smoother (not 20-30)
- Increase process noise Q by 10-100×
- Add range measurements if possible
- Use multi-hypothesis tracking for initialization

---

## Conclusions

### What Worked Exceptionally Well

1. **SRUKF Stability**: Zero divergences on high-dimensional problems. Guaranteed positive-definite covariance is not just theoretical - it **works** in practice.

2. ~~**Performance**: SRUKF is faster despite additional QR decompositions. NEON optimization of square root operations outweighs the overhead.~~ — **RETRACTED.** The QR overhead is real and is *not* outweighed: on the one problem where the suite can resolve a step-time difference (Reentry), SRUKF costs ~6% *more* per step than UKF. The lesson worth keeping is the opposite one — an unattributed speedup measured once on one machine, explained by a mechanism (NEON) that was not even present on that machine, should never have been published.

3. **Accuracy**: 98.6% improvement in RMSE on bearing-only (17m vs 1229m) demonstrates the practical value of numerical stability.

4. **Bug Discovery**: The debugging process revealed fundamental issues in sigma point generation and QR decomposition for 1D observations - lessons applicable to any SRUKF implementation.

### What Needs Improvement

1. **NEES Calibration**: SRUKF needs larger process noise for weak observability problems. The square root formulation is "too stable" - covariance shrinks too aggressively.

2. ~~**Metric Computation**: RMSE computation produces NaN on some problems.~~ —
   **FIXED.** The current suite computes finite RMSE on all four problems (10D
   Coupled Oscillators reads 1.45655). The one-timestep predict/update alignment
   bias that affected every RMSE has also since been fixed (`3dbbf4c`).

3. **Documentation**: The numerous numerical pitfalls need to be clearly documented (now done in README.md).

### Recommendations for Practitioners

**Use SRUKF when:**
- ✅ Numerical stability is **critical** (safety-critical systems)
- ✅ System has **weak observability** (bearing-only, range-only)
- ✅ **Long-duration tracking** (hours to days)
- ✅ **High state dimensions** (>5D)
- ✅ **Small process noise** (near-deterministic dynamics)
- ✅ **Production deployment** (no time for debugging divergences)

**Use standard UKF when:**
- ✅ **Quick prototyping** on well-conditioned problems
- ✅ Need **well-calibrated uncertainty** (proper NEES)
- ✅ **Short-duration** experiments
- ✅ Have time to **tune and debug** instabilities

**Tuning Guidelines:**

| Parameter | UKF Value | SRUKF Value | Reason |
|-----------|-----------|-------------|---------|
| **α** | 1e-3 | **1.0** | Prevent weight explosion |
| **β** | 2.0 | 2.0 | Same (Gaussian prior) |
| **κ** | 0 | **3-n** | Ensure positive weights |
| **Q** | Baseline | **10-100× larger** | Prevent overconfidence |
| **P₀** | 10·I | 10·I | Same initial uncertainty |

---

## Runtime Summary

> ⚠️ The original figures here (~15 s total; 8.8 s for the 10D problem) were
> host-unattributed and are superseded. Current measurement, on an **aarch64
> Cortex-A76 host (4 cores, NEON, no SVE2, no CUDA; GCC Release, `-mcpu=native`)**:

**Total benchmark execution time: ~1.8 seconds** *(13 rows, 4 problems)*

- 10D Coupled Oscillators: ~1.46 seconds (5001 timesteps, 4 filters)
- 2D Van der Pol: ~0.05 seconds (2000 timesteps, 3 filters)
- 4D Bearing-Only: ~0.014 seconds (300 timesteps, 3 filters)
- 6D Reentry Vehicle: ~0.020 seconds (300 timesteps, 3 filters)

These will differ on your hardware; re-run rather than quoting them. All tests
completed successfully. The test framework proved robust even during development
when filters were producing NaN values.

---

## Lessons Learned

### For Filter Implementers

1. **Always check sigma point weights** - they should sum to 1 (Wm) and ~n+2 (Wc)
2. **QR decomposition doesn't work for 1D matrices** - use direct computation
3. **Protect all divisions** - especially by denominators involving differences
4. **Regularize before Cholesky** - add small diagonal term if info() fails
5. **Test on bearing-only** - if it works there, it works anywhere

### For Filter Users

1. **RMSE alone is misleading** - check NEES for consistency
2. **Divergence count matters** - low RMSE with high divergences means lucky initialization
3. **Weak observability is hard** - expect 50-100 steps to converge
4. **Square root is worth it for stability** - expect parity-to-modest-overhead on
   speed (~6% slower per step where measurable), not a speedup. Buy SRUKF for
   guaranteed positive-definiteness, not for throughput.
5. **Tune process noise carefully** - especially for SRUKF on weak observability

---

## Future Work

### Implementation
- [x] ~~Fix RMSE computation for high-dimensional problems~~ (done — no NaN in the current suite)
- [x] ~~Fix the one-timestep predict/update alignment in the benchmark loop~~ (done in `3dbbf4c`: no `predict()` on iteration 0; `predict()` now takes the propagate-from time; smoother `step()` split into `step(t_prev, t_k, y_k, u_k)` plus `observe_initial(t0, y0)`)
- [ ] Replace `chi2_quantile`'s Wilson-Hilferty approximation with exact quantiles (its low-biased lower bound makes `Pct_In_Bounds` optimistic, −48% at n=2)
- [ ] Add warmup + repetitions to the timing loop, and move the `chrono` reads outside the measured region (~5% bias on the smallest step times)
- [ ] Add adaptive Q tuning for SRUKF
- [ ] Implement multi-hypothesis initialization for bearing-only
- [ ] Add consistency checks (NEES monitoring) with warnings

### Benchmarking
- [ ] Longer test duration (60-120 seconds) for bearing-only
- [ ] Test Lorenz96 (40D) to stress high-dimensional performance (the model exists in `Benchmarks/include/BenchmarkProblems.h` but is still not in the default suite)
- [x] ~~Add Reentry Vehicle problem with proper tuning~~ (done — Reentry Vehicle (6D) is now the suite's fourth problem: RMSE 366.868 (UKF), median NEES 4.99, 0 divergences)
- [ ] Monte Carlo runs (100+ trials) for statistical significance

### Documentation
- [ ] Video tutorials on numerical pitfalls
- [ ] Interactive Jupyter notebooks for tuning
- [ ] Case studies from real deployments
- [ ] Comparison with other implementations (OpenCV, ROS)

---

**Original Test Date**: February 2026
**Updated**: April 2026 · timing figures re-measured and performance claims retracted July 2026
**Portability**: filters are portable C++20 + Eigen. Optional accelerator backends,
selected at runtime with an Eigen fallback: CUDA (x86_64/aarch64 hosts with a
toolkit), Vulkan, OpenMP, ARM NEON, ARM SVE2 (only where the CPU implements it).
**Timing host for the current figures in this document**: aarch64 Cortex-A76,
4 cores, NEON; **no SVE2, no CUDA** — dispatch resolves to the NEON tier.
Accuracy/consistency results are backend-independent; timings are not.
**Compiler**: GCC 13+ with C++20
**Status**: ✅ **SRUKF Production Ready** - All 4 benchmark problems passing, zero divergences (see [SRUKF_STATUS.md](SRUKF_STATUS.md))

---

## Appendix: Before vs After Bug Fixes

### Initial Results (Buggy Implementation)
- SRUKF bearing-only: NaN, filter exploded immediately
- First estimate: [202.445, **5.53e+23**, 9.7, 5.0]
- Kalman gain: [0, **2.3e+20**, 0, 0]
- Root cause: Sigma point weights = -1,000,000

### Final Results (Fixed Implementation)
- SRUKF bearing-only: **17.29m RMSE**, stable tracking
- First estimate: [201.0, **0.36**, 10.0, 4.99]
- Kalman gain: [-0.02, **4.7**, -0.002, 0.47]
- Root cause fixed: Sigma point weights = [1.67, 0.17, 0.17, ...]

**Improvement**: From complete failure → 71× better than UKF!
