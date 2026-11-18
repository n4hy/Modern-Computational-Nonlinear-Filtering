# SRUKF Implementation Status

## Current State (v3.4.2, July 2026)

> SRUKF (R32) wraps angular **observation** innovations to [−π, π], and (R33)
> exposes the normalized innovation squared (NIS) via `getLastNIS()` plus an
> innovation gate with an optional reject-outliers policy (`setRejectOutliers()`).
> See the Numerical Safety Chain below.
>
> **v3.4.2 hardening**: `initialize()` throws `std::runtime_error` on NaN / Inf /
> asymmetric / non-PSD `P0` (including the condition number in the not-PSD
> message) instead of silently degrading to `L0 = Identity`; the
> innovation-gate threshold is now a member with `setInnovationGateChi2()` /
> `getInnovationGateChi2()` accessors; the first gate firing logs a single line
> to `std::clog` (per process since 2026-11-16; it was per filter instance); and `getGatedCount()` is exposed alongside
> `getLastNIS()`. Regression coverage: the `SRUKF_Initialize` CTest (source
> `UKF/tests/test_srukf_initialize.cpp`).

### All Benchmark Problems Working

| Problem | SRUKF RMSE | NEES median | In 95% bounds | Status |
|---------|-----------|-------------|---------------|--------|
| Coupled Oscillators (10D) | 1.457 | 9.89 | 94.5% | **WORKING** |
| Van der Pol (2D) | 0.467 | 1.14 | 94.8% | **WORKING** |
| Bearing-Only (4D) | 63.84 | 3.72 | 99.3% | **WORKING** |
| Reentry Vehicle (6D) | 367.1 | 4.99 | 95.6% | **WORKING** |

> These are accuracy figures, not timings: they are backend-independent and
> reproduce across hosts and acceleration tiers to within float reassociation
> (≤0.02% RMSE measured: NEON vs Eigen-only on aarch64, and x86_64 vs aarch64 —
> see DEVELOPMENT_NOTES "Backend portability"). At the 4 significant figures
> shown here no host qualifier is needed.

All 4 benchmark problems pass with the fixed-lag smoother (table below). The
**full-interval URTS smoother** added in v3.4.0 is not part of the benchmark
suite; it is covered by the `SRUKF_Smoother` CTest — see DEVELOPMENT_NOTES
v3.4.0. Post-`d668a6f` the full-interval variant is Sarkka's URTS in the
literal book form (see the URTS / SR-URTS section below); the fixed-lag
variant retains the cached-quantities recursion for streaming efficiency.

10D coupled oscillators previously failed with NaN — now resolved through:
1. Dimension-adaptive parameters (alpha=1.0, kappa=3-n for NX<=5; kappa=0 for NX>5)
2. Direct P_yy computation instead of QR for S_yy
3. Safe Cholesky downdate with full-covariance fallback
4. Removed global `-ffast-math` which was corrupting NaN guards

### Smoother Results (fixed-lag)

| Problem | SRUKF+Smoother RMSE | Improvement |
|---------|-------------------|-------------|
| Coupled Oscillators (10D) | 1.148 | 21% |
| Van der Pol (2D) | 0.430 | 8% |
| Bearing-Only (4D) | 51.68 | 19% |
| Reentry Vehicle (6D) | 236.3 | 36% |

## Technical Details

### FilterMath Dispatch Layer
All SRUKF linear algebra now routes through `Common/include/FilterMath.h`:
- **GEMM**: CUDA cuBLAS → SVE2 cache-blocked → NEON → Eigen
- **Cholesky**: NEON accelerated → Eigen LLT/LDLT fallback
- **Kalman Gain**: SPD solve (O(n²)) instead of explicit inverse (O(n³))
- **Cross-platform**: Non-ARM falls through to Eigen (or CUDA if available)

Every tier above Eigen is **optional**. The dispatch decision is made at compile
time and the Eigen fallback is always present, so SRUKF is correct on any target
with a C++20 compiler and Eigen; the accelerators change speed, not answers.
(In practice SRUKF uses fixed-size Eigen types, which bind to FilterMath's
compile-time `gemm` overload and never reach the runtime dispatch at all.)

> **Note**: On CUDA hosts the CUDA backend is active for SM 75–120, including
> Blackwell RTX 50-series (verified on an RTX 5070 Ti Laptop GPU / SM 120,
> CUDA 13.1.115); on hosts without CUDA the build drops to the next tier
> automatically. Compute kernels come from the OptMathKernels sibling tree — see
> [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for the version and provisioning
> policy, which is deliberately **not** duplicated here.

### Dimension-Adaptive Parameters
```cpp
if (NX <= 5) {
    alpha = 1.0f;
    kappa = 3.0f - static_cast<float>(NX);
} else {
    alpha = 1.0f;
    kappa = 0.0f;
}
beta = 2.0f;  // Optimal for Gaussian
```

### URTS / SR-URTS Full-Interval Smoother (post-`d668a6f`)

`SRUKFSmoother` implements Sarkka, *Bayesian Filtering and Smoothing*
(Cambridge, 2013), Algorithm 10.6 SR-URTS literally; the plain-covariance
sibling `UKFSmoother` (in the same header directory) implements Algorithm 10.1
URTS.  In both cases:

- **Forward pass** stores only the filtered posteriors `(m_k, P_k)` — or
  `(m_k, A_k)` in the SR variant, with `A_k A_kᵀ = P_k`.  No cached
  prediction quantities.
- **Backward pass at each `j`** regenerates sigma points from the filtered
  posterior, re-propagates through `model_.f(·, t_{j+1}, u_{j+1})` (with
  SRUKF's angular-state conventions preserved — circular mean, `std::remainder`
  wrap of deviations), and recomputes the predicted mean `m~`, predicted
  covariance `P~ + Q`, and cross-covariance `D` fresh.
- **SR gain** `G = D · P~⁻¹` in Alg. 10.6 is obtained by two triangular solves
  against the Cholesky factor `A~` — never inverts `P~`, never materialises
  the full predicted covariance.  The by-product `U = D · A~⁻ᵀ` satisfies
  `U Uᵀ = D · P~⁻¹ · Dᵀ`, which feeds the smoothed factor.
- **Smoothed factor** via the identity
  `P_j^s = (A_j A_jᵀ − U Uᵀ) + (G · A_{j+1}^s)(G · A_{j+1}^s)ᵀ`:
  downdate `A_j` column-by-column with `U` to get `A_cond` (a Cholesky factor
  of the Schur-complement conditional covariance `Cov(x_j | x_{j+1})`, PSD
  in exact arithmetic), then QR-compose with `G · A_{j+1}^s`.  A rank-1
  downdate failure (near-singular `P~`) triggers a one-shot full-covariance
  re-Cholesky fallback with the same jitter ladder used elsewhere in the
  SRUKF pipeline.
- **Iteration** (`smooth(n>0)`) re-runs the forward filter from the
  smoothed initial condition while keeping the ORIGINAL prior covariance so
  the data still dominate — the practical iterated URTS.

Fixed-lag smoothers (`UnscentedFixedLagSmoother`, `SRUKFFixedLagSmoother`)
are unchanged — they cache forward-pass predictions on purpose for streaming
efficiency, since recomputing them each backward step would double the
per-epoch compute in real-time operation.

Numerical parity check (drag-ball scenario, nonlinear quadratic drag,
500 steps): pre- vs post-`d668a6f` `SRUKFSmoother` outputs agree to
~1e-6 absolute / ~1e-6 relative on the smoothed state and covariance
trace — float32 roundoff, as expected from a different arithmetic
sequence computing the same mathematical result.  The plain-covariance
`UKFSmoother` output is bit-exact between the two implementations.
Epistemic status: *empirically consistent to tolerance stated*, not a
formal proof of equivalence.

### Numerical Safety Chain
1. Safe Cholesky downdate → returns false on failure
2. Full-covariance fallback → recomputes P from all sigma points
3. **Angular-observation innovation wrap (R32)** → for observables flagged
   `isAngularObservation(j)`, `y_k − ŷ` is wrapped to [−π, π] so an innovation
   straddling the ±π branch cut cannot inject a catastrophic ~2π correction.
4. **Innovation gating + NIS (R33)** → the normalized innovation squared
   NIS = innovᵀ (S_yy S_yyᵀ)⁻¹ innov is compared to a χ² gate: since 2026-11-18
   the χ²_NY upper quantile at tail probability 1e-4 (21.11 at NY = 3, 25.74 at
   NY = 5; was a fixed 25), settable via `setInnovationGateTailProbability(q)` or
   `setInnovationGateChi2(threshold)`. NIS is stored
   and exposed via `getLastNIS()`. On a gate trip the correction is scaled by
   `scale`: down-scaled by `√(gate/NIS)` by default, or **rejected** (`scale = 0`)
   when `setRejectOutliers(true)`.
   - **Consistency (fixed in the v3.2.1 audit):** the covariance downdate stays consistent with
     the gated gain `scale·K` via the Joseph partial-update form — the downdate
     column is scaled by `√(2·scale − scale²)`, i.e. the covariance shrinks by
     `(2·scale − scale²)·K·P_yy·Kᵀ`. Endpoints: `scale = 1` → full update;
     `scale = 0` (rejected outlier) → zero downdate, so a discarded measurement no
     longer shrinks the covariance into false certainty. Previously only the state
     was scaled while the covariance was downdated in full.
5. S_yy validation → `allFinite()` check + diagonal minimum enforcement
6. Multiple Cholesky fallbacks: accelerated → jitter → LDLT → diagonal

## Code Locations

- **FilterMath dispatch**: `Common/include/FilterMath.h`
- **SRUKF core**: `UKF/include/SRUKF.h` (class `SRUKF`)
- **Sigma points**: `UKF/include/SigmaPoints.h`
- **SRUKF fixed-lag smoother**: `UKF/include/SRUKFFixedLagSmoother.h`
  (class `SRUKFFixedLagSmoother`) — the variant exercised by the benchmarks
- **SRUKF full-interval URTS smoother**: `UKF/include/SRUKFSmoother.h`
  (class `SRUKFSmoother`) — batch + iterated, added v3.4.0, rewritten to
  Sarkka Alg. 10.6 SR-URTS in `d668a6f`; CTest `SRUKF_Smoother`.  Plain-
  covariance sibling `UKF/include/UKFSmoother.h` (Alg. 10.1 URTS); CTest
  `UKF_Smoother`
- **Benchmarks**: `Benchmarks/src/run_benchmarks.cpp`

## Future Enhancement Opportunities

### Potter's Square Root Filter / UD Factorization
For systems with extreme conditioning (cond(P) > 1e10), alternatives to Cholesky-based square root:
- UD factorization: U (unit upper triangular) × D (diagonal)
- More robust for very ill-conditioned problems
- Better for fixed-point arithmetic

### Adaptive Regularization
For production systems with unknown dynamics:
- Monitor condition number of innovation covariance
- Dynamically adjust diagonal loading based on trace
- Iterative refinement for Kalman gain
