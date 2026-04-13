# SRUKF Implementation Status

## Current State (v3.1.0, April 2026)

### All Benchmark Problems Working

| Problem | SRUKF RMSE | NEES median | In 95% bounds | Status |
|---------|-----------|-------------|---------------|--------|
| Coupled Oscillators (10D) | 1.457 | 9.89 | 94.5% | **WORKING** |
| Van der Pol (2D) | 0.466 | 1.14 | 96.0% | **WORKING** |
| Bearing-Only (4D) | 64.17 | 3.77 | 99.6% | **WORKING** |
| Reentry Vehicle (6D) | 369.2 | 4.99 | 95.6% | **WORKING** |

All 4 benchmark problems and all smoother variants pass successfully. 10D coupled oscillators previously failed with NaN — now resolved through:
1. Dimension-adaptive parameters (alpha=1.0, kappa=3-n for NX<=5; kappa=0 for NX>5)
2. Direct P_yy computation instead of QR for S_yy
3. Safe Cholesky downdate with full-covariance fallback
4. Removed global `-ffast-math` which was corrupting NaN guards

### Smoother Results

| Problem | SRUKF+Smoother RMSE | Improvement |
|---------|-------------------|-------------|
| Coupled Oscillators (10D) | 1.148 | 21% |
| Van der Pol (2D) | 0.430 | 8% |
| Bearing-Only (4D) | 52.03 | 19% |
| Reentry Vehicle (6D) | 236.8 | 36% |

## Technical Details

### FilterMath Dispatch Layer (v3.1.0)
All SRUKF linear algebra now routes through `Common/include/FilterMath.h`:
- **GEMM**: CUDA cuBLAS → SVE2 cache-blocked → NEON → Eigen
- **Cholesky**: NEON accelerated → Eigen LLT/LDLT fallback
- **Kalman Gain**: SPD solve (O(n²)) instead of explicit inverse (O(n³))
- **Cross-platform**: Non-ARM falls through to Eigen (or CUDA if available)

> **Note**: CUDA backend active for SM 75–90 (CUDA 12.x). Blackwell SM 100 pending CUDA 13+. See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

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

### Numerical Safety Chain
1. Safe Cholesky downdate → returns false on failure
2. Full-covariance fallback → recomputes P from all sigma points
3. Innovation gating → Mahalanobis distance threshold (chi-squared 25)
4. S_yy validation → `allFinite()` check + diagonal minimum enforcement
5. Multiple Cholesky fallbacks: accelerated → jitter → LDLT → diagonal

## Code Locations

- **FilterMath dispatch**: `Common/include/FilterMath.h`
- **SRUKF core**: `UKF/include/SRUKF.h`
- **Sigma points**: `UKF/include/SigmaPoints.h`
- **SRUKF smoother**: `UKF/include/SRUKFFixedLagSmoother.h`
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
