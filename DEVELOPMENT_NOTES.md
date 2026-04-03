# Development Notes

## CUDA Development Restrictions

**Date**: 2026-04-03

**Restriction**: CUDA support disabled until Ubuntu provides CUDA 13+ in official repositories.

### Issues with CUDA 12.0.140 (Ubuntu)

1. **SM 100 (Blackwell) unsupported**: `nvcc fatal: Unsupported gpu architecture 'compute_100'`
2. **Compiler flag incompatibility**: `ptxas fatal: Unknown option '-expt-relaxed-constexpr'`

### Architecture Targets (when CUDA 13+ available)

- SM 75: Turing (RTX 2080/2070/2060)
- SM 80/86: Ampere (RTX 3090/3080/3070)
- SM 89: Ada Lovelace (RTX 4090/4080/4070)
- SM 90: Hopper (H100)
- SM 100: Blackwell (RTX 5090/5080) - **requires CUDA 13+**

### Current Build Configuration

Building with: `-DCMAKE_CUDA_COMPILER=""` (CUDA disabled)

Active acceleration: Vulkan compute shaders + OpenMP

### Blocked Until CUDA 13+

- All CUDA GPU acceleration (cuBLAS GEMM, GPU particle filter)
- cuSOLVER functions (cholesky, solve, inverse) in OptimizedKernels
- New GPU-accelerated algorithms
- Blackwell architecture support

### CUDA Code Ready (commit 397b2d9)

When CUDA 13+ is available, the following features will be activated:
- cuBLAS GEMM for matrices >= 32x32
- GPU particle filter context
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

---

## Build Verification (April 3, 2026)

### Ubuntu 24.04.4 LTS (x86_64)

**System Info**:
- OS: Ubuntu 24.04.4 LTS (Noble Numbat)
- Kernel: 6.17.0-20-generic
- CUDA: 12.0.140 (disabled due to incompatibilities)
- Vulkan: 1.3.275
- Compiler: GCC 13.3.0

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=""
make -j$(nproc)
```

**Test Results** (all passing):
| Test | Status |
|------|--------|
| EKF | ✓ |
| UKF | ✓ |
| SRUKF | ✓ |
| PKF | ✓ |
| RBPKF | ✓ |
| OptimizedKernels (Vulkan) | ✓ 5/5 |
| OptimizedKernels (Radar) | ✓ 19/19 |
| OptimizedKernels (Platform) | ✓ 9/9 |
| OptimizedKernels (NEON) | Skipped (x86_64) |

---

## Future Work

### When CUDA 13+ Available

1. Re-enable CUDA in CMakeLists.txt (remove `-DCMAKE_CUDA_COMPILER=""`)
2. Add SM 100 (Blackwell) back to architecture list
3. Verify `-expt-relaxed-constexpr` flag compatibility
4. Test cuBLAS GEMM acceleration
5. Test GPU particle filter with N >= 256 particles
6. Benchmark CUDA vs Vulkan particle filter performance

### cuSOLVER Integration (OptimizedKernels)

When OptimizedKernels adds cuSOLVER support:
- GPU Cholesky decomposition
- GPU triangular solve
- GPU matrix inverse

This will enable full GPU acceleration for UKF/SRUKF sigma point operations.

---

## Changelog

### v3.1.0 (April 2026)
- Added CUDA GPU acceleration (FilterMath.h, FilterMathGPU.h, particle_filter_gpu.hpp)
- Disabled CUDA due to Ubuntu 24.04 CUDA 12.0 incompatibilities
- Updated CMakeLists.txt to exclude SM 100 (Blackwell)
- Created DEVELOPMENT_NOTES.md

### v3.0.0 (March 2026)
- FilterMath dispatch layer (SVE2 > NEON > Eigen)
- All filter code using FilterMath dispatch
- Bug fixes: -ffast-math removal, safe Cholesky downdate, RBPKF weight handling
- Full benchmark suite passing
