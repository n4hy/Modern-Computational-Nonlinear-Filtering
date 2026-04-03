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
