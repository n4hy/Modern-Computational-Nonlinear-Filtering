# Development Notes

## CUDA Development Status

**Date**: 2026-04-13

**Status**: CUDA 12.x active for SM 75–90. Blackwell (SM 100) blocked until CUDA 13+.

### Supported Architecture Targets (CUDA 12.x)

- SM 75: Turing (RTX 2080/2070/2060)
- SM 80/86: Ampere (RTX 3090/3080/3070, A100)
- SM 89: Ada Lovelace (RTX 4090/4080/4070)
- SM 90: Hopper (H100)

### Blocked Until CUDA 13+

- SM 100: Blackwell (RTX 5090/5080) — `nvcc fatal: Unsupported gpu architecture 'compute_100'`
- cuSOLVER functions (cholesky, solve, inverse) in OptimizedKernels
- New GPU-accelerated algorithms targeting Blackwell

### Current Build Configuration

CUDA auto-detected and enabled. Active acceleration: CUDA + Vulkan compute shaders + OpenMP.

### CUDA Code Ready (commit 397b2d9)

When CUDA 13+ is available, the following features will be activated:
- cuBLAS GEMM for matrices >= 32x32
- GPU particle filter context
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

---

## Build Verification (April 13, 2026)

### Ubuntu 24.04 LTS (x86_64)

**System Info**:
- OS: Ubuntu 24.04 LTS (Noble Numbat)
- Kernel: 6.8.0-107-generic
- CUDA: 12.0.140 (enabled, SM 75–90)
- Vulkan: 1.3.275
- Compiler: GCC 13.3.0
- Shader compiler: glslangValidator (glslang-tools)

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Prerequisites**:
```bash
sudo apt install build-essential cmake libeigen3-dev
sudo apt install python3 python3-pip python3-venv
sudo apt install glslang-tools          # Required for Vulkan shader compilation
sudo apt install vulkan-tools libvulkan-dev  # Optional: Vulkan runtime
```

**Test Results** (24/24 passing):
| Test | Status | Time |
|------|--------|------|
| EKF_Test | ✓ | 0.11s |
| UKF_Test | ✓ | 0.15s |
| SRUKF_Test | ✓ | 1.58s |
| PKF_Test | ✓ | 1.68s |
| PKF_Example | ✓ | 0.28s |
| RBPF_Basic | ✓ | 0.04s |
| RBPF_CTRV | ✓ | 1.57s |
| Benchmarks | ✓ | 1.33s |
| OptimizedKernels (basic) | ✓ | 0.00s |
| OptimizedKernels (Vulkan) | ✓ 4/4 | ~0.9s each |
| OptimizedKernels (Radar) | ✓ 2/2 | 0.05s |
| OptimizedKernels (NEON) | Skipped (x86_64) | — |
| OptimizedKernels (Platform) | ✓ | 0.00s |
| OptimizedKernels (CUDA) | ✓ | 6.90s |

---

## Future Work

### When CUDA 13+ Available

1. Add SM 100 (Blackwell) back to architecture list in CMakeLists.txt
2. Verify `-expt-relaxed-constexpr` flag compatibility
3. Benchmark Blackwell vs Ampere/Ada Lovelace performance
4. Benchmark CUDA vs Vulkan particle filter performance

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
