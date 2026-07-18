# Development Notes

## Audit Remediation Pass (2026-07-18)

Applied a targeted correctness + test/CI hardening pass on top of v3.2.0.
None of the numerical shifts are visible in the RMSE / NEES benchmark table,
but the failure modes each fix removes are real and would have surfaced under
odd caller behavior (bad P0, ill-conditioned S, GPU init failure, etc.).

### Correctness fixes

- **`UKF/include/UKF.h`** — `update()` covariance step now uses an LLT/LDLT
  recovery ladder with **relative** jitter `max(1e-6, 1e-8·trace(P)/NX)` instead
  of the previous fixed `1e-6·I`. Mirrors the ladder already in `SRUKF.h`.
- **`UKF/include/SRUKF.h`** — `initialize()` **throws `std::runtime_error`** on
  NaN / Inf / asymmetric / non-PSD `P0` (including the condition number of `P0`
  in the message on the not-PSD path) instead of silently degrading to
  `L0 = Identity`. Callers hear about a broken `P0` at init, not eight steps
  later.
- **`UKF/include/SRUKF.h`** — innovation-gate threshold `25.0f` is now a
  member with `setInnovationGateChi2()` / `getInnovationGateChi2()` accessors.
  First gate firing per filter instance logs a single line to `std::clog`.
- **`RBPKF/include/rbpf/rbpf_core.hpp`** — direct `det(S)` fast path removed;
  log-det is always computed via `LDLT` diagonal-product with a `1e-30f` clamp.
  Direct `det` on float32 underflows silently at high condition numbers; the
  LDLT path is uniformly safe.
- **`PKF/include/particle_filter.hpp`** — `get_mean()` and `get_covariance()`
  are no longer `const`; the previous `const_cast` on GPU state has been
  removed. GPU paths are wrapped in `try/catch` — any exception demotes the
  filter to CPU-only for the remainder of its life. `compute_mean_cpu()` and
  `compute_covariance_cpu()` are provided as const-safe alternatives.

### New regression tests

Four new test targets registered with CTest bring the count from **24/24 to 28/28**:

- `UKF/tests/test_ukf_numerical.cpp` — rank-deficient `P0` + well-conditioned
  convergence.
- `UKF/tests/test_srukf_initialize.cpp` — NaN / Inf / asymmetric / non-PSD /
  valid `P0` (throw vs no-throw).
- `PKF/tests/test_particle_const.cpp` — SFINAE `static_assert` that
  `get_mean` / `get_covariance` are **not** const-callable, while
  `compute_mean_cpu` is.
- `RBPKF/tests/test_rbpf_logdet.cpp` — LDLT log-det vs SVD reference across
  cond ∈ {1e2, 1e6, 1e12}.

### Build & CI infrastructure

- Root `CMakeLists.txt` gained `nlf_add_warning_flags(<target>)`:
  `-Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter`, plus GCC-only
  `-Wno-stringop-overread` at both compile and link (works around a well-known
  Eigen AVX-512 LTO false positive). Applied to every executable across
  EKF / UKF / PKF / RBPKF / Benchmarks.
- Root `CMakeLists.txt` gained the `NLF_ENABLE_SANITIZERS` option. When `ON`
  and the build type is `Debug` or `RelWithDebInfo`, ASan + UBSan are
  attached to C++ TUs (CUDA TUs are skipped — nvcc does not consistently
  forward `-fsanitize=…`).
- `.github/workflows/ci.yml` — two-job matrix (Release build + ctest;
  RelWithDebInfo + sanitizers). Both jobs `git clone --branch v0.5.15`
  OptMathKernels into `$HOME` before configuring so that the root
  `FetchContent_Declare(... GIT_REPOSITORY $ENV{HOME}/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)`
  line resolves.

### Verification (2026-07-18, Z890 host, CUDA 13.0, GCC 13.3)

- `cmake -B build-audit-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-audit-release -j && ctest --test-dir build-audit-release --output-on-failure`
  → **28/28 pass**.
- `cmake -B build-audit-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNLF_ENABLE_SANITIZERS=ON && cmake --build build-audit-asan -j && ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest --test-dir build-audit-asan --output-on-failure`
  → **28/28 pass**, no ASan or UBSan hits.
- `./build-audit-release/Benchmarks/run_benchmarks` — all RMSE / NEES / divergence
  numbers match the 2026-05-25 baseline in `FINAL_AUDIT_SUMMARY.md` to 3+
  decimal places (e.g. Coupled Osc UKF RMSE 1.45665 vs 1.457; Bearing SRUKF
  64.1727 vs 64.17; Reentry SRUKF+Smoother 236.858 vs 236.8). The correctness
  fixes ship no behavioural regression.

---

## OptMathKernels Dependency — Release Audit & Pinning Policy

**Date**: 2026-05-25

The compute backends (NEON / SVE2 / Vulkan / CUDA) live in the external
[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
(OptMathKernels) project, consumed via CMake `FetchContent` and dispatched to
through `Common/include/FilterMath.h` / `FilterMathGPU.h`.

### Pinning policy (the "tag/release format going forward")

OptMathKernels now publishes semantic-version release tags (`v0.5.0` … `v0.5.15`).
This project **pins to a specific tag** rather than tracking the moving `main`
branch. The pin lives in one place — `CMakeLists.txt`:

```cmake
set(OPTMATH_RELEASE_TAG "v0.5.15" CACHE STRING "Pinned OptMathKernels release tag")
FetchContent_Declare(OptimizedKernels ... GIT_TAG ${OPTMATH_RELEASE_TAG})
```

Adopting a newer kernel release is a deliberate, audited step:

1. `git -C $HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA fetch --tags`
2. Audit the upstream diff `git diff <old-tag>..<new-tag>` — pay attention to
   anything under `include/` (public API) and the backend `src/` the filters use.
3. Bump `OPTMATH_RELEASE_TAG`, reconfigure, rebuild.
4. `ctest --output-on-failure` (expect 24/24) and run the benchmark suite.
5. Update README/this file, then commit and tag the parent release.

### Audit: v0.5.13 → v0.5.15 (adopted 2026-05-25)

Previous pin tracked `main` (built at `v0.5.13`). Reviewed every commit and the
full `git diff v0.5.13..v0.5.15`:

| Change | File | Impact on this project |
|--------|------|------------------------|
| **Discrete-GPU preference in Vulkan device selection** | `src/vulkan/vulkan_backend.cpp` | Behavioral, positive. `VulkanContext::init()` now scores physical devices (discrete > integrated > virtual > CPU) and requires a compute queue, instead of blindly taking `devices[0]`. On this dual-GPU x86_64 box it now logs and selects the RTX 5070 Ti for Vulkan compute. No API change. |
| Per-source documentation coverage | `src/vulkan/vulkan_backend.cpp`, `src/platform/platform.cpp` | None — comments only. |
| x86_64 dual-GPU benchmark docs + release notes | `README.md` | None — upstream docs. |
| Version bump 0.5.13 → 0.5.15 | `CMakeLists.txt` | None. |

**Public API (`include/optmath/*.hpp`): zero changes** across v0.5.13..v0.5.15,
so every `optmath::` call site in `FilterMath.h`, `FilterMathGPU.h`, and
`particle_filter_gpu.hpp` is unaffected.

**Verification (2026-05-25):** reconfigured at `v0.5.15`, full rebuild, **24/24
CTest pass**, Vulkan tests confirm `[Vulkan] Selected GPU: NVIDIA GeForce RTX
5070 Ti Laptop GPU`, and the benchmark RMSE/NEES figures are numerically
identical to the prior run (the changed kernel path is not on the UKF/SRUKF
CUDA/Eigen benchmark path). Safe to adopt.

---

## CUDA Development Status

**Date**: 2026-05-25

**Status**: CUDA 13.x active for SM 75–120 including Blackwell (verified on
RTX 5070 Ti / SM 120, CUDA 13.1). CUDA 12.x remains supported for SM 75–90.

> Historical note: earlier revisions of this file capped support at SM 90 and
> listed Blackwell (SM 100/SM 120) as "blocked until CUDA 13+". That blocker is
> resolved — see the verification block below and the README CUDA section.

### Supported Architecture Targets

- SM 75: Turing (RTX 2080/2070/2060) — CUDA 12.x / 13.x
- SM 80/86: Ampere (RTX 3090/3080/3070, A100) — CUDA 12.x / 13.x
- SM 89: Ada Lovelace (RTX 4090/4080/4070) — CUDA 12.x / 13.x
- SM 90: Hopper (H100) — CUDA 12.x / 13.x
- **SM 100 / SM 120: Blackwell (RTX 5090/5080/5070) — CUDA 13.x only**

For Blackwell, configure with `-DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON`
(SM 120 is not in the default multi-arch list). On CUDA 12.x, `nvcc` rejects
`compute_100`/`compute_120` with `Unsupported gpu architecture`.

### Current Build Configuration

CUDA auto-detected and enabled. Active acceleration: CUDA (cuBLAS GEMM, GPU
particle filter) + Vulkan compute shaders + OpenMP. OptMathKernels cuSOLVER
Cholesky is available as of upstream v0.5.10 (verified on CUDA 13).

### CUDA Code (commit 397b2d9, active)

- cuBLAS GEMM for matrices >= 32x32
- GPU particle filter context
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

---

## Build Verification (May 25, 2026)

### Ubuntu 26.04 (x86_64) — OptMathKernels v0.5.15

**System Info**:
- OS: Ubuntu 26.04 (x86_64)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, SM 120), driver 595.71.05
- CUDA: 13.1.115 (enabled, SM native / 120)
- Vulkan: 1.4.341 (discrete GPU auto-selected — RTX 5070 Ti)
- Eigen: 3.4.0
- OptMathKernels: pinned **v0.5.15** via `OPTMATH_RELEASE_TAG`

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
make -j$(nproc)
```

**Test Results: 24/24 passing** (8 filter/benchmark + 16 OptMathKernels GPU/SIMD,
incl. `test_cuda_kernels` on the Blackwell GPU and the 4 Vulkan suites selecting
the discrete GPU). Total CTest time ≈ 5.9 s.

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

### v3.2.0 (May 2026)
- Audited OptMathKernels major updates v0.5.13 → v0.5.15 (see "Release Audit" above)
- Adopted tag/release pinning: `OPTMATH_RELEASE_TAG` pins FetchContent to a
  release tag (now `v0.5.15`) instead of tracking `main`
- Picked up upstream Vulkan discrete-GPU preference (RTX 5070 Ti now selected for
  Vulkan compute on this dual-GPU host)
- Refreshed CUDA status to CUDA 13.x / Blackwell SM 120 (resolved former blocker)
- Rebuilt, 24/24 CTest pass, benchmarks rerun (RMSE/NEES unchanged), plots regenerated

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
