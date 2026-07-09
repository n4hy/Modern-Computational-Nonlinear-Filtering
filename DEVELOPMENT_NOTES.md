# Development Notes

## OptMathKernels Dependency — Release Audit & Pinning Policy

**Date**: 2026-05-25

The compute backends (NEON / SVE2 / Vulkan / CUDA) live in the external
[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
(OptMathKernels) project, consumed via CMake `FetchContent` and dispatched to
through `Common/include/FilterMath.h` / `FilterMathGPU.h`.

### Pinning policy (the "tag/release format going forward")

OptMathKernels now publishes semantic-version release tags (`v0.5.0` … `v0.5.17`).
This project **pins to a specific tag** rather than tracking the moving `main`
branch. The pin lives in one place — `CMakeLists.txt`:

```cmake
set(OPTMATH_RELEASE_TAG "v0.5.17" CACHE STRING "Pinned OptMathKernels release tag")
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

### Audit: v0.5.15 → v0.5.17 (adopted 2026-07-08)

Fetched tags and reviewed the full `git diff v0.5.15..v0.5.17` (two upstream
releases). The entire diff touches only 5 files — `CMakeLists.txt`, `README.md`,
`requirements.txt`, `.gitignore`, and `tests/test_neon_linalg.cpp`:

| Change | File | Impact on this project |
|--------|------|------------------------|
| x86_64 desktop RTX 5090 benchmark numbers | `README.md` | None — upstream docs. |
| **NEON `TrsvLower64x64` unit-test tolerance `1e-3 → 5e-3`** | `tests/test_neon_linalg.cpp` | None functional. float32 forward-substitution over 64 rows accumulates ~O(1e-3) round-off; the twin `TrsvUpper64x64` test already used `5e-3`. Removes a latent flaky-test edge; the `neon_trsv_lower` **kernel is unchanged**. |
| Documented apt/build deps (`glslang-tools`/`glslc` for Vulkan SPIR-V shaders) | `requirements.txt` | None — build-doc only; already installed on this host (Vulkan suites compile & pass). |
| Ignore `optenv/` venv and `*.spv` artifacts | `.gitignore` | None. |
| Version bump 0.5.15 → 0.5.17 | `CMakeLists.txt` | None. |

**Public API (`include/`) and compute backends (`src/`): zero changes** across
v0.5.15..v0.5.17 (`git diff --name-only v0.5.15..v0.5.17 -- include/ src/` is
empty). No `optmath::` call site in `FilterMath.h`, `FilterMathGPU.h`, or
`particle_filter_gpu.hpp` is affected. **Note:** the upstream releases contain no
MPI/OpenMPI — the parallelism story here remains OpenMP (CPU) + CUDA/Vulkan (GPU).

**Verification (2026-07-08):** cleared `_deps/optimizedkernels-*`, reconfigured
at `v0.5.17` (dep HEAD `cb4b9ef` = tag `v0.5.17`), full rebuild, **24/24 CTest
pass** (≈ 5.8 s), benchmarks rerun and plots regenerated — RMSE/NEES figures
numerically consistent with the prior run (the changed test path is not on the
UKF/SRUKF CUDA/Eigen benchmark path; only 3 of 15 committed PNGs changed at the
byte level, all cosmetic). Safe to adopt.

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

## Build Verification (July 8, 2026)

### Ubuntu 26.04 LTS (x86_64) — OptMathKernels v0.5.17

**System Info**:
- OS: Ubuntu 26.04 LTS (x86_64)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, SM 120), driver 595.71.05
- CUDA: 13.1.115 (enabled, SM native / 120)
- Vulkan: 1.4.341 (discrete GPU auto-selected — RTX 5070 Ti)
- Eigen: 3.4.0
- OptMathKernels: pinned **v0.5.17** via `OPTMATH_RELEASE_TAG`

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
make -j$(nproc)
```

**Test Results: 24/24 passing** (8 filter/benchmark + 16 OptMathKernels GPU/SIMD,
incl. `test_cuda_kernels` on the Blackwell GPU and the 4 Vulkan suites selecting
the discrete GPU). Total CTest time ≈ 5.9 s.

---

## Before/After Validation Record (July 8, 2026)

Side-by-side validation of the v3.2.1 → v3.2.3 work (kernel bump + the three
audit fixes + optimization #1). **Before** = commit `7609df4` (session start,
OptMathKernels v0.5.15); **After** = commit `2c7dccf` (v0.5.17 + fixes + opt).
The "before" tree was built from a detached `git worktree` at that commit so the
two builds are truly independent. Same host as the Build Verification above.

### Tests — unchanged pass

| | Before | After |
|---|---|---|
| `ctest` (24) | 24/24 | 24/24 |
| under `OMP_NUM_THREADS=24` | 24/24 | 24/24 |

### Benchmark accuracy — unchanged; only the false metric corrected

| Problem / filter | RMSE before → after | NEES% | **Divergences before → after** |
|---|---|---|---|
| CoupledOsc 10D (UKF/SRUKF) | 1.4566 → 1.4566 | 94.5 | 0 → 0 |
| VanDerPol 2D | 0.4681 / 0.4663 → same | 95.9 / 96.0 | 0 → 0 |
| **Bearing-Only 4D (UKF)** | 63.8081 → 63.8084 | 99.6 | **176 → 0** |
| **Bearing-Only 4D (SRUKF)** | 64.1728 → 64.1728 | 99.6 | **175 → 0** |
| Reentry 6D (UKF) | 369.01 → 369.115 | 95.9 | 0 → 0 |
| Reentry 6D (SRUKF) | 369.185 → 369.182 | 95.6 | 0 → 0 |

RMSE/NEES identical to ~4 sig figs. The sub-0.03% wiggles are float
reassociation from the fixed-size `gemm` path (UKF cases) and fix #2 acting on
genuinely-gated Reentry-SRUKF steps — expected, not behavioral. The only real
metric change is Bearing-Only divergences (fix #3): **176/175 → 0**.

### Fix #1 (RBPF OpenMP race) — ThreadSanitizer, definitive

The RBPF test was compiled `-fsanitize=thread -fopenmp` (header-only path, no
CUDA) against both header versions and run at `OMP_NUM_THREADS=8`:

| | Worker-vs-worker races (genuine) | Site |
|---|---|---|
| Before | **~24 reports**, thread-pairs `T3/T4`, `T6/T4`, `T5/T3`, `T6/T3`, `T6/T7`… | `Eigen …/AssignmentFunctors.h:24` — writing the **shared `A/B/Q/H/R`** in `get_dynamics`/`get_observation` |
| After | **0** | — |

The residual After warnings are all *main-thread* post-loop reads: the known
GCC-libgomp barrier false positive (TSan cannot see the `omp parallel for`
barrier), present in any OpenMP+TSan program and unrelated to the fix.

### Fix #2 (SRUKF gated covariance) — deterministic reject-mode harness

Constant-position model, `reject_outliers_ = true`, one gross outlier
(NIS ≈ 8.1e5 ≫ 25 gate) at step 5; no RNG, so the only difference between the
two builds is the header. Covariance trace across the rejected step:

| Version | trace(P): pre-step → post-step | Interpretation |
|---|---|---|
| Before | 5.2547 → **5.2367** (shrinks) | covariance tightened on a *discarded* measurement → false certainty |
| After | 5.2547 → **5.2747** (+0.02 = process noise) | correct: a rejected measurement adds no information |

The before-build stays ~0.006–0.008 over-tight every subsequent step — the
compounding overconfidence the fix removes.

### Optimization #1 (fixed-size dispatch) — min of 5 runs

| Case | Before | After | Δ |
|---|---:|---:|---|
| UKF CoupledOsc 10D | 38.27 ms | 36.91 ms | −3.6% |
| SRUKF 10D | 41.78 ms | 41.64 ms | ~0 (direct Eigen, no `gemm`) |
| RMSE | 1.4566 | 1.4566 | identical |

**Conclusion:** two real defects eliminated with hard evidence (RBPF race: 24
TSan worker-races → 0; SRUKF rejected-outlier covariance shrink: quantified
trace divergence), neither visible in the pass/fail suite beforehand; filter
accuracy unchanged to 4 sig figs; UKF ~3.6% faster and numerically identical.

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

### v3.2.3 (July 2026) — perf: fixed-size dispatch fast path
- **`filtermath::gemm` / `mat_vec_mul` fixed-size overloads.** On x86 every
  filter matrix is small (2..10) and below the CUDA threshold, so these calls
  always fell to Eigen — but the dynamic `MatrixXf` signature forced heap-backed
  temporaries and runtime dispatch branches around each product. Added SFINAE
  overloads (in `Common/include/FilterMath.h`) that bind to compile-time-sized
  operands and compute directly into stack-allocated fixed-size results, with no
  dispatch branch. Genuinely dynamic operands (EKF, the RBPF model dynamics /
  observation matrices) still bind to the `MatrixXf` overloads, so the
  large-matrix CUDA/SVE2/NEON dispatch is fully preserved.
- Tightened one UKF update intermediate (`KS`) to fixed-size so both covariance
  gemms take the fast path.
- Measured: UKF 10D benchmark ~3.6% faster (38.3 → 36.9 ms, min of 5 runs);
  RMSE/NEES bit-identical. SRUKF unchanged (it uses direct Eigen, not gemm).
  Note: EKF and the RBPF per-particle KF use dynamic `MatrixXf` by design and do
  not auto-benefit — converting them is a larger interface refactor (follow-up).

### v3.2.2 (July 2026) — audit fixes
- **RBPF OpenMP data race (correctness).** `rbpf_core.hpp::step()` declared the
  per-particle work matrices `A,B,bias,Q,H,offset,R` outside the
  `#pragma omp parallel for`, so they were shared across threads and clobbered
  mid-`predict`/`update` — nondeterministic wrong results (and possible crash via
  concurrent Eigen realloc). Moved the declarations inside the loop body.
- **SRUKF gated-covariance consistency (correctness).** `SRUKF::update()` applied
  the innovation-gate `scale` only to the state correction while downdating the
  covariance by the full `K·S_yy·S_yyᵀ·Kᵀ`. A rejected outlier (`scale==0`) thus
  shrank the covariance with no state change → false certainty / divergence.
  Downdate now uses `U = scale·(K·S_yy)` (scale² reduction; no-op when rejected).
  Default down-scaling path is numerically unchanged (`scale==1`).
- **Benchmark divergence metric (harness).** Bearing-Only used the default
  `divergence_threshold = 10.0f` against a ~64 m error scale, mislabelling ~65% of
  steps as "divergences" (NEES showed the filter was consistent). Gave it a
  problem-scaled 500 m threshold (analogous to reentry's 5 km); count now 0.

### v3.2.1 (July 2026)
- Audited OptMathKernels bump v0.5.15 → v0.5.17 (see "Release Audit" above);
  upstream diff is docs + a NEON unit-test tolerance fix only — no public API,
  compute-backend, or MPI/OpenMPI change
- Bumped `OPTMATH_RELEASE_TAG` to `v0.5.17`, cleared `_deps` to force re-checkout,
  full rebuild, **24/24 CTest pass**
- Benchmarks rerun and plots regenerated (RMSE/NEES consistent; 3/15 committed
  PNGs changed cosmetically)

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
