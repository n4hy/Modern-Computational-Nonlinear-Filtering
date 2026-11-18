# Development Notes

## Portability Model — Read This First

This is a **portable C++20 / Eigen filtering library**. The filters themselves
(EKF, UKF, SRUKF, PKF, RBPF and their smoothers) contain no architecture-specific
code and build and run correctly on any target with a C++20 compiler and Eigen.

Everything below that tier is an **optional accelerator with an Eigen fallback**:

| Tier | Requires | Selected by | Fallback |
|------|----------|-------------|----------|
| CUDA (cuBLAS GEMM) | NVIDIA GPU + toolkit | auto-detect; `OPTMATH_USE_CUDA` | next tier |
| Vulkan compute | Vulkan SDK/loader | `NLF_ENABLE_VULKAN` (default ON, auto-skipped with no SDK) | next tier |
| SVE2 | ARMv9 CPU that implements it | `NLF_ENABLE_SVE2` (default ON on ARM) | next tier |
| NEON | ARM (AArch64/ARM32) | `NLF_ENABLE_NEON` (default ON on ARM) | Eigen |
| **Eigen** | nothing | always available | — |

`Common/include/FilterMath.h` makes the matching decision at **compile time**
(`FILTERMATH_ARM64`, `FILTERMATH_HAS_SVE2`, `OPTMATH_USE_CUDA`) and falls through
to Eigen, so no tier is ever load-bearing for correctness.

**Two rules for this document:**

1. **Never present one machine as "the" host.** Development and verification have
   happened on at least two very different machines (an aarch64 Raspberry Pi 5 and
   an x86_64 + RTX 5070 Ti workstation); neither is canonical.
2. **Attribute every measured number to the host that produced it.** RMSE/NEES are
   backend-independent and reproduce across hosts to within float reassociation
   (≤0.02% RMSE measured — see "Backend portability" below). **Timings are host-specific and do not transfer.**

---

## OptMathKernels Dependency — Sibling-Directory / Latest-`main` Policy

**Date**: 2026-07-10 (supersedes the earlier FetchContent tag-pinning policy)

The optional NEON / SVE2 / Vulkan / CUDA compute backends live in the external
[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
(OptMathKernels) project, dispatched to through `Common/include/FilterMath.h`.

> The live GPU dispatch path is `Common/include/FilterMath.h` (UKF/SRUKF/EKF
> linear algebra). `PKF/include/particle_filter_gpu.hpp` was **removed on
> 2026-11-16** (see the changelog). The previously-dormant
> `Common/include/FilterMathGPU.h` (`GPUBufferPool`, `GPUSigmaContext<NX>`) was
> **removed on 2026-07-23** — it had never been `#include`d by any translation
> unit across four upstream-audit cycles, so deletion changed no compiled code.

### Provisioning policy (current)

The build **requires the OptMathKernels source tree as a sibling directory**
(`../OptimizedKernelsForRaspberryPi5_NvidiaCUDA`) and builds it directly from its
**local working tree — i.e. latest `main`, not a pinned tag** — via
`add_subdirectory`. This is deliberate co-development: kernel edits are picked up
on the next NLF rebuild without a tag/bump cycle.

Provisioning is enforced in `CMakeLists.txt`:

```cmake
set(OPTMATH_DIR "${CMAKE_SOURCE_DIR}/../OptimizedKernelsForRaspberryPi5_NvidiaCUDA" CACHE PATH ...)
option(AUTO_CLONE_DEPS "clone missing OptMath over HTTPS (latest main)" OFF)
# missing + AUTO_CLONE_DEPS=OFF  -> FATAL_ERROR with instructions
# missing + AUTO_CLONE_DEPS=ON   -> git clone --branch main <https> ${OPTMATH_DIR}
add_subdirectory(${OPTMATH_DIR} ${CMAKE_BINARY_DIR}/optmath-build)
```

`./bootstrap.sh` wraps this: it checks for the sibling directory, interactively
offers to clone it over HTTPS (latest `main`) if absent, then configures + builds.

**Trade-off (reproducibility):** tracking `main` means a given NLF commit is no
longer bound to an exact kernel revision. If you need a reproducible build, check
out a specific OptMathKernels commit/tag in the sibling directory before building
(the working tree is what gets compiled), and record that revision in the release
notes. To adopt/verify newer kernel code:

1. `git -C $HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA pull` (or check out a revision).
2. Rebuild NLF (`./bootstrap.sh` or `cmake --build build`).
3. `ctest --output-on-failure` and run the benchmark suite.
4. Update README/this file, then commit the parent release.

### Audit: v0.5.13 → v0.5.15 (adopted 2026-05-25)

Previous pin tracked `main` (built at `v0.5.13`). Reviewed every commit and the
full `git diff v0.5.13..v0.5.15`:

| Change | File | Impact on this project |
|--------|------|------------------------|
| **Discrete-GPU preference in Vulkan device selection** | `src/vulkan/vulkan_backend.cpp` | Behavioral, positive. `VulkanContext::init()` now scores physical devices (discrete > integrated > virtual > CPU) and requires a compute queue, instead of blindly taking `devices[0]`. On the dual-GPU x86_64 reference host it now logs and selects the RTX 5070 Ti for Vulkan compute. No API change. |
| Per-source documentation coverage | `src/vulkan/vulkan_backend.cpp`, `src/platform/platform.cpp` | None — comments only. |
| x86_64 dual-GPU benchmark docs + release notes | `README.md` | None — upstream docs. |
| Version bump 0.5.13 → 0.5.15 | `CMakeLists.txt` | None. |

**Public API (`include/optmath/*.hpp`): zero changes** across v0.5.13..v0.5.15,
so every `optmath::` call site in `FilterMath.h` and `particle_filter_gpu.hpp` is
unaffected. (An additional never-compiled header, `FilterMathGPU.h`, also
targeted `optmath::cuda::*` at the time of this audit; it was removed on
2026-07-23 — see the note at the top of this file.)

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
| Documented apt/build deps (`glslang-tools`/`glslc` for Vulkan SPIR-V shaders) | `requirements.txt` | None — build-doc only; already installed on the x86_64 reference host (Vulkan suites compile & pass there). |
| Ignore `optenv/` venv and `*.spv` artifacts | `.gitignore` | None. |
| Version bump 0.5.15 → 0.5.17 | `CMakeLists.txt` | None. |

**Public API (`include/`) and compute backends (`src/`): zero changes** across
v0.5.15..v0.5.17 (`git diff --name-only v0.5.15..v0.5.17 -- include/ src/` is
empty). No `optmath::` call site in `FilterMath.h` or `particle_filter_gpu.hpp` is
affected. **Note:** the upstream releases contain no
MPI/OpenMPI — the parallelism story here remains OpenMP (CPU) + CUDA/Vulkan (GPU).

**Verification (2026-07-08):** cleared `_deps/optimizedkernels-*`, reconfigured
at `v0.5.17` (dep HEAD `cb4b9ef` = tag `v0.5.17`), full rebuild, **25/25 CTest
pass** (≈ 5.8 s), benchmarks rerun and plots regenerated — RMSE/NEES figures
numerically consistent with the prior run (the changed test path is not on the
UKF/SRUKF CUDA/Eigen benchmark path; only 3 of 15 committed PNGs changed at the
byte level, all cosmetic). Safe to adopt.

### Audit: v0.5.17 → v0.6.3 (adopted 2026-07-14)

The sibling tree now builds at **v0.6.3**. Unlike the previous two audits this is
a large delta — `git diff --name-only v0.5.17..HEAD` is **39 files**, including six
public `include/optmath/*.hpp` headers and ten `src/neon/*.cpp` backends. The
headline upstream change is a **NEON GEMM rewrite** (`v0.6.3: neon_gemm sent any
small-dimension GEMM down a naive path (20x)`).

**Conclusion: the GEMM rewrite does not affect this project's numerics, and
structurally cannot.** Two independent reasons, either sufficient:

1. **NLF never enters the new blocked path.** `src/neon/neon_gemm_optimized.cpp`
   defines `OPTMATH_GEMM_EIGEN_MAX = 80*80*80` and routes anything with
   `M*N*K < 80³` straight to plain Eigen. NLF's largest *dynamic* GEMM is
   `M*N*K = 216`, four orders of magnitude below the cutoff. Upstream verified the
   handoff bit-exact against Eigen across 102,400 randomized cases.
2. **Most filters never call `neon_gemm` at all.** UKF/SRUKF/RBPF-KF use
   fixed-size Eigen types, which bind to `FilterMath.h`'s compile-time `gemm`
   overload (the v3.3.0 fixed-size dispatch fast-path). Only the dynamic-`MatrixXf`
   sites — EKF and the smoothers — reach the dispatch layer at all.

Consistent with both: benchmark RMSE/NEES at v0.6.3 match the pre-bump figures
(see "Backend portability" below, where the Eigen-only build reproduces them too).

### Post-v0.6.3 sync (adopted 2026-08-02, sibling HEAD 817b643)

The sibling tree has moved a handful of commits past the v0.6.3 tag. `git
log v0.6.3..HEAD --oneline` in the OptMath tree lists eight commits; six are
CI/docs (GitHub Actions workflow, README badge, shader-count doc fix, dynamic
Vulkan ICD path, GEMM cache-blocking docs). Two are substantive: `0f49b74
cuda_radar: drop redundant syncs, fix silent bugs, bound steering launch` and
`817b643 cleanup: drop 5 orphan shaders + dead vars in cuda_ifft`.

**NLF impact: none.** The changed symbols (`cuda_radar_*`, `cuda_ifft_*`, and
the removed Vulkan shaders) are not referenced by any NLF header, source file,
or benchmark — `grep -rn 'cuda_radar\|cuda_ifft\|radar_kernels' Common UKF PKF
RBPKF EKF Benchmarks` returns empty. `FilterMath.h` dispatches only to
`cuda::cuda_gemm`, `sve2::*`, `neon::*`, and `platform::*`, none of which were
touched. Verified on Host A (x86_64, Ubuntu 24.04, GCC 13.3, CUDA 13.0.88, RTX
5090, no NEON): **34/34 CTest pass** and the benchmark regression gate reports
`PASS: 13 rows matched RMSE baselines within 0.5%, NEES-consistent, no
divergences` — identical to the pre-sync figures in SRUKF_STATUS.md.

---

## Backend portability — verified, not assumed

**Date**: 2026-07-14. Host: aarch64 Raspberry Pi 5 (Cortex-A76, 4 cores), GCC 12,
Eigen 3.4.0, OptMathKernels v0.6.3, no CUDA.

To confirm the accelerators are genuinely optional, the suite was built twice on
the same host and compared:

```bash
# accelerated (NEON)
cmake -S . -B build-neon -DCMAKE_BUILD_TYPE=Release
# portable Eigen-only fallback, no native CPU tuning
cmake -S . -B build-portable -DCMAKE_BUILD_TYPE=Release \
      -DNLF_ENABLE_NEON=OFF -DNLF_ENABLE_SVE2=OFF -DNLF_ENABLE_NATIVE_ARCH=OFF
```

Both configure, build and pass: **34/34 CTest with NEON, 33/33 Eigen-only.** The
count differs by exactly one because `test_neon_fp16` — a *dependency* test — is
not registered when NEON is off; no NLF test is lost. (Original 2026-07-14 audit
reported 30/30 vs 29/29 at v3.4.1; v3.4.2 audit-remediation adds four repo tests
— `UKF_Numerical`, `SRUKF_Initialize`, `PKF_ParticleConst`, `RBPF_LogDet` — so
the totals move by +4 on both sides without changing the +/-1 delta. OptMath
was already at v0.6.3 for the July 14 measurement, so its count is unchanged.)

Benchmark accuracy, NEON build → Eigen-only build (same host, same seeds). Values
are as reported by `run_benchmarks` (6 significant figures); "identical" below means
agreement to every printed digit, which is what was checked — not a bit-level
comparison of the raw floats:

| Problem / filter | NEON | Eigen-only | Δ |
|---|---:|---:|---|
| CoupledOsc 10D (UKF & SRUKF) | 1.45666 | 1.45666 | **identical** |
| CoupledOsc 10D smoothed | 1.14767 | 1.14767 | **identical** |
| VanDerPol 2D (UKF) | 0.468053 | 0.468053 | **identical** |
| VanDerPol 2D (SRUKF) | 0.46626 | 0.46626 | **identical** |
| VanDerPol 2D smoothed | 0.429705 | 0.429705 | **identical** |
| Bearing-Only 4D (UKF) | 63.8082 | 63.8084 | 0.0003% |
| Bearing-Only 4D (SRUKF) | 64.1723 | 64.1723 | **identical** |
| Reentry 6D (UKF) | 369.043 | 369.117 | 0.02% |
| Reentry 6D (SRUKF) | 369.192 | 369.189 | 0.0008% |
| Reentry 6D smoothed | 236.785 | 236.817 | 0.014% |

The residual differences — 0.02% at worst — are float **reassociation**: a different
summation order in the accelerated kernels, not a different answer. RMSE/NEES are
therefore backend-independent and may be quoted without naming a host. Timings may
not.

**Cross-architecture check (2026-11-13, x86_64, post-`d668a6f` tree).** Against
the 6-significant-figure table in `Benchmarks/README.md`: largest RMSE deviation
1.0e-4 relative (Reentry SRUKF smoothed, 236.275 vs 236.251); largest median-NEES
deviation 9.3e-4 relative (Reentry SRUKF, 4.98137 vs 4.986); `Pct_In_Bounds`
identical on all 13 rows. Same reassociation class as above; the 0.5% regression
gate passes. *Computed on these seeds only — not a proof of bound for other inputs.*

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

### Build Configuration (CUDA hosts)

CUDA is **auto-detected**: present → enabled, absent → the build silently drops to
the next tier (CMake logs `CUDA not found - GPU acceleration disabled`). There is
nothing to configure either way.

- **On a CUDA host** (e.g. the x86_64 + RTX 5070 Ti reference machine): CUDA
  (cuBLAS GEMM) + OpenMP. (Vulkan is built and tested as OptMath
  infrastructure; no filter dispatches to it since 2026-11-16.)
- **On a non-CUDA host** (e.g. the aarch64 Raspberry Pi 5): NEON + Vulkan/V3D +
  OpenMP, or pure Eigen if the SIMD options are off.

OptMathKernels cuSOLVER Cholesky is available as of upstream v0.5.10 (verified on
CUDA 13) — but see the cuSOLVER follow-up under "Future Work": it is not yet wired
into FilterMath's Cholesky dispatch.

### CUDA Code (commit 397b2d9, active)

- cuBLAS GEMM for matrices >= 32x32 (pinned to full FP32; only after a successful init)
- ~~GPU particle filter context~~ — removed 2026-11-16 (no device compute; slower than CPU)
- Runtime CUDA enable/disable via `filtermath::config::set_cuda_enabled()`

---

## Build Verification (July 8, 2026)

> **Historical record — one host, one dependency version.** GPU test counts and all
> timings below are specific to the x86_64 + RTX 5070 Ti reference machine and do
> not reproduce elsewhere; the RMSE/NEES figures do. For the current numbers see
> "Backend portability" above.

### Ubuntu 26.04 LTS (x86_64) — OptMathKernels v0.5.17

**System Info**:
- OS: Ubuntu 26.04 LTS (x86_64)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, SM 120), driver 595.71.05
- CUDA: 13.1.115 (enabled, SM native / 120)
- Vulkan: 1.4.341 (discrete GPU auto-selected — RTX 5070 Ti)
- Eigen: 3.4.0
- OptMathKernels: **v0.5.17**, pinned via `OPTMATH_RELEASE_TAG` — *a FetchContent
  option that no longer exists; the build now compiles the `OPTMATH_DIR` sibling
  working tree. See the provisioning policy at the top of this file.*

**Build Command**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DOPTMATH_CUDA_NATIVE=ON
make -j$(nproc)
```

**Test Results: 25/25 passing** (9 filter/benchmark + 16 OptMathKernels GPU/SIMD,
incl. `test_cuda_kernels` on the Blackwell GPU and the 4 Vulkan suites selecting
the discrete GPU). Total CTest time ≈ 5.9 s.

> **Current counts (2026-07-22, x86_64 + RTX 5090, CUDA 13.0.88, Vulkan 1.3.275,
> no NEON/SVE2, OptMath v0.6.3): 34/34** — 16 registered by this repo (`add_test`
> in `CMakeLists.txt`) + 18 from the dependency (17 base + `test_cuda_kernels`).
> The repo grew from 9 → 12 tests with `SRUKF_AngularWrap` (v3.3.0) and the three
> smoother regressions (v3.4.0), then 12 → 16 with the four audit-remediation
> regressions (v3.4.2: `UKF_Numerical`, `SRUKF_Initialize`, `PKF_ParticleConst`,
> `RBPF_LogDet`). On the Pi 5 the equivalent is 34/34 with NEON / 33/33
> Eigen-only (the -1 is `test_neon_fp16`, which requires NEON).

---

## Before/After Validation Record (July 8, 2026)

Side-by-side validation of the v3.3.0 feature work (kernel bump + the three
audit fixes + optimization #1). **Before** = commit `7609df4` (session start,
OptMathKernels v0.5.15); **After** = commit `2c7dccf` (v0.5.17 + fixes + opt).
The "before" tree was built from a detached `git worktree` at that commit so the
two builds are truly independent. Same host as the Build Verification above
(x86_64 + RTX 5070 Ti) — the RMSE/NEES/divergence figures below are
backend-independent and reproduce anywhere, but **the millisecond timings in
"Optimization #1" are specific to that host and do not transfer.**

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

### CUDA follow-ups

1. Verify `-expt-relaxed-constexpr` flag compatibility
2. Benchmark Blackwell vs Ampere/Ada Lovelace performance
3. Benchmark CUDA vs Vulkan particle filter performance

*(The former item "add SM 100 (Blackwell) back to the architecture list" is **done**:
`CMakeLists.txt` appends SM 100/120 automatically when `nvcc` is CUDA ≥ 13.)*

### cuSOLVER adoption

Upstream cuSOLVER Cholesky / triangular solve has been available since
OptMathKernels **v0.5.10**, but it is **not yet wired into FilterMath's Cholesky
dispatch** — the gap is on our side, not upstream's. Wiring it up would add:
- GPU Cholesky decomposition
- GPU triangular solve
- GPU matrix inverse

enabling full GPU acceleration for UKF/SRUKF sigma point operations on CUDA hosts.

---

## Changelog

### Unreleased — 2026-11-18: SRUKF gate threshold from a χ² tail probability

**Change.** The SRUKF innovation-gate default is now the χ²_NY upper quantile
at tail probability q = 1e-4 (`SRUKF::kDefaultGateTailProbability`), i.e. the
per-update probability that a model-consistent measurement is gated is the
same for every NY. Derivation: for a consistent filter with Gaussian noise the
whitened innovation S_yy⁻¹ν is N(0, I_NY), so NIS ~ χ²_NY. The former fixed 25
gated such measurements with probability 5.7e-7 at NY = 1, 1.4e-4 at NY = 5,
0.20 at NY = 20 and 0.999 at NY = 50 (computed, scipy), so for large NY it
shrank most valid corrections without notice.

Defaults now: 15.14 (NY = 1), 21.11 (NY = 3), 25.74 (NY = 5), 35.56 (NY = 10),
52.39 (NY = 20). New API: `setInnovationGateTailProbability(q)` and the static
`gateThresholdForTailProbability(q)` (throw `std::invalid_argument` unless
0 < q < 1). `setInnovationGateChi2(threshold)` still sets the threshold
directly. Each instantiation computes its default once (function-local
static).

**New `Common/include/ChiSquared.h`** (`nlf_chi2`): regularized P and Q,
`quantile(p, n)` and `upper_quantile(q, n)`. The upper quantile is solved on Q
directly, so deep tails keep relative precision (solving P = 1 − q loses it to
cancellation). The P solver and its bisection + Newton iteration moved here
unchanged from `Benchmarks/include/BenchmarkRunner.h`, which keeps its float
`chi2_quantile()` wrapper; the benchmark's NEES bounds are identical.

**Verification (computed).** `test_chi2_quantile`: `upper_quantile` matches
scipy 1.18.0 `chi2.isf` to ≤ 2.6e-12 relative for n ∈ {1, 2, 3, 4, 5, 7, 10,
20, 50, 100}, q down to 1e-12 (tolerance 1e-9; the references carry 12
significant figures); `regularized_gamma_q` matches `chi2.sf` down to 4.2e-43;
n = 2 closed form −2 ln q to 1e-12. `UKF_AuditRegressions` §9: default
thresholds for NY ∈ {1, 3, 5, 20}; NY = 1 behaviour with NIS = 14 (passes),
NIS = 20 (gated now, passed under 25; correction scaled by √(15.137/20)), and
q = 1e-6 (threshold 23.93, passes).

**Benchmark effect (computed).** No benchmark row gates any more (the one
Coupled-Oscillator event had NIS 25.7437, below the new NY = 5 threshold
25.7448 — a margin of 4e-5 relative, far above float reassociation). RMSE and
smoothed RMSE are unchanged at full CSV precision on all 13 rows; SRUKF
Coupled-Oscillator median NEES moves 9.89188 → 9.89187. Baselines unchanged;
the gate passes.

### Unreleased — 2026-11-16: host-A timing re-measured; SRUKF gate warning once per process

**Host-A timing (README table, `docs/images/benchmark_timing_comparison.png`).**
Re-measured with the warm-up + 5-rep harness. Reported value = median of 5
full-suite runs, each pinned to P-core 7 (`taskset -c 7`) on a Core Ultra 9
285K (8 P + 16 E cores, `powersave` governor). The host was **loaded**: other
jobs kept every core at 55–70% utilisation (2 s `/proc/stat` sample), so the
figures may be pessimistic. *Computed:* run-to-run spread ≤ 1.4% on Coupled
Oscillator and Reentry rows, 3–7% on Bearing-Only and Van der Pol, 16% on Van
der Pol UKF (0.28 µs/step). The within-run rep spread (< 2%) understates this:
two unpinned runs 20 min apart differed by up to 13%. Accuracy columns were
byte-identical across all 5 runs. Versus the old-harness figures, filter-only
rows are 1–27% faster and smoother rows 7–27% slower; the two sets differ in
harness, code version and load, so the change is not attributable to any one of
them.

**SRUKF innovation-gate warning (`UKF/include/SRUKF.h`).** The "gate active"
message was guarded by a per-instance flag, so the benchmark printed it 7 times
per Coupled Oscillator SRUKF row (accuracy pass + warm-up + 5 reps, all
identical data). The flag is now a process-wide `std::atomic<bool>`
(`UKFCore::detail::srukf_gate_warned`); per-filter counts remain in
`getGatedCount()`. The gate event itself is **not** an anomaly. *Computed:*
on Coupled Oscillators (NY = 5, 5001 updates) the gate fires once (NIS 25.74 at
step 1959, correction scaled by 0.985); mean NIS is 5.080 against 5 (H0 sd
0.045). If NIS ~ χ²₅ independently, the expected number of exceedances of 25 is
5001 · P(χ²₅ > 25) = 5001 · 1.39e-4 = 0.70, with P(≥ 1) = 0.50. So the gate
down-weights a measurement that is consistent with the model; it is the
dimension-blind default threshold that does this, not an outlier. No other
problem gates (max NIS: Van der Pol 14.2, Bearing 11.8, Reentry 15.2).

Corrected the `setInnovationGateChi2` doc comment: it called 25 "roughly the
3-sigma value for NY = 3". The 99.73% χ² quantiles are 9.00 (NY = 1), 14.16
(NY = 3), 18.21 (NY = 5) (computed, scipy); 25 is a per-update false-gate rate
of 5.7e-7, 1.5e-5, 1.4e-4 for those NY. The default (25) was left unchanged
in this entry; it became a dimension-aware χ² quantile on 2026-11-18 (entry
above).

### Unreleased — 2026-11-16: true iterated smoothers, RBPF smoothing, PKF GPU removed

Resolves open items (a)–(c) of the 2026-11-15 audit.

**(a) Iterated smoothers.** `smooth(n > 0)` no longer re-runs the forward
filter from the smoothed x₀ (data counted twice). Each iteration now
relinearises about the previous smoothed estimate and solves the resulting
affine model exactly from the ORIGINAL prior, every measurement once
(`Common/include/AffineRTS.h`, double precision):
- **EKF** — Gauss-Newton IEKS (Bell 1994, SIAM J. Optim. 4): Jacobians at the
  previous smoothed means. A fixed point is a stationary point of the MAP
  cost (derived).
- **UKF / SRUKF** — iterated posterior linearisation smoother (IPLS;
  García-Fernández, Svensson, Särkkä, IEEE TAC 62(4), 2017): sigma-point
  statistical linear regression of f and h w.r.t. the smoothed posteriors,
  shared code in `UKF/include/PosteriorLinearization.h`. The SRUKF draws its
  sigma points from the smoothed Cholesky factors and re-factors the double
  covariances. Ω/Λ (SLR residual covariances) are clipped to PSD, since with
  Wc₀ < 0 the sigma-point joint covariance need not be PSD.
- `smooth(n)` is idempotent and leaves the forward filtered pass untouched.

Measured (`Iterated_Smoothers`, computed, float filters vs double reference):
on a linear model with an initial observation and a missed measurement, every
n ∈ {0, 1, 3, 10} matches exact RTS to ≤ 8.8e-6σ (means) and ≤ 9.4e-6
(relative covariance). The replaced code was 2.9e-2σ (UKF/SRUKF) / 3.1e-2σ
(EKF) off after one iteration on the same problem. IEKS MAP gradient relative
to the largest term: 9.5e-2 (EKS) → 2.6e-5 (30 iterations). IPLS: change per
iteration 2.6e-2σ (1→2) → 4.7e-6σ (20→21); UKF vs SRUKF 2.2e-4σ. Existing
smoother tests: iterated RMSE 0.644 / 0.643 / 0.643 m (EKF/UKF/SRUKF) vs
single-pass 0.717 / 0.675 / 0.680 m.

**UKF URTS angular fix (found by the new test).** The `UKFSmoother` backward
pass used linear means and unwrapped deviations for angular states. Rotating a
bearing problem across ±π moved its smoothed bearing by up to 1.4σ, and
smoothed angles left [−π, π]. It now uses a circular predicted mean and
wrapped deviations, as `UKF::predict` and `SRUKFSmoother` do: rotation
invariance 1.25e-5σ.

**(b) RBPF fixed-lag smoothing.** `get_smoothed_mean` now runs an RTS pass back
along each particle's ancestry (filtered moments and the transition into each
epoch are stored with the particles), giving the exact conditional smoothed
linear mean given the nonlinear path (derived), then weights by the current
weights. `RBPF_Smoother`: on a fixed nonlinear path it equals an independent
double KF + RTS to 1.35e-5σ (the old filtered-at-k−lag answer: 2.25σ off). On
random paths with resampling (20 runs × 21 epochs), lag-5 RMSE 0.446 vs
0.764 for the filtered estimate at the same epoch. The per-particle passes run
under OpenMP from N = 2048 and are summed serially in particle order.

**(c) PKF GPU removed.** `particle_filter_gpu.hpp` (CUDA `GPUParticleContext`)
and the aarch64 Vulkan noise-add are deleted. Neither ran the model on the
device — `propagate` / `observation_loglik` are virtual host calls — so both
were host loops plus copies, and the CUDA path was slower at every N
measured (N = 200k: 427 vs 226 ms per 50 steps). API changes: the
`ParticleFilter` constructor loses its 4th `enable_gpu` argument;
`is_gpu_active`, `sync_to_gpu`, `sync_from_gpu` are gone; `get_mean`,
`get_covariance` (and `ParticleFixedLagSmoother::get_filtered_*`) are const
again. `PKF_ParticleConst` now asserts const-callability.

**CTests:** + `Iterated_Smoothers`, `RBPF_Smoother` → 21 filter tests;
39/39 on this host (21 + 18 OptMath).

### Unreleased — 2026-11-15 full-repo audit: correctness fixes across all modules

Five module audits (UKF, SRUKF, EKF/FilterMath, PKF, RBPF). Every fix below was
checked against a failing case first: either the auditor's reproducer, re-run
before and after the fix, or a derivation. Each one is pinned by a CTest.
Results are *computed on the stated inputs*, not proofs of general bounds.

**Tests that could not fail.** `test_ukf_numerical`, `test_srukf_initialize`,
`test_particle_const` and `test_rbpf_logdet` used `assert()`. Release builds
define `-DNDEBUG`, so their 21 checks were compiled out. They now use the new
`NLF_REQUIRE` macro in `TestCheck.h`. All four pass with the checks live.
`test_rbpf_logdet` also tested an in-file copy of the algorithm. It now calls
the library's `predictive_log_likelihood`.

**UKF / sigma points**
- **`Wc0` under the spread floor** (`SigmaPoints.h`). The default α = 1e-3
  trips the n+λ < 0.5 floor. That floor rewrote κ but kept α in `Wc0`, giving
  `Wc0` ≈ 3 instead of β = 2. *Derived:* for x ~ N(0,1) the UT returns
  var(x²) = α²κ + β, so the old floor gave 3 against an exact 2. It now raises
  α (α² = n/(n+κ)) instead of κ, and α = 0 no longer produces NaN. Plain-UKF
  benchmark RMSE moved ≤0.067% (Reentry 366.868 → 367.115); baselines updated.
  SRUKF uses α = 1 and is unaffected.
- **LDLT fallbacks ignored Eigen's pivoting.** Both `SigmaPoints.h` and the
  UKF update factored P A Pᵀ, which swapped the variances of
  [[1e4, 2e4], [2e4, 4e4]]. New helper: `ldlt_sqrt_factor`.
- **Last-resort clamp aliasing** (`UKF.h`). `P_ = P_.diagonal()…asDiagonal()`
  zeroes `P_` before reading it, so the result was 1e-10·I. The diagonal is
  now copied out first.
- **Angular models.** The UKF ignored `isAngularState` and
  `isAngularObservation`; it now uses circular means and wrapped deviations,
  as the SRUKF does. The SRUKF itself used a linear ŷ and an unwrapped P_yy:
  a bearing straddling ±π gave P+ = 0.240 against 0.0345 for the same problem
  rotated by π. Both filters are now invariant to the branch cut.
- **Non-finite measurements.** `UKF`, `SRUKF` and `EKF` `update()` now treat
  a non-finite y as a missed measurement. Previously the state became NaN.
- **SRUKF κ.** The κ = −1 "unset" sentinel overwrote a caller's κ = −1
  together with their α and β. The sentinel is now NaN, and only κ is
  auto-chosen.

**Smoothers (UKF, SRUKF, EKF)**
- **API change: full-interval time convention.** `UKFSmoother` and
  `SRUKFSmoother` passed the measurement time to f. They now take
  `step(t_prev, t_k, y, u)` and gain `observe_initial(t0, y0)`, matching the
  fixed-lag smoothers (commit 3dbbf4c). Test case: f = x + t gave filtered
  values off by up to 2.4; it is now exact. `StateSpaceModel.h` documented the
  opposite convention; the comment is corrected.
- **Non-finite guards rolled nothing back.** They skipped the history entry but
  left the filter NaN, so every later step was silently lost. Smoothers now
  restore the pre-step estimate and throw `std::runtime_error`. Accessors are
  bounds-checked (`std::out_of_range`).
- **Fixed-lag re-initialisation.** `initialize()` did not clear the smoothed
  buffer, so the previous run's estimates were returned (and a fresh object
  read an empty vector).

**FilterMath (CUDA dispatch)**
- **Failed CUDA init.** `optmath::cuda::is_available()` only counts devices.
  When `init()` failed, `cuda_mat_mul` ignored the failure and returned an
  unwritten buffer: `gemm(I, 2·ones)` gave 0. A new `detail::cuda_ready()`
  requires `init()` to succeed and is a thread-safe one-time check.
- **TF32.** OptMath enables `CUBLAS_TF32_TENSOR_OP_MATH` on SM ≥ 8.0. That
  made gemm relative error ~2.9e-4 instead of 1.4e-7, and F·P·Fᵀ of an SPD P
  came back indefinite (minimum eigenvalue −6.7e-5 against +1.0e-6). The
  precision is now pinned to FP32.
- **`kalman_gain` with non-finite S** fell into an SVD path with a
  NaN-derived rank: an Eigen assertion in Debug, out-of-bounds access in
  Release. It now returns a zero gain.

**PKF**
- **GPU-context moments read a stale buffer.** `get_mean()` returned 0
  before the first step (exact 0.4995) and the old posterior after an all-dead
  reset. Weights are now derived from the current log-weights.
- **Resampling precision.**
  - GPU resamplers: their float CDF and offsets are now double. At N = 4e6 the
    last particle had been copied 129,939 times.
  - Resampling now always uses the double host path. GPU log-weights are float,
    and float(log w) shifts strata enough that 711k of 4M particles got no
    offspring.
  - Float `u` in the float-weight overloads of PKF and RBPF is now double:
    7.7% of particles were lost at N = 4e6, now 0.5%.
- **Normalisation precision.** Normalisation no longer round-trips double
  log-weights through float.
- **Per-thread RNG overflow (PKF and RBPF).** The RNG pools were sized once
  from `omp_get_max_threads()`. A later `omp_set_num_threads(8)` indexed past
  the end, writing out of bounds. The pools now grow before each parallel
  region.

**RBPF**
- **The "bad S" sentinel gave the bad particle all the weight.** It set
  `log_det = -1e10`, so log_lik ≈ +5e9. Test case: 10% of particles with a
  rounding-indefinite S took weight 23 and pulled the mean from 10 to 204.
  S is now symmetrised and factored by LLT with a jitter ladder; if that still
  fails, the particle gets −inf.
- **NaN log-weights survived normalisation.** ESS became NaN, the filter never
  resampled again, and the means were NaN. Non-finite log-weights now become
  −inf, and zero-weight particles are skipped in the means (0·NaN = NaN).
- **`can_smooth(lag)` accepted lag > fixed_lag** and returned an overwritten
  epoch (lag 5 gave t = 9). It now rejects any lag outside [0, fixed_lag].
- **`B·u` was dropped whenever Nnl ≠ Nlin.** B is now sized Nlin × Nnl.

**PKF optimisation (measured, x86_64 Core Ultra 9 285K, 24 threads, default
libgomp passive wait; ms per 50 steps, before → after):**
- **`step` fused.** Propagate, noise and weight now run in one parallel region
  instead of three, and the loop is serial below 2048 particles. Measured:
  N = 1000: 12–15 → 2.1; N = 16k: 13–17 → 7–9; N = 64k: 52–59 → 26–28;
  N = 256k: 143–153 → 107–110. The cause, verified by switching the wait
  policy: each region re-wakes sleeping workers at about 0.7 ms per step,
  whatever N is; with `OMP_WAIT_POLICY=active` the overhead disappears.
  N = 4k is within run-to-run noise (5.8–6.7 → 6.6–8.6).
- **Mean and covariance sums in parallel** from 16384 particles, with per-thread
  partials added in a fixed order. Covariance: N = 64k 41 → 14, N = 256k
  161 → 53.
- **End to end** (`step` + resample + moments, CPU path): N = 20k 41.7 → 21.2
  with a bit-identical checksum. N = 200k 331 → 226, where the checksum moves
  9e-6 relative from summation order. Below 2048 particles the RNG stream
  differs, because the serial path uses thread 0's generator; it is still
  deterministic.
- **The GPU-context path** is slower than the CPU path at every N measured
  (N = 2k 10.6 vs 6.2, 20k 44.6 vs 21.2, 200k 427 vs 226). See open item (c).

**Warnings:** the build is now free of warnings (the `-Wshadow` `alpha` in
`SRUKF.h`, unused parameters in `StateSpaceModel.h`, and `-Wcomment` from stale
build snippets that pointed at a removed `_deps` path).

**New CTests:** `UKF_AuditRegressions` and `PKF_AuditRegressions`, plus
lag-bound checks in `RBPF_Basic`. The suite is now 37/37.

**Open at the time (all three resolved 2026-11-16, see the entry above):**
- **(a) The iterated smoothers** (`smooth(n > 0)` in EKF, UKF and SRUKF)
  restart the forward pass from the smoothed x₀ with the original P₀. That
  counts the data twice. On a linear-Gaussian model the answer moves away from
  RTS (up to 0.45σ in the EKF test), where a true IEKS would leave it
  unchanged.
- **(b) RBPF `get_smoothed_mean` returns the filtered linear state** of the
  ancestor. There is no backward Kalman pass: 0 where RTS gives 0.952.
- **(c) The PKF GPU context does no GPU computation** for normalise, resample
  or moments. Each is a host loop plus PCIe copies.

### Unreleased — benchmark timing harness: warmup, repetitions, clean timed region

`run_benchmarks` now measures accuracy and timing in separate passes. The
accuracy pass runs once, untimed, and records states/covariances. The timing
pass rebuilds the filter and replays the same step function 1 warmup + N times
(N = 5, `NLF_BENCH_REPS`), with one `steady_clock` pair around each whole step
loop. Before this the clock was read twice per step inside the measured region,
the total included per-step `MatrixXf` recording, and there was one cold run.
`Total_Time_ms` is now the median rep and `Avg_Step_Time_ms` = median / steps.
New CSV columns (appended, so readers that index by name are unaffected):
`Timing_Reps`, `Step_Time_Min_ms`, `Step_Time_Max_ms`. New gate check: every
timed run reproduces the accuracy run's final state bit-for-bit (passes on all
13 rows).

Measured effect, x86_64 host, old harness vs new (one run each): per-step time
moved −1.7% to −11.4% on 12 rows and +2.2% on SRUKF+Smoother/Van der Pol. The
largest drops are on the smallest steps (Van der Pol SRUKF, Bearing-Only UKF,
0.25–0.41 µs). Two consecutive new-harness runs agreed within 3.2% on every row.
*Computed on one host. This does not separate the clock-read cost from the
warmup effect.* Suite wall time 0.64 s → 3.46 s.

Accuracy side effect: UKF/Reentry RMSE 366.880 → 366.869 (3.0e-5 relative),
median NEES 4.98635 → 4.9857. Every other row is identical. Cause, verified:
GCC's default `-ffp-contract=fast` under `-std=gnu++20`. Moving the loop body
into a lambda changed which multiply-adds get fused. With `-ffp-contract=off`,
old and new sources produce identical output on all 13 rows. Turning contraction
off also shifts results (UKF/Reentry 367.006), so FMA contraction is a
cross-host variation source of the same order as reassociation. The 0.5%
gate is unaffected.

### Unreleased — exact chi-squared bounds for the NEES band

`Benchmark::chi2_quantile()` (`Benchmarks/include/BenchmarkRunner.h`) now returns
the exact quantile — inverse of the regularized lower incomplete gamma, bisection
then Newton in double — instead of the Wilson-Hilferty approximation, whose lower
2.5% quantile was 48% low at n=2 (0.026 vs 0.0506). New CTest `Chi2_Quantile`
(`Benchmarks/tests/test_chi2_quantile.cpp`) pins 28 (n, p) pairs against
`scipy.stats.chi2.ppf` at 1e-5 relative (measured < 3e-7), the closed-form n=2
case, and CDF∘quantile round trips up to n=400. `Pct_In_Bounds` moved: Coupled
Osc 94.49→94.47, Van der Pol 95.83/95.89→94.83/94.78, Bearing-Only 99.63→99.26,
Reentry 95.93→95.56. RMSE and median NEES unchanged; regression gate passes.
Tables in README.md, Benchmarks/README.md and SRUKF_STATUS.md refreshed.
Also: `compute_nees()` read `true_states[0]` before its size check (UB on empty
input); the read now follows the check. No output change on the suite.

### v3.4.2 (2026-07-18) — audit remediation: correctness fixes, regression tests, sanitizer CI

Targeted correctness + test/CI hardening on top of v3.4.1. Numerical output on
the standard benchmark suite is unchanged (RMSE/NEES to 3+ decimal places on
every re-run), but each fix removes a real failure mode that only surfaces
under unusual caller behaviour (bad `P0`, ill-conditioned `S`, GPU init
failure).

**Correctness fixes**

- **`UKF/include/UKF.h`** — `update()` uses a Joseph-style four-term symmetric
  covariance form (`P - K·Pxyᵀ - Pxy·Kᵀ + K·S·Kᵀ`, algebraically identical to
  `P - K·S·Kᵀ` but symmetric by construction) defended by an LLT/LDLT recovery
  ladder with **relative** jitter `max(1e-6, 1e-8·trace(P)/NX)` instead of the
  previous fixed `1e-6·I`. Mirrors the ladder in `SRUKF.h`.
- **`UKF/include/SRUKF.h`** — `initialize()` **throws `std::runtime_error`** on
  NaN / Inf / asymmetric / non-PSD `P0` (including the condition number of `P0`
  in the not-PSD message) instead of silently degrading to `L0 = Identity`.
  Callers hear about a broken `P0` at init, not eight steps later.
- **`UKF/include/SRUKF.h`** — innovation-gate threshold `25.0f` is now a member
  with `setInnovationGateChi2()` / `getInnovationGateChi2()` accessors. First
  gate firing per filter instance logs a single line to `std::clog`; NIS and
  gated count are exposed via `getLastNIS()` / `getGatedCount()`; outlier
  rejection (vs down-scaling) is toggleable via `setRejectOutliers(bool)`.
- **`RBPKF/include/rbpf/rbpf_core.hpp`** — log-det is always computed via
  `LDLT` diagonal-product with a `1e-30f` clamp. Direct `det(S)` on float32
  underflows silently at high condition numbers; the LDLT path is uniformly
  safe.
- **`PKF/include/particle_filter.hpp`** — `get_mean()` and `get_covariance()`
  are no longer `const`; the previous `const_cast` on GPU state has been
  removed. GPU paths are wrapped in `try/catch` — any exception demotes the
  filter to CPU-only for the remainder of its life. `compute_mean_cpu()` and
  `compute_covariance_cpu()` are provided as const-safe alternatives.

**New regression tests** (registered with CTest):

- `UKF/tests/test_ukf_numerical.cpp` — rank-deficient `P0` + well-conditioned
  covariance-contraction check.
- `UKF/tests/test_srukf_initialize.cpp` — NaN / Inf / asymmetric / non-PSD /
  valid `P0` (throw vs no-throw).
- `PKF/tests/test_particle_const.cpp` — SFINAE `static_assert` that
  `get_mean` / `get_covariance` are **not** const-callable, while
  `compute_mean_cpu` is.
- `RBPKF/tests/test_rbpf_logdet.cpp` — LDLT log-det vs SVD reference across
  cond ∈ {1e2, 1e6, 1e12}.

**Build & CI infrastructure**

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
  RelWithDebInfo + sanitizers). Both jobs pre-clone OptMathKernels
  v0.5.15 to `$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA` and pass
  `-DOPTMATH_DIR=$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA`
  `-DNLF_BUILD_PYTHON_VENV=OFF` to CMake, matching the sibling-directory
  provisioning contract in the root `CMakeLists.txt`.

Full suite passes in Release and under ASan+UBSan on the development host;
benchmark accuracy is unchanged from v3.4.1.

### v3.4.1 (July 2026) — build portability, honest metrics, test hardening

**Build portability** (`c8d2905`). The build had no architecture detection at all:
it force-set `ENABLE_NEON`/`ENABLE_SVE2` ON unconditionally — ARM-only instruction
sets requested on every target including x86 — and hardcoded `-march=native`, which
is an x86 spelling that aarch64 toolchains and Apple clang reject and MSVC has no
equivalent for.

- **Target-architecture detection** via `CMAKE_SYSTEM_PROCESSOR` (respects
  cross-compilation, unlike probing the build host).
- **New options**, so the user's choice is what propagates to the dependency:
  `NLF_ENABLE_NEON`, `NLF_ENABLE_SVE2` (default ON **only on ARM targets**),
  `NLF_ENABLE_VULKAN` (default ON; auto-skipped when no SDK is found),
  `NLF_ENABLE_NATIVE_ARCH` (default ON; **auto-OFF when cross-compiling**, where
  "native" would mean the build host and emit instructions the target cannot run —
  also turn it OFF for redistributable binaries).
- **`-march=native` removed.** The native-tuning flag is now *probed* with
  `check_cxx_compiler_flag`: `-mcpu=native` on ARM, `-march=native` on x86, `/O2`
  on MSVC, and portable codegen when none is accepted — an unknown toolchain
  degrades instead of failing to configure.
- **Release flags moved to `$<CONFIG:Release>` generator expressions.** The old
  `if(CMAKE_BUILD_TYPE STREQUAL "Release")` form silently no-opped twice: on a
  first configure from an empty cache, and under *every* multi-config generator
  (Ninja Multi-Config, Visual Studio, Xcode), where `CMAKE_BUILD_TYPE` is empty by
  design. Both produced binaries with no `-O3`/native-tuning/LTO while
  `CMakeCache.txt` read `Release`. An explicit `-DCMAKE_BUILD_TYPE=Release` is
  still good practice but is **no longer load-bearing** for the compile flags.
  `enable_lto_if_supported()` had the same defect and is fixed the same way: it now
  sets the per-config `INTERPROCEDURAL_OPTIMIZATION_RELEASE` property instead of
  guarding the config-agnostic one behind `STREQUAL`. **Verified 2026-07-14** under
  `-G "Ninja Multi-Config"`: every NLF target (`ekf_test`, `srukf_test`,
  `run_benchmarks`, `rbpf_core`, …) now gets `-flto` in the Release config, and the
  Debug config gets none — previously NLF's targets silently lost LTO under every
  multi-config generator while the dependency's kept it. Single-config builds were
  only passing before by accident, because the dependency's `add_subdirectory`
  defaults `CMAKE_BUILD_TYPE=Release` into the cache before our call sites run.
- **Verified:** Eigen-only, no-native-tuning build compiles and runs, 29/29 CTest
  vs 30/30 with NEON, benchmark accuracy identical to ≤0.02%. See "Backend
  portability" above.

**Stopped publishing fabricated metrics** (`c8d2905`).

- `RMSE_Position` / `RMSE_Velocity` (and their smoothed twins) **removed** from the
  metrics struct and CSV. They were never assigned and were emitted as `0.0` for
  every row of every problem — published as data. `compute_rmse_indices()` is
  retained (uncalled, deliberately) as the real implementation should the columns
  ever be wanted back.
- `compute_convergence_time()` returned `times.back()` when the filter **never**
  converged — a sentinel indistinguishable from "converged on the last step". 6 of
  13 rows were publishing it as a measurement. It now returns NaN, prints
  `did not converge`, and writes an empty CSV cell.

**Smoother test hardening** (`d03511f`). The v3.4.0 smoother tests gated only on
*relative* invariants (`smoothed < filtered`, `trace(P_s) <= trace(P_f)`), which
pass whenever both estimates degrade together; `allFinite()` accepts a negative
variance, and the one-sided trace check read a hugely negative trace as success.
Demonstrated by scaling every `filtermath::gemm` result: `x1.05` (filtered RMSE
1.354), `x1.20` (interior cov trace **−543.898** — non-PSD, impossible) and `x0.80`
(filtered RMSE 22.314, 18x) **all passed**. All three tests now assert absolute
RMSE ceilings on both estimates, a min eigenvalue ≥ −1e-4 on the smoothed
covariance at every timestep, and a two-sided interior trace bound. All three
injected defects now fail with exit=1. Note the **eigenvalue check is what catches
`x1.05`** — an absolute RMSE ceiling alone does not, since `x1.05` moves filtered
RMSE only to 1.354. Ceilings sit ~20% above the clean deterministic values
(seed 12345): EKF 1.207/0.717, UKF and SRUKF 1.218/0.680; measured drift between
`-O1` and `-O3 -march=native` is zero, so the headroom guards future toolchains
rather than observed variance.

**Dependency**: OptMathKernels sibling tree adopted at **v0.6.3** — audited, no
numerical impact (see "Audit: v0.5.17 → v0.6.3").

### v3.4.0 (July 2026)

Full-interval (batch) RTS smoothers with optional iteration, across all three
filter families — the "revisit the whole capture after the pass, as time is
available" refinement that complements the existing fixed-lag smoothers:

- `UKF/include/SRUKFSmoother.h` — square-root unscented RTS smoother.
- `UKF/include/UKFSmoother.h` — plain-covariance unscented RTS smoother.
- `EKF/include/EKFSmoother.h` — EKF RTS smoother (uses the stored transition
  Jacobian F for the smoothing gain).

Each keeps the entire forward trajectory and runs a single backward RTS pass once
all measurements are in (every measurement informs every epoch). `smooth(n>0)`
re-runs the forward filter from the smoothed initial condition — keeping the
original prior so the data still dominate — and re-smooths, re-linearising about a
better trajectory (an iterated smoother; an IEKS for the EKF). Each reuses its
family's existing RTS recursion and history struct plus the NEON-accelerated
`gemm`/`kalman_gain`/`cholesky` path, with non-finite guards.

New CTest regressions `SRUKF_Smoother` / `UKF_Smoother` / `EKF_Smoother` (moving
target, mildly nonlinear observation, off prior) confirm smoothing beats filtering
and iteration helps further:

| filter | filtered → smoothed → iterated RMSE | interior cov trace |
|--------|-------------------------------------|--------------------|
| SRUKF  | 1.22 → 0.68 → 0.66 m                 | 0.89 → 0.13        |
| UKF    | 1.22 → 0.68 → 0.66 m                 | 0.89 → 0.13        |
| EKF    | 1.21 → 0.72 → 0.64 m                 | 0.86 → 0.13        |

All three build and pass under CMake; no filter/model code changed, so the rest of
the suite is unaffected.

### v3.3.0 (July 2026)

Feature release: SRUKF angular-observation support (R32/R33), the OptMathKernels
v0.5.17 kernel bump, the Bearing-Only divergence-metric fix, and the fixed-size
`filtermath` dispatch fast-path — merged onto the v3.2.1 audit-remediation line.
Verified on x86_64 + RTX 5070 Ti / CUDA 13.1 (**25/25 CTest**, also under
`OMP_NUM_THREADS=24`); benchmark RMSE/NEES unchanged. The RBPF race and the SRUKF
gate/covariance fix were done independently on both branches and reconciled to
`main`'s versions (SRUKF uses the `sqrt(2s - s^2)` Joseph form).

- **SRUKF angular-observation innovation wrap (R32)** + **NIS exposure / reject
  policy (R33)**: angular observation innovations are wrapped to [−π, π]; NIS is
  exposed via `getLastNIS()` and the innovation gate has an optional
  `setRejectOutliers()` reject-vs-downscale policy. Covered by the registered
  `SRUKF_AngularWrap` CTest.
- **OptMathKernels kernel bump v0.5.15 → v0.5.17** (audited per the pinning policy;
  upstream diff is docs + a NEON unit-test-tolerance fix — no API/backend/MPI change).
  See "Audit: v0.5.15 → v0.5.17" above.
- **Bearing-Only divergence-metric fix**: `count_divergences()` used the default
  10.0 threshold against this problem's ~64 m error scale, mislabelling ~65% of
  steps as "divergences" (the filter was consistent — NEES 99.6% in-bounds). Gave
  it a problem-scaled 500 m threshold (like reentry's 5 km); count now 0.
- **filtermath fixed-size dispatch fast-path**: SFINAE `gemm` / `mat_vec_mul`
  overloads bind to compile-time-sized operands and compute into stack-allocated
  fixed-size results (no `MatrixXf` heap temporaries / dispatch branch); dynamic
  operands still take the CUDA/SVE2/NEON path. UKF 10D ~3.6% faster; RMSE/NEES
  bit-identical.

### v3.2.1 (July 2026) — audit remediation

Repository-wide audit (correctness, build/reproducibility, docs). Verified on two
hosts — the aarch64 Raspberry Pi 5 (CPU: NEON → Eigen path; Cortex-A76 has no SVE2,
so SVE2 sources are compiled out and never execute there; no CUDA) and the x86_64 +
RTX 5070 Ti / CUDA 13.1 reference host (25/25 CTest) — all CTest cases pass before
and after; benchmark RMSE/NEES numerically unchanged.

**Correctness**
- **RBPF OpenMP data race (HIGH)**: `A,B,bias,Q,H,offset,R` were declared outside
  the `#pragma omp parallel for` in `rbpf_core.hpp::step()` (shared → concurrent
  writes). Moved inside the loop body (thread-private). Confirmed with
  ThreadSanitizer (~24 worker-vs-worker races → 0).
- **OpenMP RNG ignored the seed**: RBPF and PKF used a `std::random_device`
  `thread_local` in the parallel path, so `config.seed` / `set_seed()` had no
  effect and runs were non-reproducible. Replaced with per-thread `mt19937_64`
  seeded deterministically (SplitMix64 mix of the base seed) + `schedule(static)`.
  Reproducibility confirmed (identical output across runs at 4 threads).
- **RBPF fixed-lag ancestry off-by-one**: `initialize()` stored the initial
  snapshot at logical slot 0 but never advanced `parent_indices_cnt_`, so step 1
  clobbered it and the smoother could never reach the true initial state. The
  counter now advances so the initial state is preserved as an ancestor.
- **UKF covariance update**: switched `P = P - K S K^T` to the symmetric Joseph
  form `P - K Pxy^T - Pxy K^T + K S K^T` (identical in exact arithmetic, PSD-safe
  by construction) and replaced the single fixed regularization bump with an
  escalating, re-checked jitter loop.
- **SRUKF gate/covariance consistency**: when the innovation gate scales the state
  correction by `s < 1`, the covariance downdate is now scaled by `sqrt(2s - s^2)`
  (the Joseph-consistent reduction for a gain `sK`) instead of removing the full
  `K S_yy S_yy^T K^T`, so the reported covariance is no longer over-confident.

**Build / reproducibility**
- Dependency source is now the **public GitHub URL** by default (was a local
  `$HOME` path that broke clean clones), overridable via `-DOPTMATH_REPO`.
  _(Superseded 2026-07-10: `OPTMATH_REPO` no longer exists. The clone URL is now
  `-DOPTMATH_HTTPS_URL` — used only with `-DAUTO_CLONE_DEPS=ON` — and the source
  path is `-DOPTMATH_DIR`.)_
- Tests now use a `CHECK` macro (exits non-zero) instead of `assert()`, which was
  compiled out under `NDEBUG` — CTest was green regardless of correctness in
  Release. Added real numeric + fixed-lag-smoother assertions to the RBPF test.
- `Benchmarks/CMakeLists.txt` guards `find_package(Eigen3)` like the other subdirs
  (was aborting configure when Eigen came from FetchContent).
- Default CUDA arch list adds Blackwell SM 100/120 automatically when nvcc is
  CUDA ≥ 13 (documented host built without manual flags otherwise).
- `install()` rules added for the filter headers (`<prefix>/include/nlf/`).
- Python venv is opt-out (`-DNLF_BUILD_PYTHON_VENV=OFF`) so offline/CI C++ builds
  don't require PyPI.

**Docs**
- Removed README "Numerical Stability" issues describing the deleted GPS/INS /
  AircraftNav simulation (kept the real Eigen-aliasing fix, stripped the fictional
  application narrative). Reconciled the contradictory bearing-only RMSE tables.
  Corrected the Benchmarks/README to the four problems actually run (Lorenz96 is
  present in the header but not in the default suite). Added a host note clarifying
  that GPU test counts/timings are from the x86_64 reference host.

_(The feature-branch additions merged on top of this audit are listed under
v3.3.0 above.)_

### Recommended follow-up: CI

**Landed in v3.4.2** — `.github/workflows/ci.yml` now runs two jobs on
`ubuntu-latest`: a Release lane (`build-and-test`) and a RelWithDebInfo lane
with ASan+UBSan (`sanitizers`). Both pre-clone OptMathKernels **v0.5.15** to
`$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA` (pinned so CI stays
reproducible even as local `main` moves), pass
`-DOPTMATH_DIR="$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA"
-DNLF_BUILD_PYTHON_VENV=OFF`, then build + `ctest --output-on-failure`. On
v0.5.15 both lanes report **32/32** (16 filter + 16 OptMath at that pinned
version). No CUDA on the GitHub runners, so those tiers exercise the
Eigen/Vulkan fallbacks — the intended portability lane.

The **sixteen** repo tests currently enforced are:

```
ctest -R "EKF_Test|UKF_Test|SRUKF_Test|SRUKF_AngularWrap|SRUKF_Smoother|UKF_Smoother|EKF_Smoother|PKF_Test|PKF_Example|RBPF_Basic|RBPF_CTRV|Benchmarks|UKF_Numerical|SRUKF_Initialize|PKF_ParticleConst|RBPF_LogDet"
```

Now that the build is architecture-aware and CI is architecture-portable
(x86_64 runner, `NLF_ENABLE_NEON`/`SVE2` default OFF there), the sustaining
lane already exercises what the pre-v3.4.1 build would have gotten wrong:
Eigen fallback with no NEON/SVE2 and no `-march=native` regressions.
`NLF_ENABLE_NATIVE_ARCH=OFF` is also the right setting for any redistributable
artifact a runner produces.

### v3.2.0 (May 2026)
- Audited OptMathKernels major updates v0.5.13 → v0.5.15 (see "Release Audit" above)
- Adopted tag/release pinning: `OPTMATH_RELEASE_TAG` pins FetchContent to a
  release tag (now `v0.5.15`) instead of tracking `main`
  _(Reverted 2026-07-10 — `OPTMATH_RELEASE_TAG` no longer exists. See
  "Provisioning policy (current)": the build compiles the `OPTMATH_DIR` sibling
  working tree.)_
- Picked up upstream Vulkan discrete-GPU preference (RTX 5070 Ti now selected for
  Vulkan compute on the dual-GPU x86_64 reference host)
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
