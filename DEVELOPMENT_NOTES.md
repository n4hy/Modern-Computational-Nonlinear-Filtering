# Development Notes

## Portability Model — Read This First

This is a **portable C++20 / Eigen filtering library**. The filters themselves
(EKF, UKF, SRUKF, PKF, RBPF and their smoothers) contain no architecture-specific
code and build and run correctly on any target with a C++20 compiler and Eigen.

Everything below that tier is an **optional accelerator with an Eigen fallback**:

| Tier | Requires | Selected by | Fallback |
|------|----------|-------------|----------|
| CUDA (cuBLAS GEMM, GPU particle filter) | NVIDIA GPU + toolkit | auto-detect; `OPTMATH_USE_CUDA` | next tier |
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
   backend-independent and reproduce anywhere (verified — see "Backend portability"
   below). **Timings are host-specific and do not transfer.**

---

## OptMathKernels Dependency — Sibling-Directory / Latest-`main` Policy

**Date**: 2026-07-10 (supersedes the earlier FetchContent tag-pinning policy)

The optional NEON / SVE2 / Vulkan / CUDA compute backends live in the external
[OptimizedKernels](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA)
(OptMathKernels) project, dispatched to through `Common/include/FilterMath.h`.

> The live GPU dispatch paths are `Common/include/FilterMath.h` (UKF/SRUKF/EKF
> linear algebra) and `PKF/include/particle_filter_gpu.hpp` (included by
> `PKF/include/particle_filter.hpp`). The previously-dormant
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
  (cuBLAS GEMM, GPU particle filter) + Vulkan compute shaders + OpenMP.
- **On a non-CUDA host** (e.g. the aarch64 Raspberry Pi 5): NEON + Vulkan/V3D +
  OpenMP, or pure Eigen if the SIMD options are off.

OptMathKernels cuSOLVER Cholesky is available as of upstream v0.5.10 (verified on
CUDA 13) — but see the cuSOLVER follow-up under "Future Work": it is not yet wired
into FilterMath's Cholesky dispatch.

### CUDA Code (commit 397b2d9, active)

- cuBLAS GEMM for matrices >= 32x32
- GPU particle filter context
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
