# Project Structure

**Analysis Date:** 2026-07-18
**Project:** Modern-Computational-Nonlinear-Filtering (NLF), version 3.4.0
**Focus:** Computation (project layout, file organization, build system)

## Top-Level Directory Layout

```
Modern-Computational-Nonlinear-Filtering/
+-- CMakeLists.txt              # Root CMake configuration (587 lines)
+-- bootstrap.sh                # Interactive installer (offers to clone deps)
+-- cmake/                      # CMake package config
|   +-- nlfConfig.cmake.in      # find_package(nlf) template
+-- Common/                     # Shared headers (no library target)
|   +-- include/                # StateSpaceModel.h, SystemModel.h, FilterMath.h, ...
+-- EKF/                        # Extended Kalman Filter module
|   +-- CMakeLists.txt
|   +-- include/                # EKF.h, EKFSmoother.h, EKFFixedLag.h, model headers
|   +-- src/                    # EKF.cpp, EKFFixedLag.cpp (non-header code)
|   +-- main.cpp                # EKF example / smoke test (ekf_test)
|   +-- tests/                  # test_ekf_smoother.cpp
+-- UKF/                        # UKF + SRUKF module
|   +-- CMakeLists.txt
|   +-- include/                # UKF.h, SRUKF.h, SigmaPoints.h, smoothers, models
|   +-- main.cpp                # UKF example (ukf_test)
|   +-- main_srukf.cpp          # SRUKF example (srukf_test)
|   +-- tests/                  # 5 regression tests (see Testing below)
+-- PKF/                        # Particle Filter module (header-only lib)
|   +-- CMakeLists.txt
|   +-- include/                # particle_filter.hpp, resampling.hpp, GPU accel
|   +-- src/                    # example_main.cpp (pkf_example)
|   +-- tests/                  # test_particle.cpp, test_particle_const.cpp
+-- RBPKF/                      # Rao-Blackwellized Particle Filter
|   +-- CMakeLists.txt
|   +-- include/rbpf/           # rbpf_core.hpp, kalman_filter.hpp, types.hpp, ...
|   +-- src/                    # resampling.cpp (only non-header source)
|   +-- examples/               # example_rbpf_ctrv.cpp
|   +-- tests/                  # test_rbpf_basic.cpp, test_rbpf_logdet.cpp
+-- Benchmarks/                 # Standardized problem benchmarks
|   +-- CMakeLists.txt
|   +-- README.md               # Benchmark documentation
|   +-- include/                # BenchmarkProblems.h, BenchmarkRunner.h
|   +-- src/                    # run_benchmarks.cpp (766 lines)
+-- scripts/                    # Python plotting helpers
|   +-- plot_benchmarks.py, plot_results.py, plot_optimized.py, ...
+-- docs/                       # Documentation
|   +-- images/                 # Generated figures (allowlisted in .gitignore)
+-- install/                    # Old CMake install output tree
|   +-- include/optmath/        # Installed OptMathKernels headers (third-party)
|   +-- include/{gtest,gmock}/  # GoogleTest (transitive test dep of OptMath)
|   +-- bin/, lib/              # Built binaries and libraries
+-- .github/workflows/          # GitHub Actions CI
|   +-- ci.yml                  # Release + sanitizer lanes
+-- .claude/                    # Claude Code local config (git-ignored)
+-- .gpd/                       # GPD workflow state (this directory)
+-- .mcp.json                   # Local MCP server config (git-ignored)
+-- nlfvenv/, .nlfvenv/         # Python venvs for plotting (git-ignored)
+-- README.md                   # Project overview
+-- DEVELOPMENT_NOTES.md        # Architectural notes for maintainers
+-- COMPARISON_RESULTS.md       # Filter comparison writeup
+-- SRUKF_STATUS.md             # SRUKF development status
+-- AUDIT_2026-07-08.md         # Audit-remediation pass log
+-- LICENSE, requirements.txt   # Legal, Python deps
+-- CSV outputs                 # bearing_*, coupled_osc_*, reentry_*, vanderpol_*, benchmark_results.csv
+-- cuda-keyring_1.1-1_all.deb  # NVIDIA CUDA keyring (installer artifact, not a code dep)
```

## Directory Purposes

**`Common/include/`:** shared abstract interfaces and dispatch layer, no library
target of its own.

- Purpose: definitions used by every filter module.
- Contains: `.h` headers only.
- Key files:
  - `StateSpaceModel.h` -- `template<int NX, int NY> UKFModel::StateSpaceModel`
    abstract base for UKF/SRUKF (virtual `f`, `h`, `Q`, `R`, `isAngularState`,
    `isAngularObservation`).
  - `SystemModel.h` -- dynamic-size `SystemModel` abstract base for EKF (adds
    Jacobians `F`, `H`).
  - `FilterMath.h` -- 504-line hardware-dispatch layer for `gemm`,
    `mat_vec_mul`, `cholesky`, `inverse`, `solve_spd`, `kalman_gain`, etc.
    CUDA > SVE2 > NEON > Eigen fallback chain, with compile-time fixed-size
    fast paths.
  - `TestCheck.h` -- `NLF_CHECK(cond, msg)` assertion macro that survives
    NDEBUG (unlike `assert()`).
  - `FileUtils.h` -- CSV I/O helper `save_to_csv` for the EKF example.

**`EKF/`:** Extended Kalman Filter with Joseph-form updates.

- Purpose: EKF and its smoothers, with the ball-toss and nonlinear-oscillator
  example models.
- Contains: mix of `.h` (interfaces + example models) and `.cpp` (EKF has a
  compiled library rather than being header-only).
- Key files:
  - `include/EKF.h`, `src/EKF.cpp` -- base EKF (dynamic-size `Eigen::VectorXf`
    / `Eigen::MatrixXf`), retains `x_pred_`, `P_pred_` for smoothing.
  - `include/EKFFixedLag.h`, `src/EKFFixedLag.cpp` -- fixed-lag RTS smoother.
  - `include/EKFSmoother.h` -- full-interval batch RTS smoother with iteration
    (IEKS).
  - `include/BallTossModel.h` -- 4D `[px, py, vx, vy]` with gravity on `vy`.
  - `include/NonlinearOscillator.h` -- Duffing-like nonlinear oscillator model
    used by `EKF/main.cpp`.
  - `tests/test_ekf_smoother.cpp` -- full-interval + iterated RTS regression.

**`UKF/`:** Unscented family (UKF, SRUKF) and their smoothers.

- Purpose: sigma-point filters and RTS smoothers.
- Contains: header-only implementations plus two `main*.cpp` drivers.
- Key files:
  - `include/SigmaPoints.h` -- `SigmaPoints<NX>` struct + Merwe generator.
  - `include/UKF.h` -- `template<int NX, int NY> UKFCore::UKF` with full
    covariance and Joseph-form update.
  - `include/SRUKF.h` -- Square-root UKF (696 lines) with QR + cholupdate.
  - `include/UKFSmoother.h`, `include/SRUKFSmoother.h` -- full-interval RTS
    smoothers (batch + iteration).
  - `include/UnscentedFixedLagSmoother.h`,
    `include/SRUKFFixedLagSmoother.h` -- fixed-lag RTS with two-time-step API.
  - `include/DragBallModel.h` -- 4D ballistic-drag model used by UKF/SRUKF
    smoke tests.
  - `main.cpp` (ukf_test), `main_srukf.cpp` (srukf_test) -- CTest smoke tests
    with `NLF_CHECK`-gated RMSE ceilings.
  - `tests/` -- five regression tests (angular wrap, smoother, numerical, init,
    UKF smoother).

**`PKF/`:** Bootstrap particle filter (SIR) with GPU / Vulkan / OpenMP acceleration.

- Purpose: general non-Gaussian particle filter, GPU-capable.
- Contains: header-only `.hpp` files, plus example / test drivers.
- Key files:
  - `include/state_space_model.hpp` -- `template<int NX, int NY>
    PKF::StateSpaceModel` with `propagate`, `observe`, `sample_process_noise`,
    `sample_observation_noise`, `observation_loglik`. State dimensions capped
    at 20 via `static_assert`.
  - `include/particle_filter.hpp` (437 lines) -- `PKF::ParticleFilter<NX, NY>`
    with OpenMP parallel propagation, systematic resampling, Vulkan
    vector-add acceleration.
  - `include/particle_filter_gpu.hpp` -- `GPUParticleContext<NX>` for CUDA
    weight normalization, ESS, resampling, mean/cov reduction.
  - `include/particle_fixed_lag.hpp` -- backward-simulation smoother.
  - `include/resampling.hpp` -- systematic / stratified resampling with
    `double`-accumulated cumulative sums.
  - `include/lorenz63_model.hpp` -- Lorenz-63 chaotic system with
    Student-t (nu=3) measurement noise (heavy-tailed non-Gaussian test case).
  - `include/noise_models.hpp` -- Gaussian and Student-t samplers / log-pdfs.
  - `src/example_main.cpp` -- runnable example (`pkf_example`).
  - `tests/test_particle.cpp`, `test_particle_const.cpp` -- functional +
    const-correctness regression.

**`RBPKF/`:** Rao-Blackwellized particle filter (partitioned linear /
  nonlinear state).

- Purpose: exploit conditional-linear-Gaussian structure by marginalizing
  the linear substate analytically per particle.
- Contains: header library `rbpf_core` (compiled with one `.cpp` for
  `resampling.cpp`).
- Key files:
  - `include/rbpf/types.hpp` -- `template<int N_NL, int N_LIN, int N_Y>
    RbpfTypes` compile-time dim + Eigen matrix typedefs.
  - `include/rbpf/rbpf_config.hpp` -- `RbpfConfig` struct: `num_particles`,
    `resampling_threshold`, `use_systematic_resampling`, `fixed_lag`, `seed`.
  - `include/rbpf/state_space_models.hpp` -- `NonlinearModel<Types>` and
    `ConditionalLinearGaussianModel<Types>` interfaces (KF matrices
    `A, B, Q, H, R, bias, offset` populated per-particle per-step).
  - `include/rbpf/kalman_filter.hpp` -- per-particle
    `LinearKalmanFilter<Types>` (predict + Joseph-form update).
  - `include/rbpf/rbpf_core.hpp` (387 lines) --
    `RaoBlackwellizedParticleFilter<Types, NonlinearModelT, CondLinModelT>`;
    OpenMP over particles; ancestry buffer when `fixed_lag > 0`.
  - `include/rbpf/resampling.hpp` -- RBPKF-specific resampling wrappers.
  - `src/resampling.cpp` -- Kahan-summed resampling (float weights).
  - `examples/example_rbpf_ctrv.cpp` -- constant-turn-rate-and-velocity CTRV
    tracking example.
  - `tests/test_rbpf_basic.cpp`, `test_rbpf_logdet.cpp` -- functional +
    logdet-stability regression.

**`Benchmarks/`:** standardized problem suite and metric harness.

- Purpose: comparable RMSE / NEES / step-time measurement across filters.
- Contains: template problem classes, benchmark runner, results CSV.
- Key files:
  - `include/BenchmarkProblems.h` (470 lines) -- `CoupledOscillators`,
    `Lorenz96`, `VanDerPolDiscontinuous`, `BearingOnlyTracking`,
    `ReentryVehicle` template classes (all subclass
    `UKFModel::StateSpaceModel<NX, NY>`).
  - `include/BenchmarkRunner.h` (431 lines) -- `BenchmarkMetrics` struct;
    RMSE, NEES, timing, convergence-time, divergence counting.
  - `src/run_benchmarks.cpp` (766 lines) -- driver `main()`, emits
    `benchmark_results.csv`. Compiled with `-fno-fast-math` and
    `EIGEN_FAST_MATH=0` for strict IEEE-754 semantics.
  - `README.md` -- benchmark documentation.

**`scripts/`:** Python plotting helpers (post-processing).

- Purpose: consume CSV outputs and produce comparison figures.
- Contains: `.py` scripts (executable), uses pandas + matplotlib.
- Key files:
  - `plot_benchmarks.py` -- comprehensive RMSE / NEES / timing bar charts
    from `benchmark_results.csv`.
  - `simple_plot_benchmarks.py` -- simplified benchmark plots.
  - `plot_results.py`, `plot_optimized.py` -- per-problem trajectory /
    error plots.
  - `pkf_plot_results.py`, `ukf_plot_results.py` -- filter-specific plots.

**`cmake/`:** package configuration templates.

- Purpose: enable `find_package(nlf)` consumers.
- Contains: `nlfConfig.cmake.in` template configured at install time.

**`docs/`:** documentation and generated images.

- Purpose: figures for the README and comparison reports.
- Contains: `images/` (PNG figures allowlisted through `.gitignore`).

**`install/`:** committed install output tree (from a previous `cmake --install`).

- Purpose: pre-built artifacts for downstream consumers who cannot build.
- Contains: `include/optmath/` (installed OptMathKernels headers),
  `include/{gtest,gmock}/` (transitive test dep of OptMath), `lib/`, `bin/`.
- Note: `.gitignore` line 22 ignores `install/` but the current contents were
  committed anyway. Delete-and-rebuild via `cmake --install build` if stale.

**`nlfvenv/`, `.nlfvenv/`:** Python virtual environments for the plotting
scripts. Auto-created by CMake when `NLF_BUILD_PYTHON_VENV=ON` (default).
`.nlfvenv/` is the current active venv; `nlfvenv/` is a legacy artifact.

## Key File Locations

**Filter Implementations (C++ headers, all templated on compile-time dims):**

- `UKF/include/UKF.h` -- UKF (198 lines).
- `UKF/include/SRUKF.h` -- SRUKF (696 lines, largest single file).
- `UKF/include/SigmaPoints.h` -- Merwe sigma point generator + weights.
- `EKF/include/EKF.h` + `EKF/src/EKF.cpp` -- EKF (dynamic-size).
- `PKF/include/particle_filter.hpp` -- bootstrap PF (437 lines).
- `RBPKF/include/rbpf/rbpf_core.hpp` -- Rao-Blackwellized PF (387 lines).

**Smoother Implementations:**

- Full-interval batch RTS: `UKF/include/UKFSmoother.h`,
  `UKF/include/SRUKFSmoother.h`, `EKF/include/EKFSmoother.h`.
- Fixed-lag RTS: `UKF/include/UnscentedFixedLagSmoother.h`,
  `UKF/include/SRUKFFixedLagSmoother.h`, `EKF/include/EKFFixedLag.h`,
  `PKF/include/particle_fixed_lag.hpp`.

**Shared Interfaces:**

- `Common/include/StateSpaceModel.h` -- UKF/SRUKF/Benchmark model interface.
- `Common/include/SystemModel.h` -- EKF model interface (with Jacobians).
- `PKF/include/state_space_model.hpp` -- PKF model interface (non-Gaussian).
- `RBPKF/include/rbpf/state_space_models.hpp` -- RBPKF nonlinear +
  conditional-linear interfaces.

**Linear Algebra Dispatch:**

- `Common/include/FilterMath.h` -- primary dispatch layer, 504 lines.

**Test Utilities:**

- `Common/include/TestCheck.h` -- `NLF_CHECK` macro (survives NDEBUG).
- `Common/include/FileUtils.h` -- CSV export helper.

**Benchmark Problems (all templates in one header):**

- `Benchmarks/include/BenchmarkProblems.h` -- `CoupledOscillators<10, 5>`,
  `Lorenz96<40, 10>`, `VanDerPolDiscontinuous<2, 1>`, `BearingOnlyTracking<4, 1>`
  (or `<4, 2>`), `ReentryVehicle<6, 2>`.

**Data / Results (repo root):**

- `benchmark_results.csv` -- primary aggregate metrics (16 columns).
- `<problem>_ukf.csv`, `<problem>_srukf.csv` -- truth vs filtered trajectories.
- `<problem>_srukf_smooth.csv`, `<problem>_ukf_smooth.csv` -- + smoothed cols.
- Observed problems in CSV outputs: `bearing`, `vanderpol`, `reentry`,
  `coupled_osc`.

**Configuration / Build:**

- `CMakeLists.txt` (587 lines) -- root build system.
- `bootstrap.sh` -- interactive dependency installer.
- `cmake/nlfConfig.cmake.in` -- `find_package(nlf)` template.
- `requirements.txt` -- Python plotting deps (`pandas`, `matplotlib`, `numpy`).
- `.github/workflows/ci.yml` -- CI (Release + sanitizer lanes).
- `.gitignore` -- excludes build dirs, CSVs (except committed), venvs,
  `install/`, `.claude/`, `.mcp.json`, `cuda-keyring*.deb`.
- `.mcp.json` -- MCP server config (git-ignored).

**Documentation (repo root, all Markdown):**

- `README.md` (~50 KB) -- project overview.
- `DEVELOPMENT_NOTES.md` (~38 KB) -- architectural notes for maintainers.
- `COMPARISON_RESULTS.md` (~25 KB) -- filter comparison writeup.
- `SRUKF_STATUS.md` (~6 KB) -- SRUKF development status.
- `AUDIT_2026-07-08.md` (~21 KB) -- audit-remediation pass log (regression
  tests, sanitizer lane, warning uniformity, doc drift).
- `Benchmarks/README.md` (~20 KB) -- benchmark suite documentation.
- `LICENSE` -- MIT (based on file size).

## Module Dependency Graph

```
                    Eigen 3.4  (transitive PUBLIC)
                         |
                   OptMathKernels                        (external sibling repo)
                   (NEON / SVE2 / Vulkan / CUDA)         optional backends
                         |
                   Common/include/FilterMath.h           (dispatch layer)
                         |
   +---------------------+---------------------+---------------------+
   |                     |                     |                     |
   Common/include        Common/include        Common/include        Common/include
   StateSpaceModel.h     SystemModel.h         (PKF: own model)      (RBPKF: own model)
   |                     |                     |                     |
   UKF/  ---+            EKF/                  PKF/                  RBPKF/
   |        |            |                     |                     |
   UKF.h    SRUKF.h      EKF.h                 particle_filter.hpp   rbpf_core.hpp
   |        |            EKFFixedLag.h         resampling.hpp        kalman_filter.hpp
   UKFSmoother.h ...     EKFSmoother.h         particle_filter_gpu   (per-particle KF)
   |        |            |                     particle_fixed_lag    |
   +--------+------------+---------------------+---------------------+
                                |
                        Benchmarks/  (depends on UKF, SRUKF, Common)
                        run_benchmarks.cpp
                                |
                        benchmark_results.csv, *_ukf.csv, *_srukf.csv
                                |
                        scripts/plot_benchmarks.py (Python)
                                |
                        docs/images/*.png
```

**CMake target dependencies (from `install(TARGETS ...)` line 507 and
`target_link_libraries`):**

- `nlf` (INTERFACE, header-only umbrella) links:
  - `Eigen3::Eigen`
  - `OptMathKernels`
  - `rbpf_core` (compiled library from `RBPKF/src/resampling.cpp`)
  - `OpenMP::OpenMP_CXX` (if found)
- `PKF_Lib` (INTERFACE) links:
  - `Eigen3::Eigen`
  - `OpenMP::OpenMP_CXX` (if found)
- Executable targets (`ekf_test`, `ukf_test`, `srukf_test`, `pkf_test`,
  `pkf_example`, `example_rbpf_ctrv`, `run_benchmarks`, all `test_*`) link
  `Eigen3::Eigen` and `OptMathKernels`.

## Naming Conventions

**Files:**

- Filter class headers: uppercase noun matching class name
  (`UKF.h`, `SRUKF.h`, `EKF.h`, `EKFSmoother.h`,
  `UnscentedFixedLagSmoother.h`). Legacy `.h`.
- Model headers (in `EKF/include/`, `UKF/include/`, `PKF/include/`): mixed
  case reflecting model name (`BallTossModel.h`, `DragBallModel.h`,
  `NonlinearOscillator.h`, `lorenz63_model.hpp`).
- PKF and RBPKF use `.hpp` (snake_case) for their newer headers
  (`particle_filter.hpp`, `rbpf_core.hpp`, `state_space_model.hpp`).
- Test executables: `test_<subject>.cpp` under `<Module>/tests/`.
- Example / smoke test driver `main.cpp` at the module root (compiled into
  `<module>_test` targets).

**Namespaces:**

- UKF family: `UKFCore` (implementations), `UKFModel` (model interface).
- PKF: `PKF`, with `PKF::Resampling`, `PKF::gpu`, `PKF::Noise`.
- RBPKF: `rbpf`.
- EKF: no namespace (global scope) -- historical.
- Linear algebra: `filtermath`, with `filtermath::config`.
- Benchmarks: `Benchmark`.

**Variables in code:**

- State: `x_` (member), `x`, `x_pred`, `x_filt`, `x_s` (smoothed).
- Covariance: `P_`, `P_pred`, `P_cross`, `P_filt`, `P_s`.
- Cholesky factor of `P` (SRUKF): `S_`, `S_pred`.
- Model matrices: `Q` (process noise cov), `R` (obs noise cov),
  `F` (state Jacobian), `H` (obs Jacobian), `A`, `B` (RBPKF linear
  dynamics), `K` (Kalman gain), `S` (innovation cov -- RBPKF only, distinct
  from SRUKF's `S`).
- Sigma point weights: `Wm` (mean), `Wc` (covariance).
- Dimensions: `NX` (state), `NY` (observation), `NSIG = 2*NX + 1`, `Nnl`,
  `Nlin`, `Ny` (RBPKF).

**CSV columns:**

- Time: `time` (float32 seconds).
- Truth: `true_x0`, `true_x1`, ...
- Filtered: `filt_x0`, `filt_x1`, ...
- Smoothed: `smooth_x0`, `smooth_x1`, ... (in `*_smooth.csv` files only).
- Measurements (when present): `meas` (scalar) or `mx`, `my` (2D).
- All values are single-precision (matching the `float` C++ path).

**CTest test case names:** CamelCase (e.g., `EKF_Test`, `SRUKF_AngularWrap`,
`PKF_ParticleConst`, `RBPF_LogDet`, `Benchmarks`).

## Where to Add New Content

**New filter class (analogous to a variant UKF):**

- Header: `<Module>/include/<Name>.h` (e.g., `UKF/include/AdditiveUKF.h`).
  Follow the `UKF/include/UKF.h` template class pattern
  (`template<int NX, int NY> class ... { ... };`) inside namespace `UKFCore`
  (for UKF-family variants) or a new namespace.
- Add to `<Module>/CMakeLists.txt` as a new executable target with
  `target_link_libraries(... Eigen3::Eigen OptMathKernels)`, plus
  `enable_lto_if_supported()` and `nlf_add_warning_flags()`.
- Add a test in `<Module>/tests/test_<name>.cpp` using `NLF_CHECK` macros
  from `TestCheck.h`.
- Register in root `CMakeLists.txt` via `add_test(NAME <TestName> COMMAND
  <target>)` in the CTest section (lines 566--586).
- Add to `Benchmarks/src/run_benchmarks.cpp` if you want the new filter
  compared against the reference filters on the standard problem set.

**New benchmark problem:**

- Add a new template class to `Benchmarks/include/BenchmarkProblems.h`
  subclassing `UKFModel::StateSpaceModel<NX, NY>` with concrete `f`, `h`,
  `Q`, `R`. Follow the pattern of `CoupledOscillators` or `VanDerPolDiscontinuous`.
- Wire it into `Benchmarks/src/run_benchmarks.cpp` alongside the existing
  problem invocations.
- No new build target needed -- it links into the existing `run_benchmarks`
  executable.

**New shared numerical primitive (e.g., a new BLAS-like routine):**

- Add to `Common/include/FilterMath.h` in the `filtermath::` namespace,
  following the CUDA > SVE2 > NEON > Eigen dispatch pattern. Provide both a
  dynamic-size `Eigen::MatrixXf` overload and a fixed-size compile-time
  overload (SFINAE-gated on `RowsAtCompileTime != Eigen::Dynamic`) for the
  common case of small filter matrices.
- Feature-gate on `FILTERMATH_HAS_CUDA`, `FILTERMATH_HAS_SVE2`,
  `FILTERMATH_ARM64` and provide the Eigen fallback path.
- Corresponding OptMathKernels function must exist upstream in
  `optmath/{cuda_backend, sve2_kernels, neon_kernels}.hpp`.

**New model (SystemModel or StateSpaceModel):**

- Add to the appropriate `<Module>/include/` (e.g., `UKF/include/MyModel.h`
  next to `DragBallModel.h`).
- Subclass the correct interface (`UKFModel::StateSpaceModel<NX, NY>` for
  UKF/SRUKF, `SystemModel` for EKF, `PKF::StateSpaceModel<NX, NY>` for PKF).
- No CMake changes required -- it's a header consumed by an existing driver.

**New Python plotting script:**

- Add to `scripts/<name>.py`.
- Consume CSV outputs from the repo root. Follow the pandas + matplotlib
  pattern of `scripts/plot_benchmarks.py`.
- Assumes the `.nlfvenv` created by CMake is active (or activate manually via
  `source .nlfvenv/bin/activate`).

**New CI check:**

- Add a job to `.github/workflows/ci.yml`. Both existing jobs pre-clone
  OptMathKernels at pinned tag `v0.5.15` before configuring, so replicate that
  step. Use `-DOPTMATH_DIR=$HOME/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
  -DNLF_BUILD_PYTHON_VENV=OFF` for hermetic builds.

**New audit / status doc:**

- Root-level Markdown following the convention of `AUDIT_2026-07-08.md`,
  `SRUKF_STATUS.md`, `COMPARISON_RESULTS.md`.

## Build and Execution

**Preferred bootstrap (interactive):**

```bash
./bootstrap.sh
# Offers to clone sibling OptMathKernels repo if missing, then:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#                       -DAUTO_CLONE_DEPS=OFF \
#                       -DOPTMATH_DIR="../OptimizedKernelsForRaspberryPi5_NvidiaCUDA"
#   cmake --build build -j$(nproc)
```

Non-interactive (accept all prompts): `FORCE_YES=1 ./bootstrap.sh`.

**Manual configure and build:**

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPTMATH_DIR=/path/to/OptimizedKernelsForRaspberryPi5_NvidiaCUDA \
    -DNLF_BUILD_PYTHON_VENV=OFF   # optional: skip Python venv
cmake --build build -j$(nproc)
```

**Auto-clone dependency (CI-friendly):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAUTO_CLONE_DEPS=ON
```

**Options** (defined in root `CMakeLists.txt`):

| Option | Default | Purpose |
|--------|---------|---------|
| `CMAKE_BUILD_TYPE` | `Release` (auto-forced for single-config) | Release / Debug / RelWithDebInfo |
| `OPTMATH_DIR` | `../OptimizedKernelsForRaspberryPi5_NvidiaCUDA` | Path to required OptMath sibling repo |
| `AUTO_CLONE_DEPS` | `OFF` | Clone OptMathKernels from GitHub if missing |
| `NLF_ENABLE_NATIVE_ARCH` | `ON` (`OFF` for cross-compile) | `-march=native` / `-mcpu=native` |
| `NLF_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan on C++ TUs (Debug/RelWithDebInfo only) |
| `NLF_ENABLE_NEON` | `ON` for ARM targets | Build NEON kernels |
| `NLF_ENABLE_SVE2` | `ON` for ARM targets | Build SVE2 kernels |
| `NLF_ENABLE_VULKAN` | `ON` | Build Vulkan backend (auto-skipped if no SDK) |
| `NLF_BUILD_PYTHON_VENV` | `ON` | Auto-create `.nlfvenv` for plotting |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90` (+ `100;120` for CUDA 13+) | CUDA target GPUs |

**Running the test suite:**

```bash
ctest --test-dir build --output-on-failure
```

**16 registered tests** (root `CMakeLists.txt` lines 569--586):
`EKF_Test`, `UKF_Test`, `SRUKF_Test`, `SRUKF_AngularWrap`, `SRUKF_Smoother`,
`UKF_Smoother`, `EKF_Smoother`, `PKF_Test`, `PKF_Example`, `RBPF_Basic`,
`RBPF_CTRV`, `Benchmarks`, plus audit-remediation regression tests
(`UKF_Numerical`, `SRUKF_Initialize`, `PKF_ParticleConst`, `RBPF_LogDet`).

**Sanitizer build:**

```bash
cmake -S . -B build-asan \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DNLF_ENABLE_SANITIZERS=ON \
    -DOPTMATH_DIR=/path/to/OptimizedKernelsForRaspberryPi5_NvidiaCUDA
cmake --build build-asan -j
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest --test-dir build-asan --output-on-failure
```

**Running benchmarks (produces CSV outputs at CWD):**

```bash
cd <repo-root>       # CSVs land at the repo root, so run from there
build/Benchmarks/run_benchmarks
# Emits benchmark_results.csv, bearing_*.csv, vanderpol_*.csv, coupled_osc_*.csv,
# reentry_*.csv
```

**Generating figures (post-benchmarks):**

```bash
source .nlfvenv/bin/activate    # if NLF_BUILD_PYTHON_VENV was ON
python scripts/plot_benchmarks.py             # comparative bar charts
python scripts/plot_results.py                # per-problem trajectories
python scripts/simple_plot_benchmarks.py      # simplified summary
# Output PNGs land in docs/images/
```

**Install (for downstream consumers):**

```bash
cmake --install build --prefix /usr/local
# Consumer usage:
#   find_package(nlf REQUIRED)
#   target_link_libraries(my_app PRIVATE nlf::nlf)
# Both #include "UKF.h" and #include <nlf/UKF.h> resolve; optmath/*.hpp lands
# at <prefix>/include/optmath/.
```

## Special Directories

**`install/`:**

- Purpose: pre-built artifacts (headers, binaries, libraries) from a previous
  `cmake --install`.
- Generated: Yes (via `cmake --install <build>`).
- Committed: Yes, currently -- despite being listed in `.gitignore` line 22
  the tree was committed at some point.
- Contents: `include/optmath/` (third-party OptMathKernels install output),
  `include/{gtest,gmock}/` (transitive test dep), `bin/`, `lib/`. The
  `optmath/{cuda_backend, cuda_error, neon_kernels, platform, radar_kernels,
  sve2_kernels, vulkan_backend}.hpp` headers here match the public API of
  the external OptMath repo.

**`.nlfvenv/`, `nlfvenv/`:**

- Purpose: Python virtual environment for the plotting scripts.
- Generated: Yes (by CMake at configure time when `NLF_BUILD_PYTHON_VENV=ON`).
- Committed: No (listed in `.gitignore` lines 19--20).
- `.nlfvenv/` is the current name; `nlfvenv/` is legacy. `CMAKE_SOURCE_DIR`
  uses `.nlfvenv/`.

**`.gpd/`:**

- Purpose: GPD (get-physics-done) workflow state (this file lives here).
- Generated: Yes (by GPD tooling).
- Committed: Not observed in `.gitignore`; likely tracked deliberately for
  workflow reproducibility.

**`.claude/`:**

- Purpose: Claude Code local agent config.
- Generated: Yes.
- Committed: No (`.gitignore` line 24).

**Build directories (`build/`, `build_baseline/`, `build_opt/`,
`build_opt_final/`, `build_ekf/`, `build_ukf/`, `build_pkf/`, `build_rbpkf/`):**

- Purpose: out-of-source CMake builds.
- Committed: No (all listed in `.gitignore` lines 1--8).

**`baseline_data/`, `optimized_data/`, `eigen_install/`, `_deps/`, `.cache/`:**

- Purpose: performance-comparison working directories and FetchContent cache.
- Committed: No (`.gitignore` lines 9--11, 18, 21).

## Documentation Set

Primary documentation is entirely Markdown at the repo root:

| File | Purpose |
|------|---------|
| `README.md` | User-facing overview: what filters exist, how to build, quick-start. |
| `DEVELOPMENT_NOTES.md` | Maintainer notes: architectural decisions, CUDA arch defaults, native-tuning rationale. |
| `COMPARISON_RESULTS.md` | Comparative filter analysis (RMSE / NEES / step-time across problems). |
| `SRUKF_STATUS.md` | SRUKF-specific status and known behavior. |
| `AUDIT_2026-07-08.md` | Audit-remediation pass log (regression tests, CI sanitizer lane, warning uniformity). |
| `Benchmarks/README.md` | Benchmark suite details: problems, metrics, invocation. |
| `LICENSE` | License (MIT-style based on file size 1068 bytes). |

`docs/` currently contains only `docs/images/` for generated figures. There is
no HTML / Doxygen output committed; all documentation lives in Markdown.

## Configuration File Inventory

| File | Purpose |
|------|---------|
| `CMakeLists.txt` (root, 587 lines) | Main build config |
| `cmake/nlfConfig.cmake.in` | `find_package(nlf)` template |
| `bootstrap.sh` (103 lines) | Interactive installer |
| `.github/workflows/ci.yml` (113 lines) | GitHub Actions CI |
| `.gitignore` | Excludes build artifacts, CSVs, venvs, install/, local configs |
| `.mcp.json` | GPD MCP server config (git-ignored) |
| `.claude/` | Claude Code local config (git-ignored) |
| `requirements.txt` | Python: `pandas>=2.0.0 matplotlib>=3.7.0 numpy>=1.24.0` |
| `cuda-keyring_1.1-1_all.deb` | NVIDIA CUDA APT keyring (git-ignored line 23; installer artifact, not a build dep) |

---

_Structure analysis: 2026-07-18. Reference build: NLF v3.4.0, C++20, Eigen 3.4._
