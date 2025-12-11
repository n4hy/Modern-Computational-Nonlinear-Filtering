# Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PKF) with Fixed-Lag Smoothing

This repository provides robust C++ implementations of the Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PKF). All implementations feature Fixed-Lag Smoothing capabilities. It handles nonlinear state-space models with time-varying dynamics, control inputs, and non-Gaussian noise (PKF).

The project demonstrates three different C++ design patterns:
1.  **EKF**: Uses **dynamic sizing** (runtime dimensions) suitable for flexible applications.
2.  **UKF**: Uses **static templating** (compile-time dimensions) using C++20, optimizing for performance and type safety.
3.  **PKF**: Uses **static templating** with a generalized interface for **non-Gaussian** processes.

## Prerequisites

*   **C++ Compiler**: C++20 compliant compiler (e.g., GCC 10+, Clang 10+).
*   **CMake**: Version 3.10 or higher.
*   **Eigen3**: Linear algebra library (version 3.4.0 required).
*   **Python 3**: For visualization scripts (requires `pandas`, `matplotlib`).

## Building the Project

### 1. Install Eigen3 (Version 3.4.0)

The project requires Eigen 3.4 features. If your system package manager has an older version (e.g., 3.3 on Ubuntu 20.04), you must install 3.4 manually.

```bash
# Create a local install directory
mkdir -p eigen_install

# Download and extract Eigen 3.4.0
wget -qO eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen.tar.gz -C eigen_install

# Rename for consistency
mv eigen_install/eigen-3.4.0 eigen_install/eigen3

# Build and Install to a local prefix
cd eigen_install/eigen3
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(pwd)/../../eigen_local
make install
```

This will install Eigen config files to `eigen_install/eigen_local/share/eigen3/cmake`.

### 2. Build the Filters

```bash
# Go back to repo root
cd ../../../
mkdir build
cd build

# Configure CMake (point to local Eigen install)
# If you installed Eigen system-wide (3.4+), you can skip -DEigen3_DIR
cmake .. -DEigen3_DIR=$(pwd)/../eigen_install/eigen_local/share/eigen3/cmake

make
```

## Running the Examples

### EKF with Fixed-Lag Smoothing
Simulates a **Nonlinear Oscillator** (Damped Pendulum).
*   **State**: [Position, Velocity]
*   **Observation**: Position

```bash
cd build
./EKF/ekf_test
```
Produces `ekf_results.csv`.

To plot the EKF results:
```bash
python3 ../scripts/plot_results.py ekf_results.csv
```

### UKF with Fixed-Lag Smoothing
Simulates a **Drag Ball** (Falling object with air drag and wind gusts).
*   **State**: [x, y, vx, vy]
*   **Observation**: [x, y]
*   **Features**: Uses Merwe Scaled Sigma Points.

```bash
cd build
./UKF/ukf_test
```
Produces `ukf_results.csv`.

To plot the UKF results:
```bash
python3 ../scripts/ukf_plot_results.py ukf_results.csv
```
This produces `ukf_results.png`.

### PKF with Fixed-Lag Smoothing
Simulates a **Lorenz-63** (Chaotic System).
*   **State**: [x, y, z]
*   **Observation**: [x, y, z] with **Student-t** noise.
*   **Features**: Bootstrap Particle Filter with Systematic/Stratified Resampling and Ancestry Tracking Smoother.

```bash
cd build
./PKF/pkf_example
```
Produces `pkf_results.csv`.

To plot the PKF results:
```bash
python3 ../scripts/pkf_plot_results.py pkf_results.csv
```
This produces `pkf_results.png`.

## Algorithm Details

### Fixed-Lag Smoother (EKF & UKF)
Both implementations use a **Windowed Rauch-Tung-Striebel (RTS)** smoother.
1.  **Forward Pass**: Standard Kalman Filter (EKF/UKF) runs online. A buffer stores the last $L$ estimates and predictions.
2.  **Backward Pass**: At each step, the smoother recurses backward from the current time $k$ to $k-L$ using the stored correlations (Cross-covariance $P_{x_k, x_{k+1}}$).

### Fixed-Lag Smoother (PKF)
Uses an **Ancestry Tracking** approach.
1.  **Forward Pass**: Standard Bootstrap Particle Filter with resampling. Stores particle history and parent indices for the last $L$ steps.
2.  **Smoothing**: Reconstructs trajectories by backtracking parent indices from $k$ to $k-L$ to approximate $p(x_{k-L} | y_{1:k})$.

### Design & Architecture

#### EKF (Legacy/Dynamic)
*   **Directory**: `EKF/`
*   **Base Class**: `Common/include/SystemModel.h`
*   **Matrices**: `Eigen::MatrixXd` (Dynamic size).
*   **Flexibility**: Model dimensions can be set at runtime.

#### UKF (Modern/Static)
*   **Directory**: `UKF/`
*   **Base Class**: `Common/include/StateSpaceModel.h`
*   **Matrices**: `Eigen::Matrix<double, NX, NY>` (Fixed size).
*   **Performance**: No heap allocation for matrices during steps.
*   **Requirements**: C++20 Concepts (implicit), Templated Interfaces.

#### PKF (Non-Gaussian/Static)
*   **Directory**: `PKF/`
*   **Base Class**: `PKF/include/state_space_model.hpp`
*   **Matrices**: `Eigen::Matrix<double, NX, NY>` (Fixed size).
*   **Features**: Supports custom likelihoods (e.g., Student-t) and particle-based representation.
*   **Requirements**: C++20.

## Project Structure

```text
.
├── CMakeLists.txt          # Root build configuration
├── Common/                 # Shared Headers
│   ├── include/
│   │   ├── StateSpaceModel.h   # Templated Base Class (UKF)
│   │   ├── SystemModel.h       # Dynamic Base Class (EKF)
│   │   └── FileUtils.h
├── EKF/                    # Extended Kalman Filter
│   ├── include/
│   ├── src/
│   └── main.cpp
├── UKF/                    # Unscented Kalman Filter
│   ├── include/
│   │   ├── UKF.h                   # Core Filter Implementation
│   │   ├── UnscentedFixedLagSmoother.h
│   │   ├── SigmaPoints.h           # Sigma Point Generation
│   │   └── DragBallModel.h         # Example Model
│   └── main.cpp            # Test Harness
├── PKF/                    # Particle Filter
│   ├── include/
│   │   ├── particle_filter.hpp
│   │   ├── particle_fixed_lag.hpp
│   │   ├── state_space_model.hpp
│   │   ├── noise_models.hpp
│   │   └── resampling.hpp
│   ├── src/
│   │   └── example_main.cpp    # Lorenz-63 Example
│   └── tests/
│       └── test_particle.cpp
└── scripts/                # Visualization
    ├── plot_results.py     # EKF Plotter
    ├── ukf_plot_results.py # UKF Plotter
    └── pkf_plot_results.py # PKF Plotter
```
