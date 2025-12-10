# Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) with Fixed-Lag Smoothing

This repository provides robust C++ implementations of the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF), featuring Fixed-Lag Smoothing (Windowed RTS Smoother). It handles nonlinear state-space models with time-varying dynamics and control inputs.

## Prerequisites

*   **C++ Compiler**: C++20 compliant compiler (e.g., GCC 10+, Clang 10+).
*   **CMake**: Version 3.10 or higher.
*   **Eigen3**: Linear algebra library (version 3.4.0 required).
*   **Python 3**: For visualization scripts (requires `pandas`, `matplotlib`).

## Building the Project

1.  **Install Eigen3**:
    The project requires Eigen 3.4. If your system package manager has an older version (e.g., 3.3), you must install 3.4 manually.

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

2.  **Build the Filters**:
    ```bash
    # Go back to repo root
    mkdir build
    cd build

    # Configure CMake (point to local Eigen install)
    cmake .. -DEigen3_DIR=/path/to/eigen_install/eigen_local/share/eigen3/cmake

    make
    ```

## Running the Examples

### EKF with Fixed-Lag Smoothing
Simulates a damped nonlinear pendulum.
```bash
cd build
./EKF/ekf_test
```
Produces `ekf_results.csv`.

### UKF with Fixed-Lag Smoothing
Simulates a ball falling with air drag and wind gusts.
```bash
cd build
./UKF/ukf_test
```
Produces `ukf_results.csv`.

### Visualization
To plot the results:
```bash
python3 ../scripts/plot_results.py
```
This produces PNG plots of the trajectories and errors.

## Code Structure

*   `Common/`: Shared headers.
    *   `SystemModel.h`: Abstract base class for EKF models (Dynamic size).
    *   `StateSpaceModel.h`: Templated base class for UKF models (Fixed size).
*   `EKF/`: EKF implementation (Dynamic size).
*   `UKF/`: New UKF implementation (C++20, Templated, Fixed size).
    *   `UKF.h`: Core Unscented Kalman Filter.
    *   `UnscentedFixedLagSmoother.h`: Fixed-Lag Smoother.
    *   `SigmaPoints.h`: Scaled Unscented Transform.
