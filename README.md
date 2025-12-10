# Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) with Fixed-Lag Smoothing

This repository provides robust C++ implementations of the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF), featuring Fixed-Lag Smoothing (Windowed RTS Smoother). It handles nonlinear state-space models with time-varying dynamics and control inputs.

## Prerequisites

*   **C++ Compiler**: C++17 compliant compiler (e.g., GCC 7+, Clang 5+).
*   **CMake**: Version 3.10 or higher.
*   **Eigen3**: Linear algebra library (version 3.4.0 recommended).
*   **Python 3**: For visualization scripts (requires `pandas`, `matplotlib`).

## Building the Project

1.  **Install Eigen3** (if not already installed):
    You can install it via your package manager or build from source.
    ```bash
    # Ubuntu/Debian
    sudo apt-get install libeigen3-dev

    # Or local install (useful for environments without root access):
    mkdir -p eigen_install
    wget -qO eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    tar -xzf eigen.tar.gz -C eigen_install
    mv eigen_install/eigen-3.4.0 eigen_install/eigen3
    # When running cmake, use: -DEIGEN3_INCLUDE_DIR=/path/to/eigen_install/eigen3
    ```

2.  **Build**:
    ```bash
    mkdir build
    cd build
    cmake ..  # Add -DEIGEN3_INCLUDE_DIR=... if needed
    make
    ```

## Running the Examples

### EKF with Fixed-Lag Smoothing (Nonlinear Oscillator)
This example simulates a damped nonlinear pendulum and compares the Filtered estimate vs. the Smoothed estimate.

```bash
cd build
./EKF/ekf_test
```
This will generate `ekf_results.csv`.

To plot the results:
```bash
python3 ../scripts/plot_results.py
```
This produces `ekf_results.png`.

### UKF with Fixed-Lag Smoothing (Drag Ball)
This example simulates a ball falling with air drag and wind gusts.

```bash
cd build
./UKF/ukf_test
```

## Code Structure

*   `Common/`: Shared headers, including the abstract `SystemModel` class.
*   `EKF/`: EKF implementation.
    *   `EKF.h/cpp`: Core EKF algorithm (Joseph form update, robust numerics).
    *   `EKFFixedLag.h/cpp`: Windowed RTS Smoother.
    *   `NonlinearOscillator.h`: Example model.
*   `UKF/`: UKF implementation.
    *   `UKF.h/cpp`: Core UKF algorithm.
    *   `UnscentedFixedLagSmoother.h/cpp`: Smoother implementation.
*   `scripts/`: Python visualization tools.

## Key Features

*   **Robust Numerics**: Uses Joseph form for covariance updates to maintain positive definiteness. Uses robust linear solvers (Cholesky/QR) instead of explicit inverses where possible.
*   **Modular Design**: Models are defined separately from the filter logic via the `SystemModel` interface.
*   **Time-Varying & Control**: Fully supports `f(x, u, t)` and `h(x, t)` dynamics.
