# Nonlinear Filtering Library

This repository contains robust C++ implementations of nonlinear state estimators, including **Extended Kalman Filter (EKF)**, **Unscented Kalman Filter (UKF)**, and **Particle Filter (PKF)**. All implementations support **Fixed-Lag Smoothing**.

The project is designed as a library (`libNonlinearFiltering`) for EKF and UKF, with a standalone demo for the Particle Filter.

## Architectures

The library demonstrates three different C++ design patterns:

1.  **EKF (Extended Kalman Filter)**
    *   **Style**: Dynamic Sizing (`Eigen::MatrixXd`).
    *   **Use Case**: Systems where dimensions might change at runtime or are not known at compile time.
    *   **Location**: `EKF/`

2.  **UKF (Unscented Kalman Filter)**
    *   **Style**: Static Templating (C++20, `Eigen::Matrix<double, NX, NY>`).
    *   **Use Case**: High-performance embedded systems with fixed dimensions. Zero heap allocation during filter steps.
    *   **Location**: `UKF/`

3.  **PKF (Particle Filter)**
    *   **Style**: Static Templating with flexible Policy-based Design.
    *   **Features**: Supports non-Gaussian noise (e.g., Student-t), Systematic/Stratified Resampling, and Ancestry Tracking for smoothing.
    *   **Location**: `PKF/` (Currently a standalone demo).

---

## Installation

### Prerequisites

*   **CMake** (>= 3.16)
*   **C++ Compiler** (C++20 compliant, e.g., GCC 10+, Clang 10+)
*   **Eigen 3.4+**

### Step 1: Install Eigen 3.4 (Manual Instruction)

The project requires Eigen 3.4. If your system package manager provides an older version, follow these steps to install it manually.

```bash
# 1. Download Eigen 3.4.0 source
mkdir -p /tmp/eigen_install
wget -qO /tmp/eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf /tmp/eigen.tar.gz -C /tmp/eigen_install

# 2. Build and Install
cd /tmp/eigen_install/eigen-3.4.0
mkdir build && cd build
cmake ..
sudo make install
```
*Note: The project's CMake build system can also automatically download Eigen via `FetchContent` if it is not found on the system.*

### Step 2: Build and Install the Library

```bash
# 1. Clone the repository
git clone <repository_url>
cd <repository_name>

# 2. Configure CMake
mkdir build && cd build
cmake ..

# 3. Build
make

# 4. Install (Default prefix: /usr/local)
sudo make install
```

This will install:
*   **Headers**: `/usr/local/include/NonlinearFiltering/{Common,EKF,UKF}/`
*   **Library**: `/usr/local/lib/libNonlinearFiltering.a`
*   **CMake Config**: `/usr/local/lib/cmake/NonlinearFiltering/`

---

## Running Demos

The build process generates three test executables in the `build/` directory:

### 1. Particle Filter (PKF) Demo
Simulates a Lorenz-63 chaotic system with Student-t observation noise.
```bash
./pkf_demo
```
*   **Output**: `pkf_results.csv`
*   **Verification**: Computes RMSE for Filtered vs. Smoothed estimates.

### 2. EKF Test
Simulates a Nonlinear Oscillator (Damped Pendulum).
```bash
./ekf_test
```
*   **Output**: `ekf_results.csv`

### 3. UKF Test
Simulates a Ball with Drag and Wind.
```bash
./ukf_test
```
*   **Output**: `ukf_results.csv`

---

## Using the Library

To use the installed library in your own project:

**CMakeLists.txt**:
```cmake
find_package(NonlinearFiltering REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE NonlinearFiltering::NonlinearFiltering)
```

**C++ Code**:
```cpp
#include <NonlinearFiltering/EKF/EKF.h>
#include <NonlinearFiltering/UKF/UKF.h>
// ...
```
