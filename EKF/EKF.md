# Extended Kalman Filter (EKF) and Fixed-Lag Smoother

## Overview

This module implements a generic Extended Kalman Filter (EKF) and a Fixed-Lag Smoother in C++ using the Eigen library for linear algebra. The system is designed to track nonlinear dynamical systems with additive Gaussian noise.

Key features:
1.  **Generic System Model**: Abstract interface for defining state transition ($f$), observation ($h$), and Jacobians ($F, H$).
2.  **Extended Kalman Filter**: Standard Predict-Update architecture.
3.  **Fixed-Lag Smoother with Feedback**: Implements a Rauch-Tung-Striebel (RTS) smoother over a sliding window. Uniquely, this implementation uses the smoothed estimate at the lag horizon to **reset and re-filter** the forward state, ensuring that the real-time filter benefits from delayed, smoothed information.

## Architecture

### 1. SystemModel (`SystemModel.h`)
Users must inherit from `SystemModel` and implement:
*   `f(x)`: State transition function.
*   `h(x)`: Observation function.
*   `F(x)`: Jacobian of $f$ with respect to $x$.
*   `H(x)`: Jacobian of $h$ with respect to $x$.
*   `Q()`, `R()`: Process and Measurement noise covariance matrices.

### 2. EKF (`EKF.h`, `EKF.cpp`)
The core filter maintaining the state estimate $\hat{x}$ and covariance $P$.
*   **Predict**: $\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1})$, $P_{k|k-1} = F P_{k-1|k-1} F^T + Q$.
*   **Update**: $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K (y_k - h(\hat{x}_{k|k-1}))$.

### 3. FixedLagSmoother (`FixedLagSmoother.h`, `FixedLagSmoother.cpp`)
Wraps the EKF to provide smoothing.

**Operation Loop:**
1.  **Forward Step**: The internal EKF predicts and updates based on the new measurement $y_T$.
2.  **Buffer**: The state and measurement are added to a sliding window buffer.
3.  **Smoothing**: If the buffer size exceeds the lag:
    *   The RTS backward pass runs from time $T$ down to $T-\text{lag}$.
    *   Formula: $\hat{x}_{k|T} = \hat{x}_{k|k} + C_k (\hat{x}_{k+1|T} - \hat{x}_{k+1|k})$, where $C_k = P_{k|k} F_k^T P_{k+1|k}^{-1}$.
4.  **Feedback (Re-filtering)**:
    *   The smoothed state at $T-\text{lag}$ ($\hat{x}_{T-\text{lag}|T}$) is treated as the new "truth".
    *   The internal EKF is reset to this smoothed state.
    *   The filter is re-run forward from $T-\text{lag}$ to $T$ using the stored measurements in the buffer.
    *   This ensures the current estimate $\hat{x}_{T|T}$ is consistent with the improved history.
5.  **Output**: The system publishes the smoothed state at $T-\text{lag}$.

## Usage

### Dependencies
*   **Eigen 3.4.0**: Must be available. Ensure `include_directories` in CMake points to it.

### Example (Ball Toss)
See `main.cpp` for a complete example of a 3D Ball Toss simulation.

```cpp
// 1. Define Model
BallTossModel model(dt, q_std, r_std);

// 2. Initialize Smoother
FixedLagSmoother smoother(&model, x0, P0, lag);

// 3. Loop
for (each measurement y) {
    Eigen::VectorXd x_out;
    Eigen::MatrixXd P_out;
    
    if (smoother.process(y, x_out, P_out)) {
        // x_out is the smoothed state at time (current_time - lag)
        std::cout << "Smoothed State: " << x_out.transpose() << std::endl;
    }
}

// 4. Flush remaining states at end
smoother.flush(x_out, P_out);
```

## Implementation Details

### Feedback Mechanism
The decision to re-filter from the smoothed lag point effectively couples the smoothing performance into the forward tracking stability. While computationally more expensive ($O(\text{lag})$ per step), it prevents divergence in nonlinear systems by periodically anchoring the trajectory to a retrospective estimate that uses future information.

### Mathematical Formulation
**EKF Prediction:**
$$ \hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}) $$
$$ P_{k|k-1} = F_{k-1} P_{k-1|k-1} F_{k-1}^T + Q $$

**EKF Update:**
$$ K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1} $$
$$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (y_k - h(\hat{x}_{k|k-1})) $$
$$ P_{k|k} = (I - K_k H_k) P_{k|k-1} $$

**RTS Smoother (Backward):**
$$ C_k = P_{k|k} F_k^T P_{k+1|k}^{-1} $$
$$ \hat{x}_{k|N} = \hat{x}_{k|k} + C_k (\hat{x}_{k+1|N} - \hat{x}_{k+1|k}) $$
$$ P_{k|N} = P_{k|k} + C_k (P_{k+1|N} - P_{k+1|k}) C_k^T $$
