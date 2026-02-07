# Comprehensive Filtering Benchmarks

This directory contains challenging benchmark problems and a comprehensive testing framework for evaluating all nonlinear filtering methods in this repository.

## Overview

The benchmark suite tests filtering methods on increasingly difficult problems with:
- **High dimensionality** (up to 40 states)
- **Strong nonlinearity**
- **Chaotic dynamics**
- **Partial observability**
- **Discontinuous forcing**

## Benchmark Problems

### 1. Coupled Oscillators (10D)
- **State dimension**: 10 (5 oscillators, position + velocity each)
- **Observation dimension**: 5 (positions only)
- **Characteristics**:
  - Nonlinear coupling between all oscillators
  - Damped dynamics
  - Nonlinear observation function: `y = pos + 0.1*sin(pos)`
- **Difficulty**: Medium-High (high dimensional, coupled nonlinear dynamics)

### 2. Lorenz96 Model (40D)
- **State dimension**: 40
- **Observation dimension**: 10 (observe every 4th variable)
- **Characteristics**:
  - Highly chaotic (used in weather prediction)
  - Spatial coupling structure
  - Forcing parameter F=8 (chaotic regime)
- **Difficulty**: Very High (chaotic, high dimensional, sparse observations)

### 3. Van der Pol with Discontinuous Forcing (2D)
- **State dimension**: 2
- **Observation dimension**: 1
- **Characteristics**:
  - Nonlinearity parameter μ=5 (stiff oscillator)
  - Square wave forcing (discontinuous control)
  - Quadratic measurement nonlinearity
- **Difficulty**: Medium (low dimensional but highly stiff and discontinuous)

### 4. Reentry Vehicle Tracking (6D)
- **State dimension**: 6 (position + velocity in 3D)
- **Observation dimension**: 3 (range, azimuth, elevation)
- **Characteristics**:
  - Exponential atmospheric model
  - Altitude-dependent drag
  - Spherical observation coordinates
  - Gravity model
- **Difficulty**: High (highly nonlinear observations and dynamics)

### 5. Bearing-Only Tracking (4D)
- **State dimension**: 4 (2D position + velocity)
- **Observation dimension**: 1 (bearing angle only)
- **Characteristics**:
  - Weakly observable (bearing-only)
  - Moving observer platform
  - Constant velocity target
- **Difficulty**: High (observability challenges)

## Filters Tested

### 1. UKF (Unscented Kalman Filter)
- Derivative-free nonlinear filtering
- Sigma point propagation
- Standard covariance propagation

### 2. SRUKF (Square Root UKF)
- **NEW**: Square root formulation for improved numerical stability
- Propagates Cholesky factor S where P = S*S^T
- Uses QR decomposition and rank-1 updates
- Guarantees positive definiteness

### 3. UKF + Fixed-Lag Smoother
- UKF with Rauch-Tung-Striebel (RTS) backward smoothing
- Lag window of 20-30 steps
- Improves estimates using future measurements

### 4. SRUKF + Fixed-Lag Smoother
- **NEW**: SRUKF with RTS smoothing in square root form
- Enhanced numerical stability for smoothing
- Optimal for long lag windows

## Evaluation Metrics

### Accuracy Metrics
- **RMSE (Root Mean Square Error)**: Overall estimation accuracy
  - Position RMSE
  - Velocity RMSE
  - Overall state RMSE
- **Smoothed RMSE**: Accuracy of smoothed estimates

### Consistency Metrics
- **NEES (Normalized Estimation Error Squared)**
  - Mean NEES (should be ≈ state dimension for consistent filter)
  - Std NEES (indicates filter consistency)

### Performance Metrics
- **Average Step Time**: Computational cost per filter step
- **Total Execution Time**: End-to-end runtime

### Convergence Metrics
- **Convergence Time**: Time to reach steady-state error
- **Number of Divergences**: Count of large errors (error > 10x threshold)

## Building and Running

### Build
```bash
mkdir build && cd build
cmake ..
make run_benchmarks
```

### Run Benchmarks
```bash
./Benchmarks/run_benchmarks
```

This will:
1. Generate ground truth trajectories for each problem
2. Run all filters on each problem
3. Compute comprehensive metrics
4. Save results to CSV files

### Output Files

#### Summary Results
- `benchmark_results.csv`: Comprehensive metrics table for all filters and problems

#### Trajectory Files
- `coupled_osc_ukf.csv`, `coupled_osc_srukf.csv`, etc.
- `vanderpol_ukf.csv`, `vanderpol_srukf.csv`, etc.
- `bearing_ukf.csv`, `bearing_srukf.csv`, etc.

Each trajectory file contains:
- Time stamps
- True state values
- Filtered estimates
- Smoothed estimates (if applicable)

### Visualization

Use the provided Python script to generate comparison plots:

```bash
python3 ../scripts/plot_benchmarks.py .
```

This generates:
- `benchmark_rmse_comparison.png`: RMSE comparison across all methods
- `benchmark_timing_comparison.png`: Computational performance comparison
- `benchmark_convergence_comparison.png`: Convergence and stability metrics
- Individual trajectory plots for each test case

## Expected Results

### SRUKF vs UKF
- **Similar accuracy** on well-conditioned problems
- **Better stability** on ill-conditioned or long-duration problems
- **Guaranteed positive definiteness** (no covariance matrix issues)
- **Slightly higher computational cost** due to QR decomposition

### Smoothers vs Filters
- **10-30% RMSE improvement** from smoothing
- **Better performance** on weakly observable problems (bearing-only)
- **2-3x computational cost** due to backward pass

### Problem-Specific Insights

#### Lorenz96 (40D Chaos)
- Filters may diverge due to chaos
- Frequent resampling/reinitialization may be needed
- High NEES values expected

#### Bearing-Only Tracking
- Large initial error due to weak observability
- Significant improvement from smoothing
- Convergence time strongly depends on observer motion

## Extending the Benchmarks

### Adding New Problems

1. Create a new model class in `include/BenchmarkProblems.h`:
```cpp
template<int NX, int NY>
class YourNewProblem : public UKFModel::StateSpaceModel<NX, NY> {
    // Implement f(), h(), Q(), R()
};
```

2. Add test case in `src/run_benchmarks.cpp`:
```cpp
{
    YourNewProblem<NX, NY> model;
    // Initialize and run benchmarks
}
```

### Adding New Filters

1. Implement filter following existing interface
2. Add benchmark runner function in `src/run_benchmarks.cpp`
3. Add to main loop

## Performance Tips

### For Real-Time Applications
- Use UKF for faster execution
- Reduce lag window size for smoothers
- Consider decimating observations on high-rate sensors

### For Accuracy
- Use SRUKF for long-duration tracking
- Use smoothers when real-time is not critical
- Increase lag window for better smoothing (up to ~100 steps)

### For Numerical Stability
- Always use SRUKF for:
  - Very high-dimensional systems (>20 states)
  - Long-duration missions (>1000 steps)
  - Systems with very small process noise
- Monitor NEES: values >> state dimension indicate filter divergence

## References

1. **UKF**: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", 2004
2. **SRUKF**: Van der Merwe & Wan, "The Square-Root Unscented Kalman Filter", 2001
3. **RTS Smoothing**: Rauch et al., "Maximum Likelihood Estimates of Linear Dynamic Systems", 1965
4. **Lorenz96**: Lorenz, "Predictability: A Problem Partly Solved", 1996

## License

Part of the Modern Computational Nonlinear Filtering project.
