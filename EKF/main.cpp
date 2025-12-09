#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "BallTossModel.h"
#include "FixedLagSmoother.h"

using namespace Eigen;
using namespace std;

// Helper to generate Gaussian noise
double randn(double mean, double stddev, std::mt19937& gen) {
    std::normal_distribution<double> d(mean, stddev);
    return d(gen);
}

int main() {
    std::cout << "Starting EKF Ball Toss Simulation..." << std::endl;

    // Parameters
    double dt = 0.1;
    double T = 5.0; // Seconds
    int steps = static_cast<int>(T / dt);
    double q_std = 0.01; // Process noise std (small)
    double r_std = 0.5;  // Measurement noise std
    int lag = 5;         // Smoothing lag steps

    // Setup Model
    BallTossModel model(dt, q_std, r_std);

    // Initial State (True)
    // Start at (0,0,0) with velocity (10, 5, 20)
    VectorXd x_true(6);
    x_true << 0, 0, 0, 10, 5, 20;

    // Initial Guess (Filter)
    // Slightly off
    VectorXd x_est = x_true;
    x_est(0) += 2.0;
    x_est(3) -= 1.0;

    MatrixXd P_est = MatrixXd::Identity(6, 6) * 1.0;
    P_est(3,3) = 5.0; P_est(4,4) = 5.0; P_est(5,5) = 5.0;

    // Random Number Generator
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility

    // Initialize Smoother
    FixedLagSmoother smoother(&model, x_est, P_est, lag);

    // Storage for analysis
    std::vector<VectorXd> true_history;
    std::vector<VectorXd> meas_history;
    std::vector<VectorXd> smooth_history;
    std::vector<int> smooth_times;

    // Simulate Loop
    for (int k = 0; k < steps; ++k) {
        // 1. Evolve Truth
        // Add process noise to truth? Yes, usually.
        VectorXd w(6);
        for(int i=0; i<6; ++i) w(i) = randn(0, q_std, gen);
        x_true = model.f(x_true) + w;

        // 2. Generate Measurement
        VectorXd v(3);
        for(int i=0; i<3; ++i) v(i) = randn(0, r_std, gen);
        VectorXd y = model.h(x_true) + v;

        true_history.push_back(x_true);
        meas_history.push_back(y);

        // 3. Process with Smoother
        VectorXd x_out;
        MatrixXd P_out;
        bool ready = smoother.process(y, x_out, P_out);

        if (ready) {
            int time_idx = k - lag + 1; // +1 because we started loop at next step logic
            // Actually, let's trace indices.
            // Loop k=0: buffer size 1 (initial) -> process -> buffer size 2 (init + k=0).
            // Lag=5. Need size 6.
            // So output starts when k >= lag - 1?

            // Just push back what we get
            smooth_history.push_back(x_out);

            // To align with truth, we need to know WHICH time step this output corresponds to.
            // The smoother outputs x_{T-lag}.
            // Current step is k.
            // In process(), we pushed current step k.
            // Output is buffer_[0].
            // If we started with buffer containing x_init (time -1 or 0?), it's tricky.
            // In Smoother constructor, we pushed init state (let's call it time 0).
            // k=0 process: Pushes time 1 state. Buffer: [0, 1].
            // If lag=1. Size > 1? Yes (2). Output 0.
            // So if k=0, we output time 0.
            // Truth history index 0 is time 1 (after first evolution).

            // Let's adjust alignment.
            // Truth history[k] is x_{k+1} if we consider x_true init as x_0.
            // Let's store x_0 in truth history first.
        }
    }

    // Flush remaining
    VectorXd x_out;
    MatrixXd P_out;
    while(smoother.flush(x_out, P_out)) {
        smooth_history.push_back(x_out);
    }

    // Analysis
    // We need to align true_history and smooth_history.
    // FixedLagSmoother constructor takes x0. That is time 0.
    // loop k=0 generates x1 (truth), y1. process(y1) generates x1_est, stores in buffer.
    // Buffer: [x0, x1].
    // If lag=1, returns x0 smoothed.
    // So smooth_history[0] corresponds to Time 0.

    // true_history contains [x1, x2, ...].
    // Let's prepend the known x_true initial to true_history for RMSE calc.
    VectorXd x_init_true(6);
    x_init_true << 0, 0, 0, 10, 5, 20; // Re-create from top
    true_history.insert(true_history.begin(), x_init_true);

    // Calculate RMSE
    double mse_pos = 0;
    double mse_vel = 0;
    int n = std::min(true_history.size(), smooth_history.size());

    std::cout << "\nResults (Time, True Z, Smooth Z):" << std::endl;
    for (int i = 0; i < n; ++i) {
        VectorXd err = true_history[i] - smooth_history[i];
        mse_pos += err.head(3).squaredNorm();
        mse_vel += err.tail(3).squaredNorm();

        if (i % 10 == 0) {
            std::cout << "t=" << i*dt << ": "
                      << true_history[i](2) << " vs " << smooth_history[i](2)
                      << " (Err: " << err(2) << ")" << std::endl;
        }
    }

    double rmse_pos = std::sqrt(mse_pos / n);
    double rmse_vel = std::sqrt(mse_vel / n);

    std::cout << "\nRMSE Position: " << rmse_pos << std::endl;
    std::cout << "RMSE Velocity: " << rmse_vel << std::endl;

    if (rmse_pos < 2.0 && rmse_vel < 5.0) {
        std::cout << "SUCCESS: Tracking within expected bounds." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Error too high." << std::endl;
        return 1;
    }
}
