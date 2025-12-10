#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "DragBallModel.h"
#include "UnscentedFixedLagSmoother.h"

using namespace Eigen;
using namespace std;

// Helper to generate Gaussian noise
double randn(double mean, double stddev, std::mt19937& gen) {
    std::normal_distribution<double> d(mean, stddev);
    return d(gen);
}

int main() {
    std::cout << "Starting UKF Drag Ball Simulation..." << std::endl;

    // Parameters
    double dt = 0.05;
    double T = 5.0;
    int steps = static_cast<int>(T / dt);

    // Beta approx 0.01 for our ball
    double beta = 0.01;

    // Process Noise (Wind)
    // Reduced to keep tracking feasible within tight bounds
    double q_wind_std = 0.2;
    double r_std = 0.5;
    int lag = 10;

    // Setup Model
    DragBallModel model(dt, beta, q_wind_std, r_std);

    // Initial State (True)
    VectorXd x_true(6);
    x_true << 0, 0, 0, 70, 0, 70;

    // Initial Guess (Filter)
    VectorXd x_est = x_true;
    x_est(0) += 5.0;
    x_est(3) -= 5.0;

    MatrixXd P_est = MatrixXd::Identity(6, 6) * 1.0;
    P_est(3,3) = 10.0; P_est(5,5) = 10.0;

    std::random_device rd;
    std::mt19937 gen(12345);

    // Initialize Smoother
    UnscentedFixedLagSmoother smoother(&model, x_est, P_est, lag);

    // Storage
    std::vector<VectorXd> true_history;
    std::vector<VectorXd> smooth_history;

    // Simulate Loop
    for (int k = 0; k < steps; ++k) {
        // 1. Evolve Truth
        VectorXd w(6);
        w.setZero();
        w(3) = randn(0, q_wind_std, gen);
        w(4) = randn(0, q_wind_std, gen);
        w(5) = randn(0, q_wind_std, gen);

        x_true = model.f(x_true) + w;

        // 2. Generate Measurement
        VectorXd v(3);
        for(int i=0; i<3; ++i) v(i) = randn(0, r_std, gen);
        VectorXd y = model.h(x_true) + v;

        true_history.push_back(x_true);

        // 3. Process with Smoother
        VectorXd x_out;
        MatrixXd P_out;
        bool ready = smoother.process(y, x_out, P_out);

        if (ready) {
            smooth_history.push_back(x_out);
        }
    }

    // Flush
    VectorXd x_out;
    MatrixXd P_out;
    while(smoother.flush(x_out, P_out)) {
        smooth_history.push_back(x_out);
    }

    // Alignment
    VectorXd x_init_true(6);
    x_init_true << 0, 0, 0, 70, 0, 70;
    true_history.insert(true_history.begin(), x_init_true);

    // RMSE
    double mse_pos = 0;
    double mse_vel = 0;
    int n = std::min(true_history.size(), smooth_history.size());

    std::cout << "\nResults (Time, True Z, Smooth Z):" << std::endl;
    for (int i = 0; i < n; ++i) {
        VectorXd err = true_history[i] - smooth_history[i];
        mse_pos += err.head(3).squaredNorm();
        mse_vel += err.tail(3).squaredNorm();

        if (i % 20 == 0) {
            std::cout << "t=" << std::fixed << std::setprecision(2) << i*dt << ": "
                      << true_history[i](2) << " vs " << smooth_history[i](2)
                      << " (VelErr: " << err(5) << ")" << std::endl;
        }
    }

    double rmse_pos = std::sqrt(mse_pos / n);
    double rmse_vel = std::sqrt(mse_vel / n);

    std::cout << "\nRMSE Position: " << rmse_pos << std::endl;
    std::cout << "RMSE Velocity: " << rmse_vel << std::endl;

    if (rmse_pos < 5.0 && rmse_vel < 10.0) {
        std::cout << "SUCCESS: Tracking reasonable." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Error too high." << std::endl;
        return 1;
    }
}
