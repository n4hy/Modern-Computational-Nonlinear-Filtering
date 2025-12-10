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
    std::cout << "Starting UKF Drag Ball Simulation (Comparison)..." << std::endl;

    double dt = 0.05;
    double T = 5.0;
    int steps = static_cast<int>(T / dt);

    double beta = 0.01;
    double q_wind_std = 0.2;
    double r_std = 0.5;
    int lag = 10;

    DragBallModel model(dt, beta, q_wind_std, r_std);

    VectorXd x_true(6);
    x_true << 0, 0, 0, 70, 0, 70;

    VectorXd x_est = x_true;
    x_est(0) += 5.0;
    x_est(3) -= 5.0;

    MatrixXd P_est = MatrixXd::Identity(6, 6) * 1.0;
    P_est(3,3) = 10.0; P_est(5,5) = 10.0;

    std::random_device rd;
    std::mt19937 gen(12345);

    UnscentedFixedLagSmoother smoother(&model, x_est, P_est, lag);

    std::vector<VectorXd> true_history;
    std::vector<VectorXd> smooth_history;
    std::vector<VectorXd> filt_history;

    for (int k = 0; k < steps; ++k) {
        VectorXd w(6);
        w.setZero();
        w(3) = randn(0, q_wind_std, gen);
        w(4) = randn(0, q_wind_std, gen);
        w(5) = randn(0, q_wind_std, gen);

        x_true = model.f(x_true) + w;

        VectorXd v(3);
        for(int i=0; i<3; ++i) v(i) = randn(0, r_std, gen);
        VectorXd y = model.h(x_true) + v;

        true_history.push_back(x_true);

        VectorXd x_out, x_filt;
        MatrixXd P_out, P_filt;
        bool ready = smoother.process(y, x_out, P_out, x_filt, P_filt);

        if (ready) {
            smooth_history.push_back(x_out);
            filt_history.push_back(x_filt);
        }
    }

    // Flush
    VectorXd x_out;
    MatrixXd P_out;
    while(smoother.flush(x_out, P_out)) {
        smooth_history.push_back(x_out);
        filt_history.push_back(x_out); // Dummy
    }

    VectorXd x_init_true(6);
    x_init_true << 0, 0, 0, 70, 0, 70;
    true_history.insert(true_history.begin(), x_init_true);

    int n = std::min(true_history.size(), smooth_history.size());

    double sum_sq_filt = 0, sum_sq_smooth = 0;
    int count = 0;

    std::cout << "\nResults (Time | True Z | Filt Z (Err) | Smooth Z (Err)):" << std::endl;
    for (int i = 0; i < n; ++i) {
        VectorXd err_s = true_history[i] - smooth_history[i];
        VectorXd err_f = true_history[i] - filt_history[i];

        double norm_s = err_s.norm();
        double norm_f = err_f.norm();

        if (i < n - lag) {
             sum_sq_filt += norm_f * norm_f;
             sum_sq_smooth += norm_s * norm_s;
             count++;
        }

        if (i % 20 == 0) {
            std::cout << "t=" << std::fixed << std::setprecision(2) << i*dt << " | "
                      << true_history[i](2) << " | "
                      << filt_history[i](2) << " (" << norm_f << ") | "
                      << smooth_history[i](2) << " (" << norm_s << ")"
                      << std::endl;
        }
    }

    if (count > 0) {
        double rmse_f = std::sqrt(sum_sq_filt / count);
        double rmse_s = std::sqrt(sum_sq_smooth / count);
        std::cout << "\nRMSE (Pre-flush): Filtered=" << rmse_f << ", Smoothed=" << rmse_s << std::endl;

        if (rmse_s < rmse_f) {
             std::cout << "SUCCESS: Smoothing reduced error." << std::endl;
             return 0;
        } else {
             std::cout << "WARNING: Smoothing did not reduce error (might be noise)." << std::endl;
             return 0;
        }
    }
    return 0;
}
