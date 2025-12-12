#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include "EKF.h"
#include "EKFFixedLag.h"
#include "NonlinearOscillator.h"

// Helper to generate noise
float randn(float mean, float stddev) {
    static std::mt19937 gen(42);
    std::normal_distribution<float> d(mean, stddev);
    return d(gen);
}

int main() {
    // 1. Setup Simulation
    float dt = 0.05f;
    int steps = 200;
    NonlinearOscillator model(dt);

    Eigen::VectorXf x_true(2);
    x_true << 1.0f, 0.0f; // Initial state: pos=1, vel=0

    // Arrays to store history
    std::vector<float> time_hist;
    std::vector<Eigen::VectorXf> x_true_hist;
    std::vector<Eigen::VectorXf> y_meas_hist;

    std::cout << "Generating Data..." << std::endl;
    for (int k = 0; k < steps; ++k) {
        float t = k * dt;

        // Propagate truth (no process noise in truth generation for smoother curve, or add it?)
        // Let's add process noise to make it interesting for the filter
        Eigen::VectorXf u(0); // No control
        Eigen::VectorXf w(2);
        w << randn(0.0f, 0.01f), randn(0.0f, 0.01f);

        x_true = model.f(x_true, u, t) + w;

        // Generate measurement
        Eigen::VectorXf v(1);
        v << randn(0.0f, 0.1f);
        Eigen::VectorXf y = model.h(x_true, t) + v;

        time_hist.push_back(t);
        x_true_hist.push_back(x_true);
        y_meas_hist.push_back(y);
    }

    // 2. Setup Filter and Smoother
    int lag = 10;
    Eigen::VectorXf x0 = Eigen::VectorXf::Zero(2); // Imperfect initialization
    Eigen::MatrixXf P0 = Eigen::MatrixXf::Identity(2, 2) * 1.0f;

    EKFFixedLag smoother(&model, x0, P0, lag);

    // Store results
    std::vector<Eigen::VectorXf> x_filt_hist;
    std::vector<Eigen::VectorXf> x_smooth_hist;

    std::cout << "Running EKF + Smoother..." << std::endl;
    for (int k = 0; k < steps; ++k) {
        Eigen::VectorXf u(0);
        smoother.step(y_meas_hist[k], u, time_hist[k]);

        auto [x_filt, P_filt] = smoother.getFilteredState();
        // Get smoothed state at lag L (returns x_{k-L|k})

        x_filt_hist.push_back(x_filt);

        if (k >= lag) {
            auto [x_s, P_s] = smoother.getSmoothedState(lag);
            x_smooth_hist.push_back(x_s);
        } else {
            // Placeholder
        }
    }

    // 3. Compute RMSE
    double mse_filt = 0.0;
    double mse_smooth = 0.0;
    int smooth_count = 0;

    std::ofstream file("ekf_results.csv");
    file << "t,true_pos,true_vel,meas,filt_pos,filt_vel,smooth_pos,smooth_vel\n";

    for (int k = 0; k < steps; ++k) {
        // Filter Error
        Eigen::VectorXf err_f = x_true_hist[k] - x_filt_hist[k];
        mse_filt += err_f.squaredNorm();

        if (k >= lag) {
            int time_idx = k - lag;
            // The x_smooth_hist was pushed when k >= lag.
            // The first entry in x_smooth_hist corresponds to k=lag, time_idx=0.
            int hist_idx = k - lag;

            Eigen::VectorXf x_s = x_smooth_hist[hist_idx];
            Eigen::VectorXf err_s = x_true_hist[time_idx] - x_s;
            mse_smooth += err_s.squaredNorm();
            smooth_count++;
        }
    }

    double rmse_filt = std::sqrt(mse_filt / steps);
    double rmse_smooth = (smooth_count > 0) ? std::sqrt(mse_smooth / smooth_count) : 0.0;

    std::cout << "Filter RMSE:   " << rmse_filt << std::endl;
    std::cout << "Smoother RMSE: " << rmse_smooth << " (Lag " << lag << ")" << std::endl;

    for (int i = 0; i < steps; ++i) {
        float t = time_hist[i];
        float tp = x_true_hist[i](0);
        float tv = x_true_hist[i](1);
        float m = y_meas_hist[i](0);

        float fp = x_filt_hist[i](0);
        float fv = x_filt_hist[i](1);

        float sp = 0.0f;
        float sv = 0.0f;

        if (i + lag < steps) {
            sp = x_smooth_hist[i](0);
            sv = x_smooth_hist[i](1);
        }

        file << t << "," << tp << "," << tv << "," << m << ","
             << fp << "," << fv << "," << sp << "," << sv << "\n";
    }

    file.close();
    std::cout << "Results saved to ekf_results.csv" << std::endl;

    return 0;
}
