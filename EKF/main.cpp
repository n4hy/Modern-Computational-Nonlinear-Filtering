#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <cstdlib>

#include "BallTossModel.h"
#include "FixedLagSmoother.h"
#include "FileUtils.h"

using namespace Eigen;
using namespace std;

void run_plotting_script(const std::string& csv_file, const std::string& title) {
    // Path: ../../scripts/plot_results.py assuming running from build/EKF
    std::string cmd = "python3 ../../scripts/plot_results.py " + csv_file + " --title \"" + title + "\"";
    std::cout << "Executing: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Plotting script failed with code " << ret << std::endl;
    }
}

double randn(double mean, double stddev, std::mt19937& gen) {
    std::normal_distribution<double> d(mean, stddev);
    return d(gen);
}
// --------------------------------

int main(int argc, char* argv[]) {
    bool use_graphics = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--graphics") == 0) {
            use_graphics = true;
        }
    }

    std::cout << "Starting EKF Ball Toss Simulation (Comparison)..." << std::endl;
    if (use_graphics) std::cout << "Graphics Enabled." << std::endl;

    double dt = 0.1;
    double T = 5.0;
    int steps = static_cast<int>(T / dt);
    double q_std = 0.01;
    double r_std = 0.5;
    int lag = 5;

    BallTossModel model(dt, q_std, r_std);

    VectorXd x_true(6);
    x_true << 0, 0, 0, 10, 5, 20;

    VectorXd x_est = x_true;
    x_est(0) += 2.0;
    x_est(3) -= 1.0;

    MatrixXd P_est = MatrixXd::Identity(6, 6) * 1.0;
    P_est(3,3) = 5.0; P_est(4,4) = 5.0; P_est(5,5) = 5.0;

    std::random_device rd;
    std::mt19937 gen(42);

    FixedLagSmoother smoother(&model, x_est, P_est, lag);

    std::vector<VectorXd> true_history;
    std::vector<VectorXd> smooth_history;
    std::vector<VectorXd> filt_history;
    std::vector<MatrixXd> smooth_cov_history;
    std::vector<MatrixXd> filt_cov_history;

    // Simulate Loop
    for (int k = 0; k < steps; ++k) {
        VectorXd w(6);
        for(int i=0; i<6; ++i) w(i) = randn(0, q_std, gen);
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
            smooth_cov_history.push_back(P_out);
            filt_cov_history.push_back(P_filt);
        }
    }

    // Flush
    VectorXd x_out, x_filt;
    MatrixXd P_out, P_filt;
    while(smoother.flush(x_out, P_out)) {
        smooth_history.push_back(x_out);
        filt_history.push_back(x_out);
        smooth_cov_history.push_back(P_out);
        filt_cov_history.push_back(P_out); // Fallback
    }

    // Alignment
    VectorXd x_init_true(6);
    x_init_true << 0, 0, 0, 10, 5, 20;
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

        if (i % 10 == 0) {
            std::cout << "t=" << std::fixed << std::setprecision(2) << i*dt << " | "
                      << true_history[i](2) << " | "
                      << filt_history[i](2) << " (" << norm_f << ") | "
                      << smooth_history[i](2) << " (" << norm_s << ")"
                      << std::endl;
        }
    }

    if (use_graphics) {
        save_to_csv("ekf_results.csv", dt, true_history, filt_history, smooth_history, filt_cov_history, smooth_cov_history);
        run_plotting_script("ekf_results.csv", "EKF Ball Toss Results");
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
