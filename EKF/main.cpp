#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include "EKF.h"
#include "EKFFixedLag.h"
#include "NonlinearOscillator.h"

// Helper to generate noise
double randn(double mean, double stddev) {
    static std::mt19937 gen(42);
    std::normal_distribution<> d(mean, stddev);
    return d(gen);
}

int main() {
    // 1. Setup Simulation
    double dt = 0.05;
    int steps = 200;
    NonlinearOscillator model(dt);

    Eigen::VectorXd x_true(2);
    x_true << 1.0, 0.0; // Initial state: pos=1, vel=0

    // Arrays to store history
    std::vector<double> time_hist;
    std::vector<Eigen::VectorXd> x_true_hist;
    std::vector<Eigen::VectorXd> y_meas_hist;

    std::cout << "Generating Data..." << std::endl;
    for (int k = 0; k < steps; ++k) {
        double t = k * dt;

        // Propagate truth (no process noise in truth generation for smoother curve, or add it?)
        // Let's add process noise to make it interesting for the filter
        Eigen::VectorXd u(0); // No control
        Eigen::VectorXd w(2);
        w << randn(0, 0.01), randn(0, 0.01);

        x_true = model.f(x_true, u, t) + w;

        // Generate measurement
        Eigen::VectorXd v(1);
        v << randn(0, 0.1);
        Eigen::VectorXd y = model.h(x_true, t) + v;

        time_hist.push_back(t);
        x_true_hist.push_back(x_true);
        y_meas_hist.push_back(y);
    }

    // 2. Setup Filter and Smoother
    int lag = 10;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2); // Imperfect initialization
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2) * 1.0;

    EKFFixedLag smoother(&model, x0, P0, lag);

    // Store results
    std::vector<Eigen::VectorXd> x_filt_hist;
    std::vector<Eigen::VectorXd> x_smooth_hist;

    std::cout << "Running EKF + Smoother..." << std::endl;
    for (int k = 0; k < steps; ++k) {
        Eigen::VectorXd u(0);
        smoother.step(y_meas_hist[k], u, time_hist[k]);

        auto [x_filt, P_filt] = smoother.getFilteredState();
        // Get smoothed state at lag L (returns x_{k-L|k})
        // For plotting, we want to align time.
        // x_smooth_hist will store the *final* smoothed estimate for each time step.
        // But the smoother only gives us the estimate at k-L fully smoothed.
        // Actually, we can retrieve smoothed estimate for *current* time k (which is same as filtered)
        // or any time in window.
        // To build a "Best Estimate" trajectory, we usually wait until we have the full lag.
        // So at step k, we get the smoothed estimate for k-lag.

        x_filt_hist.push_back(x_filt);

        // We will store the smoothed estimate for time k-lag
        // If k < lag, we don't have a fully smoothed estimate for k-lag yet (it's negative time).
        // Let's just store what we have.
        // A common way to plot is to plot x_{t|T} (fixed interval) or x_{t|t+L} (fixed lag).
        // Here we build the x_{t|t+L} trajectory.

        if (k >= lag) {
            auto [x_s, P_s] = smoother.getSmoothedState(lag);
            x_smooth_hist.push_back(x_s);
        } else {
            // Placeholder or wait?
            // Let's push back zero or handled later
            // We will just offset the plot logic.
        }
    }

    // 3. Compute RMSE
    // Compare x_true[k] with x_filt[k]
    // Compare x_true[k-lag] with x_smooth[k-lag|k] (which is what we stored)

    double mse_filt = 0.0;
    double mse_smooth = 0.0;
    int smooth_count = 0;

    std::ofstream file("ekf_results.csv");
    file << "t,true_pos,true_vel,meas,filt_pos,filt_vel,smooth_pos,smooth_vel\n";

    for (int k = 0; k < steps; ++k) {
        // Filter Error
        Eigen::VectorXd err_f = x_true_hist[k] - x_filt_hist[k];
        mse_filt += err_f.squaredNorm();

        double smooth_pos = 0.0; // NaN placeholder in CSV usually better, using 0 for now
        double smooth_vel = 0.0;

        if (k >= lag) {
            int time_idx = k - lag;
            // The x_smooth_hist was pushed when k >= lag.
            // The first entry in x_smooth_hist corresponds to k=lag, time_idx=0.
            int hist_idx = k - lag;

            Eigen::VectorXd x_s = x_smooth_hist[hist_idx];
            Eigen::VectorXd err_s = x_true_hist[time_idx] - x_s;
            mse_smooth += err_s.squaredNorm();
            smooth_count++;

            smooth_pos = x_s(0);
            smooth_vel = x_s(1);

            // For CSV, we want to align rows by time t.
            // But we are iterating k (current time).
            // Let's write the CSV based on time index 't' where we have data.
        }
    }

    double rmse_filt = std::sqrt(mse_filt / steps);
    double rmse_smooth = (smooth_count > 0) ? std::sqrt(mse_smooth / smooth_count) : 0.0;

    std::cout << "Filter RMSE:   " << rmse_filt << std::endl;
    std::cout << "Smoother RMSE: " << rmse_smooth << " (Lag " << lag << ")" << std::endl;

    // Rewrite CSV properly aligned by time index 'i'
    // We have:
    // x_true_hist[i] for i = 0..steps-1
    // x_filt_hist[i] for i = 0..steps-1 (Estimate of x_i given y_0..y_i)
    // x_smooth_hist has entry for i = 0..(steps-1-lag).
    //   x_smooth_hist[0] is estimate of x_0 given y_0..y_L

    // We can only plot smoothed data up to steps-1-lag.

    for (int i = 0; i < steps; ++i) {
        double t = time_hist[i];
        double tp = x_true_hist[i](0);
        double tv = x_true_hist[i](1);
        double m = y_meas_hist[i](0);

        double fp = x_filt_hist[i](0);
        double fv = x_filt_hist[i](1);

        double sp = 0.0;
        double sv = 0.0;

        // Do we have a smoothed estimate for time i?
        // We get smoothed estimate for time i when simulation reaches i + lag.
        // So we need k = i + lag < steps.
        if (i + lag < steps) {
            // The value was stored in x_smooth_hist[k - lag] = x_smooth_hist[i]
            sp = x_smooth_hist[i](0);
            sv = x_smooth_hist[i](1);
        } else {
            // Use filtered as fallback or mark empty?
            // Let's use filtered for the "tail" where lag window isn't full yet?
            // Or just leave 0. Let's leave 0 to see the lag effect.
        }

        file << t << "," << tp << "," << tv << "," << m << ","
             << fp << "," << fv << "," << sp << "," << sv << "\n";
    }

    file.close();
    std::cout << "Results saved to ekf_results.csv" << std::endl;

    return 0;
}
