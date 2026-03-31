#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include "FilterMath.h"

namespace Benchmark {

struct BenchmarkMetrics {
    std::string filter_name;
    std::string problem_name;

    // Error metrics
    float rmse_position = 0.0f;
    float rmse_velocity = 0.0f;
    float rmse_overall = 0.0f;

    // Smoothed metrics (if applicable)
    float rmse_smoothed_position = 0.0f;
    float rmse_smoothed_velocity = 0.0f;
    float rmse_smoothed_overall = 0.0f;

    // NEES consistency metrics
    float median_nees = 0.0f;
    float pct_in_bounds = 0.0f;  // % of steps within 95% chi-squared bounds
    float chi2_lower = 0.0f;     // 95% chi-squared lower bound
    float chi2_upper = 0.0f;     // 95% chi-squared upper bound
    int nees_valid = 0;           // Steps with well-conditioned P
    int nees_total = 0;           // Total steps evaluated (post burn-in)

    // Performance metrics
    float avg_step_time_ms = 0.0f;
    float total_time_ms = 0.0f;

    // Convergence metrics
    float convergence_time = 0.0f;  // Time to reach steady-state error
    int num_divergences = 0;         // Number of times error exceeded threshold

    void print() const {
        std::cout << std::setprecision(6);
        std::cout << "\n=== " << filter_name << " on " << problem_name << " ===" << std::endl;
        std::cout << "Filtered RMSE: " << rmse_overall << std::endl;
        std::cout << "  Position: " << rmse_position << std::endl;
        std::cout << "  Velocity: " << rmse_velocity << std::endl;

        if (rmse_smoothed_overall > 0) {
            std::cout << "Smoothed RMSE: " << rmse_smoothed_overall << std::endl;
            std::cout << "  Position: " << rmse_smoothed_position << std::endl;
            std::cout << "  Velocity: " << rmse_smoothed_velocity << std::endl;
        }

        // NEES consistency line
        std::cout << "NEES: median=" << std::setprecision(4) << median_nees;
        if (pct_in_bounds >= 0.0f) {
            std::cout << std::fixed << std::setprecision(1)
                      << "  in-95%-bounds: " << pct_in_bounds << "%"
                      << " (" << nees_valid << "/" << nees_total << " valid)"
                      << std::setprecision(2)
                      << "  chi2_95=[" << chi2_lower << ", " << chi2_upper << "]";
        }
        std::cout << std::defaultfloat << std::setprecision(6) << std::endl;

        std::cout << "Avg Step Time: " << avg_step_time_ms << " ms" << std::endl;
        std::cout << "Total Time: " << total_time_ms << " ms" << std::endl;
        std::cout << "Convergence Time: " << convergence_time << " s" << std::endl;
        std::cout << "Divergences: " << num_divergences << std::endl;
    }

    void save_to_csv(std::ofstream& file) const {
        file << filter_name << ","
             << problem_name << ","
             << rmse_overall << ","
             << rmse_position << ","
             << rmse_velocity << ","
             << rmse_smoothed_overall << ","
             << rmse_smoothed_position << ","
             << rmse_smoothed_velocity << ","
             << median_nees << ","
             << pct_in_bounds << ","
             << nees_valid << ","
             << nees_total << ","
             << avg_step_time_ms << ","
             << total_time_ms << ","
             << convergence_time << ","
             << num_divergences << std::endl;
    }

    static void write_csv_header(std::ofstream& file) {
        file << "Filter,Problem,RMSE_Overall,RMSE_Position,RMSE_Velocity,"
             << "RMSE_Smoothed_Overall,RMSE_Smoothed_Position,RMSE_Smoothed_Velocity,"
             << "Median_NEES,Pct_In_Bounds,NEES_Valid,NEES_Total,"
             << "Avg_Step_Time_ms,Total_Time_ms,"
             << "Convergence_Time,Num_Divergences" << std::endl;
    }
};

template<typename State, typename Observation>
struct TrajectoryData {
    std::vector<float> times;
    std::vector<State> true_states;
    std::vector<Observation> measurements;
    std::vector<State> filtered_states;
    std::vector<State> smoothed_states;
    std::vector<Eigen::MatrixXf> filtered_covs;
    std::vector<Eigen::MatrixXf> smoothed_covs;
};

/**
 * Compute RMSE between two state trajectories
 */
template<typename State>
float compute_rmse(const std::vector<State>& true_states,
                   const std::vector<State>& estimated_states) {
    if (true_states.size() != estimated_states.size()) {
        std::cerr << "Warning: trajectory sizes don't match" << std::endl;
        return -1.0f;
    }

    float sum_sq_error = 0.0f;
    int count = 0;

    for (size_t i = 0; i < true_states.size(); ++i) {
        State error = true_states[i] - estimated_states[i];
        sum_sq_error += error.squaredNorm();
        count++;
    }

    return std::sqrt(sum_sq_error / count);
}

/**
 * Compute RMSE for specific state indices (e.g., just positions or velocities)
 */
template<typename State>
float compute_rmse_indices(const std::vector<State>& true_states,
                           const std::vector<State>& estimated_states,
                           const std::vector<int>& indices) {
    if (true_states.size() != estimated_states.size()) {
        return -1.0f;
    }

    float sum_sq_error = 0.0f;
    int count = 0;

    for (size_t i = 0; i < true_states.size(); ++i) {
        for (int idx : indices) {
            float error = true_states[i](idx) - estimated_states[i](idx);
            sum_sq_error += error * error;
            count++;
        }
    }

    return std::sqrt(sum_sq_error / count);
}

/**
 * Wilson-Hilferty approximation for chi-squared quantiles.
 * Returns chi2_p(n) for probability p with n degrees of freedom.
 */
inline float chi2_quantile(float p, int n) {
    // z_p from inverse normal CDF (hardcode common values)
    float z;
    if (p <= 0.025f) z = -1.96f;
    else if (p >= 0.975f) z = 1.96f;
    else if (p <= 0.05f) z = -1.645f;
    else if (p >= 0.95f) z = 1.645f;
    else z = 0.0f;

    float nf = static_cast<float>(n);
    float cube = 1.0f - 2.0f / (9.0f * nf) + z * std::sqrt(2.0f / (9.0f * nf));
    return nf * cube * cube * cube;
}

/**
 * Compute NEES (Normalized Estimation Error Squared) with robust statistics.
 *
 * Returns a struct-like tuple: {median_nees, pct_in_bounds, chi2_lower, chi2_upper, n_valid, n_total}
 *
 * Key improvements over naive NEES:
 *  - Skips burn-in period (first 10% of trajectory)
 *  - Checks covariance condition number; skips ill-conditioned steps
 *  - Uses LDLT solve instead of explicit inverse for numerical stability
 *  - Reports median (robust to outliers) instead of mean
 *  - Reports % within 95% chi-squared bounds (the standard consistency test)
 */
template<typename State>
void compute_nees(const std::vector<State>& true_states,
                  const std::vector<State>& estimated_states,
                  const std::vector<Eigen::MatrixXf>& covs,
                  BenchmarkMetrics& metrics) {
    const int n = true_states[0].size();  // state dimension

    if (true_states.size() != estimated_states.size() ||
        true_states.size() != covs.size() ||
        true_states.size() < 20) {
        metrics.median_nees = -1.0f;
        metrics.pct_in_bounds = 0.0f;
        return;
    }

    // Chi-squared 95% bounds for n degrees of freedom
    float lower = chi2_quantile(0.025f, n);
    float upper = chi2_quantile(0.975f, n);
    metrics.chi2_lower = lower;
    metrics.chi2_upper = upper;

    // Skip first 10% as burn-in
    size_t burn_in = true_states.size() / 10;
    size_t total = true_states.size() - burn_in;
    metrics.nees_total = static_cast<int>(total);

    // Condition number threshold: if cond(P) > this, skip the step
    const float max_condition = 1e8f;

    std::vector<float> nees_values;
    nees_values.reserve(total);

    for (size_t i = burn_in; i < true_states.size(); ++i) {
        const Eigen::MatrixXf& P = covs[i];

        // Check for degenerate covariance
        float min_diag = std::numeric_limits<float>::max();
        float max_diag = 0.0f;
        bool bad_diag = false;
        for (int d = 0; d < P.rows(); ++d) {
            float pd = P(d, d);
            if (pd <= 0.0f || !std::isfinite(pd)) {
                bad_diag = true;
                break;
            }
            min_diag = std::min(min_diag, pd);
            max_diag = std::max(max_diag, pd);
        }
        if (bad_diag || min_diag <= 0.0f) continue;

        // Diagonal condition number check (cheap proxy for full condition number)
        if (max_diag / min_diag > max_condition) continue;

        // Use LDLT for robust solve: NEES = e^T P^{-1} e = (P^{-1} e)^T e
        Eigen::VectorXf error = (estimated_states[i] - true_states[i]).template cast<float>();
        Eigen::LDLT<Eigen::MatrixXf> ldlt(P);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) continue;

        Eigen::VectorXf P_inv_e = ldlt.solve(error);
        float nees = error.dot(P_inv_e);

        if (!std::isfinite(nees) || nees < 0.0f) continue;

        nees_values.push_back(nees);
    }

    metrics.nees_valid = static_cast<int>(nees_values.size());

    if (nees_values.empty()) {
        metrics.median_nees = -1.0f;
        metrics.pct_in_bounds = 0.0f;
        return;
    }

    // Median
    std::sort(nees_values.begin(), nees_values.end());
    size_t mid = nees_values.size() / 2;
    metrics.median_nees = (nees_values.size() % 2 == 0)
        ? 0.5f * (nees_values[mid - 1] + nees_values[mid])
        : nees_values[mid];

    // Percentage within chi-squared 95% bounds
    int in_bounds = 0;
    for (float val : nees_values) {
        if (val >= lower && val <= upper) {
            in_bounds++;
        }
    }
    metrics.pct_in_bounds = 100.0f * static_cast<float>(in_bounds) / static_cast<float>(nees_values.size());
}

/**
 * Detect convergence time - when error falls below threshold and stays there
 */
template<typename State>
float compute_convergence_time(const std::vector<float>& times,
                                const std::vector<State>& true_states,
                                const std::vector<State>& estimated_states,
                                float threshold = 0.5f) {
    if (true_states.size() != estimated_states.size()) {
        return -1.0f;
    }

    const int window_size = 50;  // Must stay below threshold for this many steps

    for (size_t i = window_size; i < true_states.size(); ++i) {
        bool converged = true;
        for (int j = 0; j < window_size; ++j) {
            State error = true_states[i-j] - estimated_states[i-j];
            if (error.norm() > threshold) {
                converged = false;
                break;
            }
        }
        if (converged) {
            return times[i];
        }
    }

    return times.back();  // Didn't converge
}

/**
 * Count divergences - times when error exceeded a large threshold
 */
template<typename State>
int count_divergences(const std::vector<State>& true_states,
                      const std::vector<State>& estimated_states,
                      float threshold = 10.0f) {
    int count = 0;
    for (size_t i = 0; i < true_states.size(); ++i) {
        State error = true_states[i] - estimated_states[i];
        if (error.norm() > threshold) {
            count++;
        }
    }
    return count;
}

/**
 * Save full trajectory to CSV for detailed analysis
 */
template<typename State, typename Observation>
void save_trajectory_csv(const std::string& filename,
                         const TrajectoryData<State, Observation>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    int state_dim = data.true_states[0].size();

    // Header
    file << "time";
    for (int i = 0; i < state_dim; ++i) {
        file << ",true_x" << i;
    }
    for (int i = 0; i < state_dim; ++i) {
        file << ",filt_x" << i;
    }
    if (!data.smoothed_states.empty()) {
        for (int i = 0; i < state_dim; ++i) {
            file << ",smooth_x" << i;
        }
    }
    file << std::endl;

    // Data
    for (size_t i = 0; i < data.times.size(); ++i) {
        file << data.times[i];

        for (int j = 0; j < state_dim; ++j) {
            file << "," << data.true_states[i](j);
        }
        for (int j = 0; j < state_dim; ++j) {
            file << "," << data.filtered_states[i](j);
        }
        if (!data.smoothed_states.empty()) {
            for (int j = 0; j < state_dim; ++j) {
                file << "," << data.smoothed_states[i](j);
            }
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

} // namespace Benchmark

#endif // BENCHMARK_RUNNER_H
