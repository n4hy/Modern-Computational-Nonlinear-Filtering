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
#include <optmath/neon_kernels.hpp>

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

    // Consistency metrics (NEES - Normalized Estimation Error Squared)
    float mean_nees = 0.0f;
    float std_nees = 0.0f;

    // Performance metrics
    float avg_step_time_ms = 0.0f;
    float total_time_ms = 0.0f;

    // Convergence metrics
    float convergence_time = 0.0f;  // Time to reach steady-state error
    int num_divergences = 0;         // Number of times error exceeded threshold

    void print() const {
        std::cout << "\n=== " << filter_name << " on " << problem_name << " ===" << std::endl;
        std::cout << "Filtered RMSE: " << rmse_overall << std::endl;
        std::cout << "  Position: " << rmse_position << std::endl;
        std::cout << "  Velocity: " << rmse_velocity << std::endl;

        if (rmse_smoothed_overall > 0) {
            std::cout << "Smoothed RMSE: " << rmse_smoothed_overall << std::endl;
            std::cout << "  Position: " << rmse_smoothed_position << std::endl;
            std::cout << "  Velocity: " << rmse_smoothed_velocity << std::endl;
        }

        std::cout << "NEES: " << mean_nees << " ± " << std_nees << std::endl;
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
             << mean_nees << ","
             << std_nees << ","
             << avg_step_time_ms << ","
             << total_time_ms << ","
             << convergence_time << ","
             << num_divergences << std::endl;
    }

    static void write_csv_header(std::ofstream& file) {
        file << "Filter,Problem,RMSE_Overall,RMSE_Position,RMSE_Velocity,"
             << "RMSE_Smoothed_Overall,RMSE_Smoothed_Position,RMSE_Smoothed_Velocity,"
             << "Mean_NEES,Std_NEES,Avg_Step_Time_ms,Total_Time_ms,"
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
 * Compute Normalized Estimation Error Squared (NEES)
 * NEES should be chi-squared distributed with n degrees of freedom
 * Mean should be ~n, where n is state dimension
 */
template<typename State>
std::pair<float, float> compute_nees(const std::vector<State>& true_states,
                                     const std::vector<State>& estimated_states,
                                     const std::vector<Eigen::MatrixXf>& covs) {
    if (true_states.size() != estimated_states.size() ||
        true_states.size() != covs.size()) {
        return {-1.0f, -1.0f};
    }

    std::vector<float> nees_values;
    nees_values.reserve(true_states.size());

    for (size_t i = 0; i < true_states.size(); ++i) {
        State error = estimated_states[i] - true_states[i];
        Eigen::MatrixXf P_inv = optmath::neon::neon_inverse(covs[i]);
        if (P_inv.size() == 0) {
            P_inv = covs[i].inverse();  // Fallback
        }
        float nees = error.transpose() * P_inv * error;
        nees_values.push_back(nees);
    }

    // Compute mean and std
    float mean = 0.0f;
    for (float val : nees_values) {
        mean += val;
    }
    mean /= nees_values.size();

    float variance = 0.0f;
    for (float val : nees_values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= nees_values.size();

    return {mean, std::sqrt(variance)};
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
