/**
 * @file MonteCarloRunner.h
 * @brief Monte Carlo Simulation Framework for Navigation Analysis
 *
 * Provides:
 * - Parallel execution of multiple simulation trials
 * - Statistical analysis of filter performance
 * - Convergence time computation
 * - Divergence detection and counting
 * - CSV output for post-processing
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

namespace AircraftNav {

/**
 * @brief Monte Carlo configuration
 */
struct MonteCarloConfig {
    int num_trials = 1000;
    double simulation_duration_s = 300.0;
    double gps_outage_start_s = 60.0;
    double gps_outage_duration_s = 30.0;

    bool parallel_execution = true;
    int num_threads = 0;  // 0 = auto-detect

    // Convergence criteria
    double convergence_threshold_m = 500.0;
    double convergence_window_s = 10.0;

    // Divergence criteria
    double divergence_threshold_m = 5000.0;

    // Output configuration
    std::string output_directory = ".";
    bool save_trajectories = false;  // Full trajectory for each trial
    bool verbose = false;

    // Random seed (0 = random)
    uint64_t base_seed = 0;
};

/**
 * @brief Per-trial results
 */
struct TrialResult {
    int trial_id;
    uint64_t seed;

    // Position errors at key times
    double error_at_outage_start_m;
    double error_at_outage_end_m;
    double final_error_m;
    double max_error_m;

    // Convergence
    double convergence_time_s;  // Time after outage to reach threshold
    bool converged;
    bool diverged;

    // RMSE over phases
    double rmse_gps_phase_m;
    double rmse_outage_phase_m;
    double rmse_recovery_phase_m;

    // Velocity errors
    double max_velocity_error_mps;
    double final_velocity_error_mps;

    // Number of Iridium measurements used
    int num_iridium_measurements;
    int num_visible_satellites_avg;
};

/**
 * @brief Aggregate Monte Carlo statistics
 */
struct MonteCarloStatistics {
    // Position error statistics
    double mean_final_error_m;
    double std_final_error_m;
    double median_final_error_m;
    double percentile_95_error_m;
    double percentile_99_error_m;
    double max_final_error_m;

    // Convergence statistics
    double mean_convergence_time_s;
    double std_convergence_time_s;
    double median_convergence_time_s;
    int num_converged;
    int num_diverged;
    double convergence_rate;  // Fraction that converged

    // RMSE by phase
    double mean_rmse_gps_m;
    double mean_rmse_outage_m;
    double mean_rmse_recovery_m;

    // Overall metrics
    int total_trials;
    double total_runtime_s;
};

/**
 * @brief Simulation trial function signature
 *
 * Takes trial ID and seed, returns TrialResult
 */
using TrialFunction = std::function<TrialResult(int, uint64_t)>;

/**
 * @brief Monte Carlo simulation runner
 *
 * Executes multiple simulation trials in parallel and computes
 * aggregate statistics for filter performance analysis.
 */
class MonteCarloRunner {
public:
    explicit MonteCarloRunner(const MonteCarloConfig& config)
        : config_(config)
    {
        if (config_.num_threads == 0) {
            config_.num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        }

        if (config_.base_seed == 0) {
            config_.base_seed = std::random_device{}();
        }
    }

    /**
     * @brief Run Monte Carlo simulation
     * @param trial_func Function to execute for each trial
     * @return Aggregate statistics
     */
    MonteCarloStatistics run(TrialFunction trial_func) {
        auto start_time = std::chrono::high_resolution_clock::now();

        results_.clear();
        results_.resize(config_.num_trials);

        if (config_.parallel_execution) {
            runParallel(trial_func);
        } else {
            runSequential(trial_func);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(end_time - start_time).count();

        MonteCarloStatistics stats = computeStatistics();
        stats.total_runtime_s = runtime;

        return stats;
    }

    /**
     * @brief Get individual trial results
     */
    const std::vector<TrialResult>& getResults() const { return results_; }

    /**
     * @brief Save results to CSV
     */
    void saveSummaryCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Header
        file << "trial_id,seed,error_outage_start,error_outage_end,final_error,max_error,"
             << "convergence_time,converged,diverged,"
             << "rmse_gps,rmse_outage,rmse_recovery,"
             << "max_vel_error,final_vel_error,num_iridium_meas,avg_visible_sats\n";

        // Data
        for (const auto& r : results_) {
            file << r.trial_id << ","
                 << r.seed << ","
                 << r.error_at_outage_start_m << ","
                 << r.error_at_outage_end_m << ","
                 << r.final_error_m << ","
                 << r.max_error_m << ","
                 << r.convergence_time_s << ","
                 << (r.converged ? 1 : 0) << ","
                 << (r.diverged ? 1 : 0) << ","
                 << r.rmse_gps_phase_m << ","
                 << r.rmse_outage_phase_m << ","
                 << r.rmse_recovery_phase_m << ","
                 << r.max_velocity_error_mps << ","
                 << r.final_velocity_error_mps << ","
                 << r.num_iridium_measurements << ","
                 << r.num_visible_satellites_avg << "\n";
        }

        file.close();
        std::cout << "Saved " << results_.size() << " trial results to " << filename << std::endl;
    }

    /**
     * @brief Save aggregate statistics to CSV
     */
    void saveStatisticsCSV(const std::string& filename, const MonteCarloStatistics& stats) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        file << "metric,value\n";
        file << "total_trials," << stats.total_trials << "\n";
        file << "total_runtime_s," << stats.total_runtime_s << "\n";
        file << "mean_final_error_m," << stats.mean_final_error_m << "\n";
        file << "std_final_error_m," << stats.std_final_error_m << "\n";
        file << "median_final_error_m," << stats.median_final_error_m << "\n";
        file << "percentile_95_error_m," << stats.percentile_95_error_m << "\n";
        file << "percentile_99_error_m," << stats.percentile_99_error_m << "\n";
        file << "max_final_error_m," << stats.max_final_error_m << "\n";
        file << "mean_convergence_time_s," << stats.mean_convergence_time_s << "\n";
        file << "std_convergence_time_s," << stats.std_convergence_time_s << "\n";
        file << "median_convergence_time_s," << stats.median_convergence_time_s << "\n";
        file << "num_converged," << stats.num_converged << "\n";
        file << "num_diverged," << stats.num_diverged << "\n";
        file << "convergence_rate," << stats.convergence_rate << "\n";
        file << "mean_rmse_gps_m," << stats.mean_rmse_gps_m << "\n";
        file << "mean_rmse_outage_m," << stats.mean_rmse_outage_m << "\n";
        file << "mean_rmse_recovery_m," << stats.mean_rmse_recovery_m << "\n";

        file.close();
        std::cout << "Saved statistics to " << filename << std::endl;
    }

    /**
     * @brief Print statistics summary
     */
    void printStatistics(const MonteCarloStatistics& stats) const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║            MONTE CARLO SIMULATION RESULTS                        ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Total trials:           " << std::setw(6) << stats.total_trials
                  << "                                  ║\n";
        std::cout << "║ Runtime:                " << std::setw(6) << std::fixed << std::setprecision(1)
                  << stats.total_runtime_s << " s                                ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ POSITION ERROR                                                   ║\n";
        std::cout << "║   Mean final error:     " << std::setw(8) << std::fixed << std::setprecision(1)
                  << stats.mean_final_error_m << " m                            ║\n";
        std::cout << "║   Std dev:              " << std::setw(8) << stats.std_final_error_m
                  << " m                            ║\n";
        std::cout << "║   Median:               " << std::setw(8) << stats.median_final_error_m
                  << " m                            ║\n";
        std::cout << "║   95th percentile:      " << std::setw(8) << stats.percentile_95_error_m
                  << " m                            ║\n";
        std::cout << "║   99th percentile:      " << std::setw(8) << stats.percentile_99_error_m
                  << " m                            ║\n";
        std::cout << "║   Maximum:              " << std::setw(8) << stats.max_final_error_m
                  << " m                            ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ CONVERGENCE                                                      ║\n";
        std::cout << "║   Mean time:            " << std::setw(8) << std::fixed << std::setprecision(1)
                  << stats.mean_convergence_time_s << " s                            ║\n";
        std::cout << "║   Median time:          " << std::setw(8) << stats.median_convergence_time_s
                  << " s                            ║\n";
        std::cout << "║   Converged:            " << std::setw(6) << stats.num_converged
                  << " (" << std::setw(5) << std::setprecision(1)
                  << stats.convergence_rate * 100 << "%)                     ║\n";
        std::cout << "║   Diverged:             " << std::setw(6) << stats.num_diverged
                  << "                                  ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ RMSE BY PHASE                                                    ║\n";
        std::cout << "║   GPS phase:            " << std::setw(8) << std::setprecision(2)
                  << stats.mean_rmse_gps_m << " m                            ║\n";
        std::cout << "║   Outage phase:         " << std::setw(8)
                  << stats.mean_rmse_outage_m << " m                            ║\n";
        std::cout << "║   Recovery phase:       " << std::setw(8)
                  << stats.mean_rmse_recovery_m << " m                            ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

private:
    MonteCarloConfig config_;
    std::vector<TrialResult> results_;
    std::mutex results_mutex_;
    std::atomic<int> completed_trials_{0};

    void runSequential(TrialFunction trial_func) {
        for (int i = 0; i < config_.num_trials; ++i) {
            uint64_t seed = config_.base_seed + i;
            results_[i] = trial_func(i, seed);

            if (config_.verbose && (i + 1) % 100 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << config_.num_trials << " trials\n";
            }
        }
    }

    void runParallel(TrialFunction trial_func) {
        std::vector<std::thread> threads;

        auto worker = [this, &trial_func](int start, int end) {
            for (int i = start; i < end; ++i) {
                uint64_t seed = config_.base_seed + i;
                TrialResult result = trial_func(i, seed);

                {
                    std::lock_guard<std::mutex> lock(results_mutex_);
                    results_[i] = result;
                }

                int completed = ++completed_trials_;
                if (config_.verbose && completed % 100 == 0) {
                    std::cout << "Completed " << completed << "/" << config_.num_trials << " trials\n";
                }
            }
        };

        int trials_per_thread = config_.num_trials / config_.num_threads;
        int remainder = config_.num_trials % config_.num_threads;

        int start = 0;
        for (int t = 0; t < config_.num_threads; ++t) {
            int count = trials_per_thread + (t < remainder ? 1 : 0);
            threads.emplace_back(worker, start, start + count);
            start += count;
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    MonteCarloStatistics computeStatistics() const {
        MonteCarloStatistics stats;
        stats.total_trials = static_cast<int>(results_.size());

        if (results_.empty()) {
            return stats;
        }

        // Collect final errors
        std::vector<double> final_errors;
        std::vector<double> convergence_times;
        double sum_rmse_gps = 0.0;
        double sum_rmse_outage = 0.0;
        double sum_rmse_recovery = 0.0;
        stats.num_converged = 0;
        stats.num_diverged = 0;

        for (const auto& r : results_) {
            final_errors.push_back(r.final_error_m);

            if (r.converged) {
                stats.num_converged++;
                convergence_times.push_back(r.convergence_time_s);
            }
            if (r.diverged) {
                stats.num_diverged++;
            }

            sum_rmse_gps += r.rmse_gps_phase_m;
            sum_rmse_outage += r.rmse_outage_phase_m;
            sum_rmse_recovery += r.rmse_recovery_phase_m;
        }

        // Sort for percentiles
        std::sort(final_errors.begin(), final_errors.end());

        // Mean and std dev
        double sum = std::accumulate(final_errors.begin(), final_errors.end(), 0.0);
        stats.mean_final_error_m = sum / final_errors.size();

        double sq_sum = 0.0;
        for (double e : final_errors) {
            sq_sum += (e - stats.mean_final_error_m) * (e - stats.mean_final_error_m);
        }
        stats.std_final_error_m = std::sqrt(sq_sum / final_errors.size());

        // Percentiles
        stats.median_final_error_m = final_errors[final_errors.size() / 2];
        stats.percentile_95_error_m = final_errors[static_cast<size_t>(0.95 * final_errors.size())];
        stats.percentile_99_error_m = final_errors[static_cast<size_t>(0.99 * final_errors.size())];
        stats.max_final_error_m = final_errors.back();

        // Convergence statistics
        if (!convergence_times.empty()) {
            std::sort(convergence_times.begin(), convergence_times.end());
            double ct_sum = std::accumulate(convergence_times.begin(), convergence_times.end(), 0.0);
            stats.mean_convergence_time_s = ct_sum / convergence_times.size();

            double ct_sq_sum = 0.0;
            for (double t : convergence_times) {
                ct_sq_sum += (t - stats.mean_convergence_time_s) * (t - stats.mean_convergence_time_s);
            }
            stats.std_convergence_time_s = std::sqrt(ct_sq_sum / convergence_times.size());
            stats.median_convergence_time_s = convergence_times[convergence_times.size() / 2];
        }

        stats.convergence_rate = static_cast<double>(stats.num_converged) / stats.total_trials;

        // RMSE by phase
        stats.mean_rmse_gps_m = sum_rmse_gps / results_.size();
        stats.mean_rmse_outage_m = sum_rmse_outage / results_.size();
        stats.mean_rmse_recovery_m = sum_rmse_recovery / results_.size();

        return stats;
    }
};

} // namespace AircraftNav
