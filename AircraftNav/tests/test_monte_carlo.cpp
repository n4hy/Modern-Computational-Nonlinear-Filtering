/**
 * @file test_monte_carlo.cpp
 * @brief Unit tests for Monte Carlo framework
 */

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>

#include "MonteCarloRunner.h"

using namespace AircraftNav;

class MonteCarloTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.num_trials = 10;  // Small for testing
        config_.parallel_execution = false;  // Easier to debug
        config_.verbose = false;
    }

    MonteCarloConfig config_;
};

TEST_F(MonteCarloTest, BasicExecution) {
    MonteCarloRunner runner(config_);

    // Simple trial function that returns deterministic results
    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0 + trial_id;
        result.convergence_time_s = 30.0;
        result.converged = true;
        result.diverged = false;
        result.rmse_gps_phase_m = 5.0;
        result.rmse_outage_phase_m = 200.0;
        result.rmse_recovery_phase_m = 50.0;
        result.max_error_m = 300.0;
        result.max_velocity_error_mps = 1.0;
        result.final_velocity_error_mps = 0.5;
        result.num_iridium_measurements = 100;
        result.num_visible_satellites_avg = 3;
        return result;
    };

    MonteCarloStatistics stats = runner.run(trial_func);

    EXPECT_EQ(stats.total_trials, config_.num_trials);
    EXPECT_EQ(stats.num_converged, config_.num_trials);
    EXPECT_EQ(stats.num_diverged, 0);
}

TEST_F(MonteCarloTest, Statistics) {
    MonteCarloRunner runner(config_);

    // Trial function with known statistics
    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        // Final error: 100, 101, 102, ..., 109
        result.final_error_m = 100.0 + trial_id;
        result.convergence_time_s = 30.0;
        result.converged = true;
        result.diverged = false;
        result.rmse_gps_phase_m = 5.0;
        result.rmse_outage_phase_m = 200.0;
        result.rmse_recovery_phase_m = 50.0;
        return result;
    };

    MonteCarloStatistics stats = runner.run(trial_func);

    // Mean should be 104.5
    EXPECT_NEAR(stats.mean_final_error_m, 104.5, 0.1);

    // Median should be 104 or 105
    EXPECT_GE(stats.median_final_error_m, 104.0);
    EXPECT_LE(stats.median_final_error_m, 105.0);

    // Max should be 109
    EXPECT_NEAR(stats.max_final_error_m, 109.0, 0.1);

    // Convergence rate should be 100%
    EXPECT_NEAR(stats.convergence_rate, 1.0, 0.01);
}

TEST_F(MonteCarloTest, PartialConvergence) {
    MonteCarloRunner runner(config_);

    // Half converge, half don't
    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0;
        result.converged = (trial_id % 2 == 0);
        result.diverged = false;
        result.convergence_time_s = result.converged ? 30.0 : 1000.0;
        return result;
    };

    MonteCarloStatistics stats = runner.run(trial_func);

    // 5 out of 10 should converge (trials 0, 2, 4, 6, 8)
    EXPECT_EQ(stats.num_converged, 5);
    EXPECT_NEAR(stats.convergence_rate, 0.5, 0.01);
}

TEST_F(MonteCarloTest, Divergence) {
    MonteCarloRunner runner(config_);

    // One trial diverges
    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0;
        result.converged = true;
        result.diverged = (trial_id == 5);
        result.convergence_time_s = 30.0;
        return result;
    };

    MonteCarloStatistics stats = runner.run(trial_func);

    EXPECT_EQ(stats.num_diverged, 1);
}

TEST_F(MonteCarloTest, ParallelExecution) {
    config_.num_trials = 100;
    config_.parallel_execution = true;
    config_.num_threads = 4;

    MonteCarloRunner runner(config_);

    // Thread-safe trial function
    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        // Simulate some work
        double sum = 0.0;
        for (int i = 0; i < 1000; ++i) {
            sum += std::sin(seed + i);
        }

        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0 + sum * 0.001;  // Use result to prevent optimization
        result.converged = true;
        result.diverged = false;
        result.convergence_time_s = 30.0;
        return result;
    };

    auto start = std::chrono::high_resolution_clock::now();
    MonteCarloStatistics stats = runner.run(trial_func);
    auto end = std::chrono::high_resolution_clock::now();

    double runtime = std::chrono::duration<double>(end - start).count();

    EXPECT_EQ(stats.total_trials, config_.num_trials);
    std::cout << "Parallel execution time: " << runtime << " s\n";
}

TEST_F(MonteCarloTest, CSVOutput) {
    MonteCarloRunner runner(config_);

    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0;
        result.error_at_outage_start_m = 5.0;
        result.error_at_outage_end_m = 300.0;
        result.max_error_m = 350.0;
        result.converged = true;
        result.diverged = false;
        result.convergence_time_s = 30.0;
        result.rmse_gps_phase_m = 5.0;
        result.rmse_outage_phase_m = 200.0;
        result.rmse_recovery_phase_m = 50.0;
        result.max_velocity_error_mps = 1.0;
        result.final_velocity_error_mps = 0.5;
        result.num_iridium_measurements = 100;
        result.num_visible_satellites_avg = 3;
        return result;
    };

    MonteCarloStatistics stats = runner.run(trial_func);

    // Test file writing (files will be created in current directory)
    std::string summary_file = "/tmp/test_mc_summary.csv";
    std::string stats_file = "/tmp/test_mc_stats.csv";

    runner.saveSummaryCSV(summary_file);
    runner.saveStatisticsCSV(stats_file, stats);

    // Verify files exist
    std::ifstream summary(summary_file);
    EXPECT_TRUE(summary.is_open());
    summary.close();

    std::ifstream stats_f(stats_file);
    EXPECT_TRUE(stats_f.is_open());
    stats_f.close();

    // Clean up
    std::remove(summary_file.c_str());
    std::remove(stats_file.c_str());
}

TEST_F(MonteCarloTest, SeedDeterminism) {
    config_.base_seed = 42;

    MonteCarloRunner runner1(config_);
    MonteCarloRunner runner2(config_);

    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0 + 10.0 * dist(rng);
        result.converged = true;
        result.diverged = false;
        result.convergence_time_s = 30.0;
        return result;
    };

    MonteCarloStatistics stats1 = runner1.run(trial_func);
    MonteCarloStatistics stats2 = runner2.run(trial_func);

    // Same seed should give same results
    EXPECT_FLOAT_EQ(stats1.mean_final_error_m, stats2.mean_final_error_m);
}

TEST_F(MonteCarloTest, ResultsAccess) {
    MonteCarloRunner runner(config_);

    auto trial_func = [](int trial_id, uint64_t seed) -> TrialResult {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;
        result.final_error_m = 100.0 + trial_id;
        result.converged = true;
        result.diverged = false;
        return result;
    };

    runner.run(trial_func);

    const auto& results = runner.getResults();
    EXPECT_EQ(results.size(), static_cast<size_t>(config_.num_trials));

    // Check individual results
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i].trial_id, static_cast<int>(i));
        EXPECT_NEAR(results[i].final_error_m, 100.0 + i, 0.1);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
