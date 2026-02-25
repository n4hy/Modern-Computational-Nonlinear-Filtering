#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <optmath/neon_kernels.hpp>

#include "BenchmarkProblems.h"
#include "BenchmarkRunner.h"
#include "UKF.h"
#include "SRUKF.h"
#include "UnscentedFixedLagSmoother.h"
#include "SRUKFFixedLagSmoother.h"

using namespace Benchmark;

/**
 * Generate ground truth trajectory and measurements
 */
template<typename Model>
auto generate_trajectory(Model& model,
                        const typename Model::State& x0,
                        float T_final,
                        float dt,
                        unsigned int seed = 42) {
    using State = typename Model::State;
    using Observation = typename Model::Observation;

    std::mt19937 gen(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    TrajectoryData<State, Observation> data;

    State x = x0;
    float t = 0.0f;
    State u = State::Zero();

    // Get noise covariances
    auto Q = model.Q(0.0f);
    auto R = model.R(0.0f);

    // Cholesky factors for noise generation using NEON-accelerated decomposition
    Eigen::MatrixXf L_Q = optmath::neon::neon_cholesky(Q);
    if (L_Q.size() == 0) {
        Eigen::LLT<typename Model::StateMat> llt_Q(Q);
        L_Q = llt_Q.matrixL();
    }
    Eigen::MatrixXf L_R = optmath::neon::neon_cholesky(R);
    if (L_R.size() == 0) {
        Eigen::LLT<typename Model::ObsMat> llt_R(R);
        L_R = llt_R.matrixL();
    }

    constexpr int STATE_DIM = Model::STATE_DIM;
    constexpr int OBS_DIM = Model::OBS_DIM;

    while (t <= T_final) {
        data.times.push_back(t);
        data.true_states.push_back(x);

        // Generate measurement
        Observation y_true = model.h(x, t);
        Observation noise_R;
        for (int i = 0; i < OBS_DIM; ++i) {
            noise_R(i) = normal(gen);
        }
        Observation y_meas = y_true + L_R * noise_R;
        data.measurements.push_back(y_meas);

        // Propagate true state
        State x_next = model.f(x, t, u);

        // Add process noise
        State noise_Q;
        for (int i = 0; i < STATE_DIM; ++i) {
            noise_Q(i) = normal(gen);
        }
        x = x_next + L_Q * noise_Q;

        t += dt;
    }

    return data;
}

/**
 * Run UKF on a trajectory
 */
template<typename Model>
BenchmarkMetrics run_ukf_benchmark(Model& model,
                                    TrajectoryData<typename Model::State, typename Model::Observation>& data,
                                    const typename Model::State& x0,
                                    const typename Model::StateMat& P0,
                                    const std::string& problem_name) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "UKF";
    metrics.problem_name = problem_name;

    UKFCore::UKF<NX, NY> filter(model);
    filter.initialize(x0, P0);

    State u = State::Zero();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.times.size(); ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        filter.predict(data.times[i], u);
        filter.update(data.times[i], data.measurements[i]);

        auto step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> step_duration = step_end - step_start;
        metrics.avg_step_time_ms += step_duration.count();

        data.filtered_states.push_back(filter.getState());
        data.filtered_covs.push_back(filter.getCovariance());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    metrics.total_time_ms = duration.count();
    metrics.avg_step_time_ms /= data.times.size();

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    auto nees = compute_nees(data.true_states, data.filtered_states, data.filtered_covs);
    metrics.mean_nees = nees.first;
    metrics.std_nees = nees.second;

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, 10.0f);

    return metrics;
}

/**
 * Run SRUKF on a trajectory
 */
template<typename Model>
BenchmarkMetrics run_srukf_benchmark(Model& model,
                                      TrajectoryData<typename Model::State, typename Model::Observation>& data,
                                      const typename Model::State& x0,
                                      const typename Model::StateMat& P0,
                                      const std::string& problem_name) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "SRUKF";
    metrics.problem_name = problem_name;

    UKFCore::SRUKF<NX, NY> filter(model);
    filter.initialize(x0, P0);

    State u = State::Zero();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.times.size(); ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        filter.predict(data.times[i], u);
        filter.update(data.times[i], data.measurements[i]);

        auto step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> step_duration = step_end - step_start;
        metrics.avg_step_time_ms += step_duration.count();

        data.filtered_states.push_back(filter.getState());
        data.filtered_covs.push_back(filter.getCovariance());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    metrics.total_time_ms = duration.count();
    metrics.avg_step_time_ms /= data.times.size();

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    auto nees = compute_nees(data.true_states, data.filtered_states, data.filtered_covs);
    metrics.mean_nees = nees.first;
    metrics.std_nees = nees.second;

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, 10.0f);

    return metrics;
}

/**
 * Run UKF with smoother
 */
template<typename Model>
BenchmarkMetrics run_ukf_smoother_benchmark(Model& model,
                                             TrajectoryData<typename Model::State, typename Model::Observation>& data,
                                             const typename Model::State& x0,
                                             const typename Model::StateMat& P0,
                                             const std::string& problem_name,
                                             int lag = 20) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "UKF+Smoother";
    metrics.problem_name = problem_name;

    UKFCore::UnscentedFixedLagSmoother<NX, NY> smoother(model, lag);
    smoother.initialize(x0, P0);

    State u = State::Zero();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.times.size(); ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        smoother.step(data.times[i], data.measurements[i], u);

        auto step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> step_duration = step_end - step_start;
        metrics.avg_step_time_ms += step_duration.count();

        data.filtered_states.push_back(smoother.get_filtered_state());
        data.filtered_covs.push_back(smoother.get_filtered_covariance());

        // Get smoothed estimate (lag steps back)
        data.smoothed_states.push_back(smoother.get_smoothed_state(lag));
        data.smoothed_covs.push_back(smoother.get_smoothed_covariance(lag));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    metrics.total_time_ms = duration.count();
    metrics.avg_step_time_ms /= data.times.size();

    // Compute filtered metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    // Compute smoothed metrics
    metrics.rmse_smoothed_overall = compute_rmse(data.true_states, data.smoothed_states);

    auto nees = compute_nees(data.true_states, data.filtered_states, data.filtered_covs);
    metrics.mean_nees = nees.first;
    metrics.std_nees = nees.second;

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, 10.0f);

    return metrics;
}

/**
 * Run SRUKF with smoother
 */
template<typename Model>
BenchmarkMetrics run_srukf_smoother_benchmark(Model& model,
                                                TrajectoryData<typename Model::State, typename Model::Observation>& data,
                                                const typename Model::State& x0,
                                                const typename Model::StateMat& P0,
                                                const std::string& problem_name,
                                                int lag = 20) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "SRUKF+Smoother";
    metrics.problem_name = problem_name;

    UKFCore::SRUKFFixedLagSmoother<NX, NY> smoother(model, lag);
    smoother.initialize(x0, P0);

    State u = State::Zero();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.times.size(); ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        smoother.step(data.times[i], data.measurements[i], u);

        auto step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> step_duration = step_end - step_start;
        metrics.avg_step_time_ms += step_duration.count();

        data.filtered_states.push_back(smoother.get_filtered_state());
        data.filtered_covs.push_back(smoother.get_filtered_covariance());

        // Get smoothed estimate
        data.smoothed_states.push_back(smoother.get_smoothed_state(lag));
        data.smoothed_covs.push_back(smoother.get_smoothed_covariance(lag));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    metrics.total_time_ms = duration.count();
    metrics.avg_step_time_ms /= data.times.size();

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);
    metrics.rmse_smoothed_overall = compute_rmse(data.true_states, data.smoothed_states);

    auto nees = compute_nees(data.true_states, data.filtered_states, data.filtered_covs);
    metrics.mean_nees = nees.first;
    metrics.std_nees = nees.second;

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, 10.0f);

    return metrics;
}

int main() {
    std::cout << "=== Modern Computational Nonlinear Filtering Benchmarks ===" << std::endl;
    std::cout << "Testing UKF, SRUKF, and smoothers on challenging problems" << std::endl;

    std::vector<BenchmarkMetrics> all_metrics;

    // Open summary CSV
    std::ofstream summary_file("benchmark_results.csv");
    BenchmarkMetrics::write_csv_header(summary_file);

    // ========== Problem 1: Coupled Oscillators (10D) ==========
    {
        std::cout << "\n\n========== Coupled Oscillators (10D) ==========" << std::endl;

        CoupledOscillators<10, 5> model;
        Eigen::Matrix<float, 10, 1> x0;
        x0 << 1.0f, 0.0f, -0.5f, 0.0f, 0.8f, 0.0f, -0.3f, 0.0f, 0.6f, 0.0f;

        Eigen::Matrix<float, 10, 10> P0 = Eigen::Matrix<float, 10, 10>::Identity();

        auto data = generate_trajectory(model, x0, 50.0f, 0.01f, 42);

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "CoupledOscillators10D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("coupled_osc_ukf.csv", data_copy);
        }

        // SRUKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "CoupledOscillators10D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("coupled_osc_srukf.csv", data_copy);
        }

        // UKF + Smoother
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            data_copy.smoothed_states.clear();
            data_copy.smoothed_covs.clear();
            auto metrics = run_ukf_smoother_benchmark(model, data_copy, x0, P0,
                                                       "CoupledOscillators10D", 20);
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("coupled_osc_ukf_smooth.csv", data_copy);
        }

        // SRUKF + Smoother
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            data_copy.smoothed_states.clear();
            data_copy.smoothed_covs.clear();
            auto metrics = run_srukf_smoother_benchmark(model, data_copy, x0, P0,
                                                         "CoupledOscillators10D", 20);
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("coupled_osc_srukf_smooth.csv", data_copy);
        }
    }

    // ========== Problem 2: Van der Pol with Discontinuous Forcing ==========
    {
        std::cout << "\n\n========== Van der Pol (2D, Discontinuous) ==========" << std::endl;

        VanDerPolDiscontinuous<2, 1> model;
        Eigen::Matrix<float, 2, 1> x0;
        x0 << 1.0f, 0.0f;

        Eigen::Matrix<float, 2, 2> P0 = Eigen::Matrix<float, 2, 2>::Identity();

        auto data = generate_trajectory(model, x0, 20.0f, 0.01f, 43);

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "VanDerPol2D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("vanderpol_ukf.csv", data_copy);
        }

        // SRUKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "VanDerPol2D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("vanderpol_srukf.csv", data_copy);
        }

        // SRUKF + Smoother
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            data_copy.smoothed_states.clear();
            data_copy.smoothed_covs.clear();
            auto metrics = run_srukf_smoother_benchmark(model, data_copy, x0, P0,
                                                         "VanDerPol2D", 20);
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("vanderpol_srukf_smooth.csv", data_copy);
        }
    }

    // ========== Problem 3: Bearing-Only Tracking ==========
    {
        std::cout << "\n\n========== Bearing-Only Tracking (4D) ==========" << std::endl;

        BearingOnlyTracking<4, 1> model;
        Eigen::Matrix<float, 4, 1> x0;
        x0 << 200.0f, 0.0f, 10.0f, 5.0f;  // Initial position and velocity

        Eigen::Matrix<float, 4, 4> P0 = Eigen::Matrix<float, 4, 4>::Identity();
        P0 *= 10.0f;  // Large initial uncertainty

        auto data = generate_trajectory(model, x0, 30.0f, 0.1f, 44);

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "BearingOnly4D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("bearing_ukf.csv", data_copy);
        }

        // SRUKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "BearingOnly4D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("bearing_srukf.csv", data_copy);
        }

        // SRUKF + Smoother
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            data_copy.smoothed_states.clear();
            data_copy.smoothed_covs.clear();
            auto metrics = run_srukf_smoother_benchmark(model, data_copy, x0, P0,
                                                         "BearingOnly4D", 30);
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("bearing_srukf_smooth.csv", data_copy);
        }
    }

    // ========== Problem 4: Reentry Vehicle (6D) ==========
    {
        std::cout << "\n\n========== Reentry Vehicle (6D) ==========" << std::endl;

        ReentryVehicle<6, 3> model;
        // Initial state: 100km above Earth surface, suborbital reentry trajectory
        // Position in Earth-centered coords: altitude = ||pos|| - R0
        float R0 = model.R0;
        Eigen::Matrix<float, 6, 1> x0;
        x0 << R0 + 100000.0f, 0.0f, 0.0f, 0.0f, 3000.0f, -200.0f;

        Eigen::Matrix<float, 6, 6> P0 = Eigen::Matrix<float, 6, 6>::Identity();
        P0.block<3,3>(0,0) *= 10000.0f;   // Position uncertainty (m^2)
        P0.block<3,3>(3,3) *= 100.0f;     // Velocity uncertainty ((m/s)^2)

        auto data = generate_trajectory(model, x0, 20.0f, 0.1f, 45);

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "ReentryVehicle6D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("reentry_ukf.csv", data_copy);
        }

        // SRUKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "ReentryVehicle6D");
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("reentry_srukf.csv", data_copy);
        }

        // SRUKF + Smoother
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            data_copy.smoothed_states.clear();
            data_copy.smoothed_covs.clear();
            auto metrics = run_srukf_smoother_benchmark(model, data_copy, x0, P0,
                                                         "ReentryVehicle6D", 20);
            metrics.print();
            metrics.save_to_csv(summary_file);
            all_metrics.push_back(metrics);
            save_trajectory_csv("reentry_srukf_smooth.csv", data_copy);
        }
    }

    summary_file.close();

    std::cout << "\n\n========== Benchmark Complete ==========" << std::endl;
    std::cout << "Results saved to benchmark_results.csv" << std::endl;
    std::cout << "Trajectory files saved with prefix: coupled_osc_, vanderpol_, bearing_, reentry_" << std::endl;

    return 0;
}
