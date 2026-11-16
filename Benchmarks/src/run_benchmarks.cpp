#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include "FilterMath.h"
#include "TestCheck.h"

#include "BenchmarkProblems.h"
#include "BenchmarkRunner.h"
#include "UKF.h"
#include "SRUKF.h"
#include "UnscentedFixedLagSmoother.h"
#include "SRUKFFixedLagSmoother.h"

using namespace Benchmark;

/**
 * Generate a ground truth trajectory by propagating the model with process
 * noise, and produce noisy measurements at each step. Uses accelerated
 * Cholesky from FilterMath for noise covariance factorization.
 */
template<typename Model>
// ============================================================================
//  Timestep alignment  (read before touching any filter loop below)
// ============================================================================
// This generator defines the convention the whole suite is scored against:
//
//     true_states[i]   = x_i,  the state at times[i]
//     measurements[i]  = h(x_i, times[i]) + v      -- y_i observes x_i AT t_i
//     true_states[i+1] = f(x_i, times[i]) + w      -- f is evaluated at the
//                                                     SOURCE time, not the target
//
// Two consequences the filter loops must respect:
//
//  1. `initialize()` places the estimate at times[0], so iteration 0 must NOT
//     predict -- y_0 already observes the state being held. The loops used to
//     call predict() unconditionally, which advanced the estimate to times[1],
//     fused y_0 (an observation of x_0) into it, and scored the result against
//     true_states[0]. Every stored estimate sat one step ahead of the truth it
//     was compared with, biasing every published RMSE.
//
//  2. `predict(t)` evaluates model.f(x, t): t names where the state currently
//     IS, not where it is going. Advancing from t_{i-1} to t_i therefore takes
//     times[i-1]. Passing times[i] propagated from the wrong end of the
//     interval, which matters for any time-varying model (BearingOnlyTracking's
//     observer position is a function of t).
//
// The same distinction is why SmootherType::step() takes both a `t_prev` and a
// `t_k` rather than one time for both roles.
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

    // Cholesky factors for noise generation using accelerated decomposition
    Eigen::MatrixXf L_Q = filtermath::cholesky(Q);
    if (L_Q.size() == 0) {
        Eigen::LLT<typename Model::StateMat> llt_Q(Q);
        if (llt_Q.info() == Eigen::Success) {
            L_Q = llt_Q.matrixL();
        } else {
            // Diagonal fallback: use sqrt of diagonal elements
            L_Q = Eigen::MatrixXf::Zero(Q.rows(), Q.cols());
            for (int i = 0; i < Q.rows(); ++i)
                L_Q(i, i) = std::sqrt(std::max(Q(i, i), 0.0f));
        }
    }
    Eigen::MatrixXf L_R = filtermath::cholesky(R);
    if (L_R.size() == 0) {
        Eigen::LLT<typename Model::ObsMat> llt_R(R);
        if (llt_R.info() == Eigen::Success) {
            L_R = llt_R.matrixL();
        } else {
            L_R = Eigen::MatrixXf::Zero(R.rows(), R.cols());
            for (int i = 0; i < R.rows(); ++i)
                L_R(i, i) = std::sqrt(std::max(R(i, i), 0.0f));
        }
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

// ============================================================================
//  Timing protocol
// ============================================================================
// Accuracy and timing are measured in SEPARATE passes over the same trajectory:
//
//   1. Accuracy pass: one run that records every filtered/smoothed state and
//      covariance. Not timed.
//   2. Timing pass: kTimingWarmup discarded runs, then timing_reps() measured
//      runs, each on a freshly constructed and initialised filter driven by the
//      SAME step function as the accuracy pass. Each run is timed with ONE clock
//      pair around the whole step loop; nothing else is inside the timed region.
//
// The previous loop read the clock twice per step (two vDSO calls inside the
// measured region, a fixed cost comparable to the 0.3-0.5 us steps of the
// smallest problems) and its total also included the per-step push_back of a
// heap-allocated MatrixXf covariance. It ran once, cold. Reported now:
//   Total_Time_ms     = median over reps of the step-loop wall time
//   Avg_Step_Time_ms  = Total_Time_ms / steps
//   Step_Time_Min_ms / Step_Time_Max_ms = per-step time of the fastest / slowest rep
// The final filter state of every timed run is returned so the caller can check
// it against the accuracy pass (same inputs, same code: it must agree).
//
// Repetitions default to 5; override with NLF_BENCH_REPS=<1..1000>.
constexpr int kTimingWarmup = 1;

inline int timing_reps() {
    if (const char* s = std::getenv("NLF_BENCH_REPS")) {
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        if (end != s && *end == '\0' && v >= 1 && v <= 1000) return static_cast<int>(v);
        std::cerr << "NLF_BENCH_REPS='" << s << "' ignored (need an integer in 1..1000)\n";
    }
    return 5;
}

template<typename State>
struct TimingResult {
    double median_total_ms = 0.0;
    double min_total_ms = 0.0;
    double max_total_ms = 0.0;
    int reps = 0;
    State final_state;   // final filtered state of the last timed run
};

template<typename State, typename MakeFilter, typename Step, typename Probe>
TimingResult<State> time_filter_loop(MakeFilter make_filter, Step step, Probe final_state,
                                     size_t n_steps) {
    using Clock = std::chrono::steady_clock;
    TimingResult<State> out;
    out.reps = timing_reps();
    std::vector<double> totals;
    totals.reserve(static_cast<size_t>(out.reps));

    for (int r = -kTimingWarmup; r < out.reps; ++r) {
        auto filter = make_filter();               // construction + initialize: not timed
        const auto t0 = Clock::now();
        for (size_t i = 0; i < n_steps; ++i) step(*filter, i);
        const auto t1 = Clock::now();
        // Reading the final state after the clock stops keeps the loop observable
        // (it depends on every step), so it cannot be elided.
        out.final_state = final_state(*filter);
        if (r >= 0) totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(totals.begin(), totals.end());
    const size_t m = totals.size() / 2;
    out.median_total_ms = (totals.size() % 2) ? totals[m] : 0.5 * (totals[m - 1] + totals[m]);
    out.min_total_ms = totals.front();
    out.max_total_ms = totals.back();
    return out;
}

template<typename State>
void apply_timing(const TimingResult<State>& t, size_t n_steps, const State& accuracy_final,
                  BenchmarkMetrics& metrics) {
    const double n = static_cast<double>(n_steps);
    metrics.total_time_ms = static_cast<float>(t.median_total_ms);
    metrics.avg_step_time_ms = static_cast<float>(t.median_total_ms / n);
    metrics.step_time_min_ms = static_cast<float>(t.min_total_ms / n);
    metrics.step_time_max_ms = static_cast<float>(t.max_total_ms / n);
    metrics.timing_reps = t.reps;
    metrics.timing_matches_accuracy = (t.final_state == accuracy_final);
}

/**
 * Run UKF on a trajectory
 */
template<typename Model>
BenchmarkMetrics run_ukf_benchmark(Model& model,
                                    TrajectoryData<typename Model::State, typename Model::Observation>& data,
                                    const typename Model::State& x0,
                                    const typename Model::StateMat& P0,
                                    const std::string& problem_name,
                                    float divergence_threshold = 10.0f) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "UKF";
    metrics.problem_name = problem_name;

    State u = State::Zero();

    // See "Timestep alignment" above generate_trajectory(): no propagation on
    // iteration 0 (the estimate is already at times[0]), and predict() takes
    // the time the state is currently AT, hence times[i-1].
    auto step = [&](UKFCore::UKF<NX, NY>& f, size_t i) {
        if (i > 0) f.predict(data.times[i - 1], u);
        f.update(data.times[i], data.measurements[i]);
    };
    auto make_filter = [&] {
        auto f = std::make_unique<UKFCore::UKF<NX, NY>>(model);
        f->initialize(x0, P0);
        return f;
    };

    // Accuracy pass (untimed).
    {
        auto filter = make_filter();
        for (size_t i = 0; i < data.times.size(); ++i) {
            step(*filter, i);
            data.filtered_states.push_back(filter->getState());
            data.filtered_covs.push_back(filter->getCovariance());
        }
    }

    // Timing pass (see "Timing protocol" above).
    const auto timing = time_filter_loop<State>(
        make_filter, step, [](const UKFCore::UKF<NX, NY>& f) { return State(f.getState()); },
        data.times.size());
    apply_timing(timing, data.times.size(), data.filtered_states.back(), metrics);

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    compute_nees(data.true_states, data.filtered_states, data.filtered_covs, metrics);

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, divergence_threshold);

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
                                      const std::string& problem_name,
                                      float divergence_threshold = 10.0f) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "SRUKF";
    metrics.problem_name = problem_name;

    State u = State::Zero();

    // See "Timestep alignment" above generate_trajectory(): no propagation on
    // iteration 0 (the estimate is already at times[0]), and predict() takes
    // the time the state is currently AT, hence times[i-1].
    auto step = [&](UKFCore::SRUKF<NX, NY>& f, size_t i) {
        if (i > 0) f.predict(data.times[i - 1], u);
        f.update(data.times[i], data.measurements[i]);
    };
    auto make_filter = [&] {
        auto f = std::make_unique<UKFCore::SRUKF<NX, NY>>(model);
        f->initialize(x0, P0);
        return f;
    };

    // Accuracy pass (untimed).
    {
        auto filter = make_filter();
        for (size_t i = 0; i < data.times.size(); ++i) {
            step(*filter, i);
            data.filtered_states.push_back(filter->getState());
            data.filtered_covs.push_back(filter->getCovariance());
        }
    }

    // Timing pass (see "Timing protocol" above).
    const auto timing = time_filter_loop<State>(
        make_filter, step, [](const UKFCore::SRUKF<NX, NY>& f) { return State(f.getState()); },
        data.times.size());
    apply_timing(timing, data.times.size(), data.filtered_states.back(), metrics);

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    compute_nees(data.true_states, data.filtered_states, data.filtered_covs, metrics);

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, divergence_threshold);

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
                                             int lag = 20,
                                             float divergence_threshold = 10.0f) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "UKF+Smoother";
    metrics.problem_name = problem_name;

    data.smoother_lag = lag;
    State u = State::Zero();

    // See "Timestep alignment" above generate_trajectory(). Iteration 0 fuses
    // y_0 where the estimate already is; from then on we propagate from
    // times[i-1] to times[i] and fuse the measurement taken at times[i].
    // The fixed-lag backward pass runs inside step(), so timing step() times it.
    auto step = [&](UKFCore::UnscentedFixedLagSmoother<NX, NY>& s, size_t i) {
        if (i == 0) s.observe_initial(data.times[0], data.measurements[0]);
        else        s.step(data.times[i - 1], data.times[i], data.measurements[i], u);
    };
    auto make_filter = [&] {
        auto s = std::make_unique<UKFCore::UnscentedFixedLagSmoother<NX, NY>>(model, lag);
        s->initialize(x0, P0);
        return s;
    };

    // Accuracy pass (untimed).
    {
        auto smoother = make_filter();
        for (size_t i = 0; i < data.times.size(); ++i) {
            step(*smoother, i);
            data.filtered_states.push_back(smoother->get_filtered_state());
            data.filtered_covs.push_back(smoother->get_filtered_covariance());
            // Smoothed estimate for time i-lag.
            data.smoothed_states.push_back(smoother->get_smoothed_state(lag));
            data.smoothed_covs.push_back(smoother->get_smoothed_covariance(lag));
        }
    }

    // Timing pass (see "Timing protocol" above).
    const auto timing = time_filter_loop<State>(
        make_filter, step,
        [](const UKFCore::UnscentedFixedLagSmoother<NX, NY>& s) { return State(s.get_filtered_state()); },
        data.times.size());
    apply_timing(timing, data.times.size(), data.filtered_states.back(), metrics);

    // Compute filtered metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    // Compute smoothed metrics with correct time alignment
    // smoothed_states[i] = smoother estimate for time i-lag
    if (data.smoothed_states.size() > static_cast<size_t>(lag)) {
        std::vector<State> aligned_true(data.true_states.begin(),
                                         data.true_states.begin() + static_cast<long>(data.true_states.size()) - lag);
        std::vector<State> aligned_smooth(data.smoothed_states.begin() + lag,
                                           data.smoothed_states.end());
        metrics.rmse_smoothed_overall = compute_rmse(aligned_true, aligned_smooth);
    }

    compute_nees(data.true_states, data.filtered_states, data.filtered_covs, metrics);

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, divergence_threshold);

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
                                                int lag = 20,
                                                float divergence_threshold = 10.0f) {
    using State = typename Model::State;
    constexpr int NX = Model::STATE_DIM;
    constexpr int NY = Model::OBS_DIM;

    BenchmarkMetrics metrics;
    metrics.filter_name = "SRUKF+Smoother";
    metrics.problem_name = problem_name;

    data.smoother_lag = lag;
    State u = State::Zero();

    // See "Timestep alignment" above generate_trajectory(). Iteration 0 fuses
    // y_0 where the estimate already is; from then on we propagate from
    // times[i-1] to times[i] and fuse the measurement taken at times[i].
    // The fixed-lag backward pass runs inside step(), so timing step() times it.
    auto step = [&](UKFCore::SRUKFFixedLagSmoother<NX, NY>& s, size_t i) {
        if (i == 0) s.observe_initial(data.times[0], data.measurements[0]);
        else        s.step(data.times[i - 1], data.times[i], data.measurements[i], u);
    };
    auto make_filter = [&] {
        auto s = std::make_unique<UKFCore::SRUKFFixedLagSmoother<NX, NY>>(model, lag);
        s->initialize(x0, P0);
        return s;
    };

    // Accuracy pass (untimed).
    {
        auto smoother = make_filter();
        for (size_t i = 0; i < data.times.size(); ++i) {
            step(*smoother, i);
            data.filtered_states.push_back(smoother->get_filtered_state());
            data.filtered_covs.push_back(smoother->get_filtered_covariance());
            // Smoothed estimate for time i-lag.
            data.smoothed_states.push_back(smoother->get_smoothed_state(lag));
            data.smoothed_covs.push_back(smoother->get_smoothed_covariance(lag));
        }
    }

    // Timing pass (see "Timing protocol" above).
    const auto timing = time_filter_loop<State>(
        make_filter, step,
        [](const UKFCore::SRUKFFixedLagSmoother<NX, NY>& s) { return State(s.get_filtered_state()); },
        data.times.size());
    apply_timing(timing, data.times.size(), data.filtered_states.back(), metrics);

    // Compute metrics
    metrics.rmse_overall = compute_rmse(data.true_states, data.filtered_states);

    // Compute smoothed metrics with correct time alignment
    if (data.smoothed_states.size() > static_cast<size_t>(lag)) {
        std::vector<State> aligned_true(data.true_states.begin(),
                                         data.true_states.begin() + static_cast<long>(data.true_states.size()) - lag);
        std::vector<State> aligned_smooth(data.smoothed_states.begin() + lag,
                                           data.smoothed_states.end());
        metrics.rmse_smoothed_overall = compute_rmse(aligned_true, aligned_smooth);
    }

    compute_nees(data.true_states, data.filtered_states, data.filtered_covs, metrics);

    metrics.convergence_time = compute_convergence_time(data.times, data.true_states,
                                                         data.filtered_states, 1.0f);
    metrics.num_divergences = count_divergences(data.true_states, data.filtered_states, divergence_threshold);

    return metrics;
}

/**
 * Main benchmark driver: runs UKF, SRUKF, and their smoother variants on
 * four challenging problems (Coupled Oscillators, Van der Pol, Bearing-Only,
 * Reentry Vehicle). Collects RMSE, NEES, timing, and divergence metrics
 * and exports to benchmark_results.csv plus per-problem trajectory CSVs.
 */
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

        // Divergence threshold scaled to this problem. The target starts at
        // (200,0) and travels ~300 m over 30 s, giving a steady-state tracking
        // RMSE ~= 64 m. The default 10.0 threshold is far below that scale, so
        // it flagged ~65% of steps as "divergences" even though NEES shows the
        // filter is perfectly consistent. 500 m marks genuine loss-of-track
        // (well above steady-state error), analogous to reentry's 5 km gate.
        float bearing_div_thresh = 500.0f;

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "BearingOnly4D", bearing_div_thresh);
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
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "BearingOnly4D", bearing_div_thresh);
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
                                                         "BearingOnly4D", 30, bearing_div_thresh);
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
        // Initial state: 80km above Earth surface, realistic reentry trajectory
        // Position offset from radar to provide good observability
        float R0 = model.R0;
        float alt0 = 80000.0f;  // 80km initial altitude
        // Start vehicle offset in y-z plane from radar for better angle diversity
        Eigen::Matrix<float, 6, 1> x0;
        x0 << R0 + alt0 * 0.7f,           // x position
              alt0 * 0.5f,                 // y position (offset for observability)
              alt0 * 0.5f,                 // z position
              -500.0f,                     // vx (approaching radar)
              2000.0f,                     // vy (tangential)
              -300.0f;                     // vz (descending)

        Eigen::Matrix<float, 6, 6> P0 = Eigen::Matrix<float, 6, 6>::Identity();
        P0.block<3,3>(0,0) *= 100000.0f;  // Position uncertainty (m^2) - 316m std dev
        P0.block<3,3>(3,3) *= 10000.0f;   // Velocity uncertainty ((m/s)^2) - 100m/s std dev

        auto data = generate_trajectory(model, x0, 30.0f, 0.1f, 45);

        // Divergence threshold for reentry: 5km position or velocity error
        float reentry_div_thresh = 5000.0f;

        // UKF
        {
            auto data_copy = data;
            data_copy.filtered_states.clear();
            data_copy.filtered_covs.clear();
            auto metrics = run_ukf_benchmark(model, data_copy, x0, P0, "ReentryVehicle6D", reentry_div_thresh);
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
            auto metrics = run_srukf_benchmark(model, data_copy, x0, P0, "ReentryVehicle6D", reentry_div_thresh);
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
                                                         "ReentryVehicle6D", 20, reentry_div_thresh);
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

    // ------------------------------------------------------------------
    //  Regression gate (this binary is registered as the Benchmarks CTest case)
    // ------------------------------------------------------------------
    // Gates on ACCURACY and CONSISTENCY only, never on step time. RMSE/NEES are
    // backend- and host-independent, so they mean the same thing on every machine;
    // wall-clock does not (even with the warmup + median-of-reps protocol above,
    // absolute step time is a property of the host), so a timing gate would just
    // flake on any loaded or slower box.
    //
    // RMSE is gated against a per-(filter, problem) baseline with a two-sided
    // relative band, NOT a loose ceiling. Every RMSE/NEES figure here is
    // deterministic -- the generators are seeded, and re-running this suite
    // reproduces them bit-identically (only the wall-clock columns move). The
    // tolerance therefore absorbs cross-host floating-point reassociation, which is
    // orders of magnitude smaller, and nothing else.
    //
    // The band is two-sided deliberately. A one-sided ceiling only notices a filter
    // getting worse. The last real defect here -- the one-timestep misalignment
    // fixed in 3dbbf4c -- moved reentry RMSE by 0.589%, and the ~25% ceilings that
    // replaced this table were baselined on the *buggy* values and reported PASS for
    // its entire lifetime. Drift in either direction is a regression signal; an
    // intentional improvement is expected to update this table in the same commit.
    static constexpr float kRmseTol = 0.005f;  // 0.5%: tight enough for the 0.589% class
    struct RmseBaseline {
        const char* filter;
        const char* problem;
        float rmse;           // filtered RMSE
        float rmse_smoothed;  // smoothed RMSE; 0 for rows that do not smooth
    };
    const RmseBaseline baselines[] = {
        {"UKF",            "CoupledOscillators10D",   1.45655f,   0.0f},
        {"SRUKF",          "CoupledOscillators10D",   1.45655f,   0.0f},
        {"UKF+Smoother",   "CoupledOscillators10D",   1.45655f,   1.14768f},
        {"SRUKF+Smoother", "CoupledOscillators10D",   1.45655f,   1.14768f},
        {"UKF",            "VanDerPol2D",             0.468783f,  0.0f},  // post Wc0-floor fix (was 0.468888)
        {"SRUKF",          "VanDerPol2D",             0.467096f,  0.0f},
        {"SRUKF+Smoother", "VanDerPol2D",             0.467096f,  0.430381f},
        {"UKF",            "BearingOnly4D",          63.4926f,    0.0f},  // post Wc0-floor fix (was 63.4902)
        {"SRUKF",          "BearingOnly4D",          63.8392f,    0.0f},
        {"SRUKF+Smoother", "BearingOnly4D",          63.8392f,   51.6818f},
        {"UKF",            "ReentryVehicle6D",      367.115f,     0.0f},  // post Wc0-floor fix (was 366.868)
        {"SRUKF",          "ReentryVehicle6D",      367.115f,     0.0f},
        {"SRUKF+Smoother", "ReentryVehicle6D",      367.115f,   236.251f},
    };

    std::cout << "\n--- Regression gate ---" << std::endl;
    NLF_CHECK(all_metrics.size() == 13,
              "benchmark suite produced all 13 filter/problem rows");

    for (const auto& m : all_metrics) {
        const std::string tag = m.filter_name + " on " + m.problem_name;

        NLF_CHECK(std::isfinite(m.rmse_overall) && m.rmse_overall > 0.0f,
                  ("filtered RMSE is finite and positive: " + tag).c_str());
        NLF_CHECK(m.num_divergences == 0,
                  ("filter did not diverge: " + tag).c_str());
        // The timed runs replay the accuracy run (same inputs, same step function).
        // If their final state differs, the timings describe a different computation.
        NLF_CHECK(m.timing_matches_accuracy,
                  ("timed runs reproduce the accuracy run's final state: " + tag).c_str());

        // A smoother that does not improve on the filter it smooths is broken.
        if (m.rmse_smoothed_overall > 0.0f) {
            NLF_CHECK(std::isfinite(m.rmse_smoothed_overall),
                      ("smoothed RMSE is finite: " + tag).c_str());
            NLF_CHECK(m.rmse_smoothed_overall < m.rmse_overall,
                      ("smoothing reduces RMSE: " + tag).c_str());
        }

        // NEES consistency: a filter whose covariance no longer describes its own
        // error lands far outside the 95% band. The floor is deliberately slack
        // (measured 94.5-99.3% with exact chi-square bounds) so only a gross inconsistency trips it.
        NLF_CHECK(std::isfinite(m.median_nees) && m.median_nees > 0.0f,
                  ("median NEES is finite and positive: " + tag).c_str());
        NLF_CHECK(m.nees_valid > 0, ("NEES had well-conditioned samples: " + tag).c_str());
        NLF_CHECK(m.pct_in_bounds >= 80.0f,
                  ("NEES stays inside the 95% chi-square band: " + tag).c_str());

        const RmseBaseline* base = nullptr;
        for (const auto& b : baselines) {
            if (m.problem_name == b.problem && m.filter_name == b.filter) {
                base = &b;
                break;
            }
        }
        // Without this, a row whose name matches no entry would skip the RMSE gate
        // silently -- the failure mode being that adding a filter appears to pass.
        NLF_CHECK(base != nullptr, ("row is covered by an RMSE baseline: " + tag).c_str());

        NLF_CHECK(std::abs(m.rmse_overall - base->rmse) <= kRmseTol * base->rmse,
                  ("filtered RMSE matches baseline within 0.5%: " + tag).c_str());

        if (base->rmse_smoothed > 0.0f) {
            NLF_CHECK(std::abs(m.rmse_smoothed_overall - base->rmse_smoothed)
                          <= kRmseTol * base->rmse_smoothed,
                      ("smoothed RMSE matches baseline within 0.5%: " + tag).c_str());
        }
    }
    std::cout << "PASS: " << all_metrics.size()
              << " rows matched RMSE baselines within " << (kRmseTol * 100.0f)
              << "%, NEES-consistent, no divergences" << std::endl;

    return 0;
}
