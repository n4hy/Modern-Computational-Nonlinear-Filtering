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
    //
    // Per-subvector (position/velocity) RMSE is deliberately NOT reported. Fields
    // for it existed here and were emitted as RMSE_Position / RMSE_Velocity, but
    // nothing ever assigned them, so every published value was a hard 0.0 rather
    // than a measurement. compute_rmse_indices() below implements the calculation
    // and is ready to use; wiring it up needs a per-problem index map (which state
    // components are "position" is model-specific and not always well-defined --
    // ReentryVehicle's 6D state includes a ballistic coefficient that is neither),
    // so that is left as a deliberate choice for whoever adds it.
    float rmse_overall = 0.0f;

    // Smoothed metrics (if applicable)
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
    //
    // convergence_time is NaN when the filter never converged. It previously
    // returned the final timestamp in that case, which is indistinguishable from
    // having genuinely converged on the last step -- 6 of the suite's 13 rows
    // (all Bearing-Only and all Reentry) reported that sentinel as if it were a
    // measurement. NaN propagates honestly: it prints as "did not converge",
    // serialises to an empty CSV cell, and leaves a gap in the plots instead of
    // a fabricated data point.
    float convergence_time = std::numeric_limits<float>::quiet_NaN();
    int num_divergences = 0;         // Number of times error exceeded threshold

    bool converged() const { return !std::isnan(convergence_time); }

    /** Print a formatted summary of all metrics for this filter/problem pair. */
    void print() const {
        std::cout << std::setprecision(6);
        std::cout << "\n=== " << filter_name << " on " << problem_name << " ===" << std::endl;
        std::cout << "Filtered RMSE: " << rmse_overall << std::endl;

        if (rmse_smoothed_overall > 0) {
            std::cout << "Smoothed RMSE: " << rmse_smoothed_overall << std::endl;
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
        std::cout << "Convergence Time: ";
        if (converged()) std::cout << convergence_time << " s" << std::endl;
        else             std::cout << "did not converge" << std::endl;
        std::cout << "Divergences: " << num_divergences << std::endl;
    }

    /** Append one row of metrics to an open CSV file. */
    void save_to_csv(std::ofstream& file) const {
        file << filter_name << ","
             << problem_name << ","
             << rmse_overall << ","
             << rmse_smoothed_overall << ","
             << median_nees << ","
             << pct_in_bounds << ","
             << nees_valid << ","
             << nees_total << ","
             << avg_step_time_ms << ","
             << total_time_ms << ",";
        // Empty cell rather than a number when the filter never converged: pandas
        // and every spreadsheet read it as missing, which is what it is.
        if (converged()) file << convergence_time;
        file << "," << num_divergences << std::endl;
    }

    /** Write the CSV column header row for benchmark results. */
    static void write_csv_header(std::ofstream& file) {
        file << "Filter,Problem,RMSE_Overall,"
             << "RMSE_Smoothed_Overall,"
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
    // Fixed-lag offset of the smoother: smoothed_states[i] is the smoothed
    // estimate for time index (i - smoother_lag). 0 for filter-only runs.
    int smoother_lag = 0;
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
 * Regularized lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a), a > 0, x >= 0.
 *
 * Series for x < a + 1, Lentz continued fraction for Q = 1 - P otherwise; both
 * are the standard convergent representations (Abramowitz & Stegun 6.5.29 and
 * 6.5.31). Evaluated in double: the caller's float interface does not need it,
 * but the Newton iteration in chi2_quantile() does.
 */
inline double regularized_gamma_p(double a, double x) {
    if (x <= 0.0) return 0.0;
    const double log_prefactor = a * std::log(x) - x - std::lgamma(a);
    const double eps = 1e-15;
    if (x < a + 1.0) {
        // P(a,x) = e^{-x} x^a / Gamma(a) * sum_{k>=0} x^k / (a (a+1) ... (a+k))
        double term = 1.0 / a;
        double sum = term;
        for (int k = 1; k < 1000; ++k) {
            term *= x / (a + k);
            sum += term;
            if (std::abs(term) < std::abs(sum) * eps) break;
        }
        return sum * std::exp(log_prefactor);
    }
    // Q(a,x) = e^{-x} x^a / Gamma(a) * 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a- ...)))
    // evaluated by the modified Lentz algorithm.
    const double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < eps) break;
    }
    return 1.0 - std::exp(log_prefactor) * h;
}

/**
 * Exact chi-squared quantile: the x with P(chi2_n <= x) = p, for 0 < p < 1 and
 * n >= 1. Since chi2_n is Gamma(n/2, 2), x = 2 * P^{-1}(n/2, p) where P is the
 * regularized lower incomplete gamma.
 *
 * Solved by bisection on P(n/2, y) to a tight bracket, then Newton steps on the
 * same function (dP/dy = y^{a-1} e^{-y} / Gamma(a)) to round-off. Robust for
 * every n, including n = 1 where P is steep at the origin.
 *
 * This replaced a Wilson-Hilferty approximation whose lower 2.5% quantile was
 * 48% low at n = 2 (0.026 vs 0.0506), which made Pct_In_Bounds read optimistic.
 * Agrees with scipy.stats.chi2.ppf to < 3e-7 relative (float rounding of the
 * result) for n in {1, 2, 3, 4, 6, 10, 40, 100}; tests/test_chi2_quantile.cpp
 * pins that at 1e-5.
 */
inline float chi2_quantile(float p, int n) {
    if (!(p > 0.0f && p < 1.0f) || n < 1) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const double a = 0.5 * n;
    const double target = static_cast<double>(p);

    // Bracket: P(a, 0) = 0 < target; grow the upper end until P exceeds target.
    double lo = 0.0;
    double hi = std::max(1.0, 2.0 * a);
    while (regularized_gamma_p(a, hi) < target) {
        lo = hi;
        hi *= 2.0;
    }
    for (int i = 0; i < 200 && (hi - lo) > 1e-12 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (regularized_gamma_p(a, mid) < target) lo = mid; else hi = mid;
    }
    double y = 0.5 * (lo + hi);

    // Newton polish; a step that leaves the bracket is discarded.
    for (int i = 0; i < 8; ++i) {
        const double f = regularized_gamma_p(a, y) - target;
        const double df = std::exp((a - 1.0) * std::log(y) - y - std::lgamma(a));
        if (!(df > 0.0) || !std::isfinite(df)) break;
        const double y_new = y - f / df;
        if (!(y_new > lo && y_new < hi)) break;
        if (std::abs(y_new - y) < 1e-15 * y) { y = y_new; break; }
        y = y_new;
    }
    return static_cast<float>(2.0 * y);
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
    // Validate before touching true_states[0]: reading it first was UB on an
    // empty input.
    if (true_states.size() != estimated_states.size() ||
        true_states.size() != covs.size() ||
        true_states.size() < 20) {
        metrics.median_nees = -1.0f;
        metrics.pct_in_bounds = 0.0f;
        return;
    }

    const int n = static_cast<int>(true_states[0].size());  // state dimension

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

    // Median via nth_element (O(n) instead of O(n log n) full sort)
    size_t mid = nees_values.size() / 2;
    std::nth_element(nees_values.begin(), nees_values.begin() + mid, nees_values.end());
    if (nees_values.size() % 2 == 0) {
        float median_hi = nees_values[mid];
        // Find max in lower partition for the other median element
        float median_lo = *std::max_element(nees_values.begin(), nees_values.begin() + mid);
        metrics.median_nees = 0.5f * (median_lo + median_hi);
    } else {
        metrics.median_nees = nees_values[mid];
    }

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
 * Detect convergence time - when error falls below threshold and stays there.
 *
 * Returns NaN if the filter never converges, so a non-convergence cannot be
 * mistaken for a measurement.
 *
 * `threshold` is an ABSOLUTE error norm, not a relative one, so it only means
 * anything against a problem whose error scale is comparable to it. Every call
 * site in run_benchmarks.cpp passes 1.0; the 0.5 default is not used. Problems
 * whose error scale far exceeds the threshold (BearingOnly ~64, ReentryVehicle
 * ~369) therefore never converge by this definition and always report NaN --
 * that is a real property of the metric, not a filter failure. Reporting it
 * usefully for those problems needs a per-problem or relative threshold.
 */
template<typename State>
float compute_convergence_time(const std::vector<float>& times,
                                const std::vector<State>& true_states,
                                const std::vector<State>& estimated_states,
                                float threshold = 0.5f) {
    if (true_states.size() != estimated_states.size()) {
        return std::numeric_limits<float>::quiet_NaN();
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

    return std::numeric_limits<float>::quiet_NaN();  // Never converged
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
            // smoothed_states[k] is the estimate for time index (k - smoother_lag),
            // so the smoothed estimate aligned to row i lives at index i + lag.
            // The final `lag` rows have no smoothed estimate yet: emit empty fields
            // (parsed as NaN) instead of the spurious pre-warmup zeros.
            size_t s_idx = i + static_cast<size_t>(data.smoother_lag);
            if (s_idx < data.smoothed_states.size()) {
                for (int j = 0; j < state_dim; ++j) {
                    file << "," << data.smoothed_states[s_idx](j);
                }
            } else {
                for (int j = 0; j < state_dim; ++j) {
                    file << ",";
                }
            }
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

} // namespace Benchmark

#endif // BENCHMARK_RUNNER_H
