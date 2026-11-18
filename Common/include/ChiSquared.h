#ifndef NLF_CHI_SQUARED_H
#define NLF_CHI_SQUARED_H

// =============================================================================
// Exact chi-squared distribution functions (double precision).
// -----------------------------------------------------------------------------
// chi2_n is Gamma(n/2, scale 2), so
//     P(chi2_n <= x) = P(n/2, x/2),   P(chi2_n > x) = Q(n/2, x/2),
// with P, Q the regularized lower / upper incomplete gamma functions.
//
// Users:
//   * Benchmarks (NEES 95% band): quantile(p, n), lower tail.
//   * SRUKF innovation gate: upper_quantile(q, n), the x with
//     P(chi2_n > x) = q. Solved on Q directly, so q = 1e-12 keeps full
//     relative precision (solving P = 1 - q would lose it to cancellation).
//
// Moved here from Benchmarks/include/BenchmarkRunner.h (2026-11-18), which
// keeps its float chi2_quantile() wrapper for the existing call sites.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <limits>

namespace nlf_chi2 {

namespace detail {

// Series (Abramowitz & Stegun 6.5.29) for x < a + 1, returning P;
// modified-Lentz continued fraction (A&S 6.5.31) otherwise, returning Q.
// `is_p` reports which one was computed.
inline double gamma_tail_raw(double a, double x, bool& is_p) {
    const double log_prefactor = a * std::log(x) - x - std::lgamma(a);
    const double eps = 1e-15;
    if (x < a + 1.0) {
        is_p = true;
        double term = 1.0 / a;
        double sum = term;
        for (int k = 1; k < 1000; ++k) {
            term *= x / (a + k);
            sum += term;
            if (std::abs(term) < std::abs(sum) * eps) break;
        }
        return sum * std::exp(log_prefactor);
    }
    is_p = false;
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
    return std::exp(log_prefactor) * h;
}

}  // namespace detail

/** Regularized lower incomplete gamma P(a, x), a > 0, x >= 0. */
inline double regularized_gamma_p(double a, double x) {
    if (x <= 0.0) return 0.0;
    bool is_p = false;
    const double v = detail::gamma_tail_raw(a, x, is_p);
    return is_p ? v : 1.0 - v;
}

/** Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), a > 0, x >= 0.
 *  Computed directly (not as 1 - P) in the tail x >= a + 1. */
inline double regularized_gamma_q(double a, double x) {
    if (x <= 0.0) return 1.0;
    bool is_p = false;
    const double v = detail::gamma_tail_raw(a, x, is_p);
    return is_p ? 1.0 - v : v;
}

namespace detail {

// Solve F(y) = target for y > 0, where F is P(a, .) (upper = false) or
// 1 - Q(a, .) evaluated as target - Q (upper = true); both increase in y with
// derivative y^{a-1} e^{-y} / Gamma(a). Bisection to a tight bracket, then
// Newton to round-off; a Newton step that leaves the bracket is discarded.
inline double solve_gamma(double a, double target, bool upper) {
    const auto f = [&](double y) {
        return upper ? target - regularized_gamma_q(a, y) : regularized_gamma_p(a, y) - target;
    };
    double lo = 0.0;
    double hi = std::max(1.0, 2.0 * a);
    while (f(hi) < 0.0) {
        lo = hi;
        hi *= 2.0;
    }
    for (int i = 0; i < 200 && (hi - lo) > 1e-12 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) < 0.0) lo = mid; else hi = mid;
    }
    double y = 0.5 * (lo + hi);
    for (int i = 0; i < 8; ++i) {
        const double fy = f(y);
        const double df = std::exp((a - 1.0) * std::log(y) - y - std::lgamma(a));
        if (!(df > 0.0) || !std::isfinite(df)) break;
        const double y_new = y - fy / df;
        if (!(y_new > lo && y_new < hi)) break;
        if (std::abs(y_new - y) < 1e-15 * y) { y = y_new; break; }
        y = y_new;
    }
    return y;
}

}  // namespace detail

/** Lower quantile: the x with P(chi2_n <= x) = p, 0 < p < 1, n >= 1. NaN otherwise. */
inline double quantile(double p, int n) {
    if (!(p > 0.0 && p < 1.0) || n < 1) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 * detail::solve_gamma(0.5 * n, p, false);
}

/** Upper quantile: the x with P(chi2_n > x) = q, 0 < q < 1, n >= 1. NaN otherwise. */
inline double upper_quantile(double q, int n) {
    if (!(q > 0.0 && q < 1.0) || n < 1) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 * detail::solve_gamma(0.5 * n, q, true);
}

}  // namespace nlf_chi2

#endif  // NLF_CHI_SQUARED_H
