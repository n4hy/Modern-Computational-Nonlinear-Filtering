// Regression test for Benchmark::chi2_quantile() — the chi-squared quantiles
// that bound the NEES 95% consistency band in compute_nees().
//
// The previous implementation was a Wilson-Hilferty approximation whose lower
// 2.5% quantile was 48% low at n = 2, so Pct_In_Bounds admitted steps it should
// have rejected. This pins the exact values: references are
// scipy.stats.chi2.ppf (scipy 1.18.0), which agrees with the standard tables.

#include <cmath>
#include <cstdio>
#include "BenchmarkRunner.h"
#include "TestCheck.h"

namespace {

struct Ref {
    int n;
    double p;
    double x;
};

// scipy.stats.chi2.ppf(p, n)
const Ref kRefs[] = {
    {1,   0.025, 0.000982069117}, {1,   0.975, 5.02388619},
    {1,   0.05,  0.00393214},     {1,   0.5,   0.454936423},
    {2,   0.025, 0.050635616},    {2,   0.975, 7.37775891},
    {2,   0.05,  0.102586589},    {2,   0.5,   1.38629436},
    {3,   0.025, 0.215795283},    {3,   0.975, 9.3484036},
    {4,   0.025, 0.484418557},    {4,   0.975, 11.1432868},
    {4,   0.05,  0.710723021},    {4,   0.5,   3.35669398},
    {6,   0.025, 1.23734425},     {6,   0.975, 14.4493753},
    {6,   0.5,   5.34812063},
    {10,  0.025, 3.24697278},     {10,  0.975, 20.4831774},
    {10,  0.05,  3.94029914},     {10,  0.5,   9.34181777},
    {40,  0.025, 24.4330392},     {40,  0.975, 59.3417071},
    {40,  0.5,   39.3353448},
    {100, 0.025, 74.2219275},     {100, 0.975, 129.561197},
    {100, 0.05,  77.9294652},     {100, 0.5,   99.3341292},
};

}  // namespace

int main() {
    using Benchmark::chi2_quantile;
    using Benchmark::regularized_gamma_p;

    // The function returns float, so the achievable agreement is bounded by
    // float rounding of the result (~1.2e-7 relative); 1e-5 leaves margin.
    const double kRelTol = 1e-5;

    for (const Ref& r : kRefs) {
        const double got = chi2_quantile(static_cast<float>(r.p), r.n);
        const double rel = std::abs(got - r.x) / r.x;
        std::printf("n=%3d p=%.3f  got=%.9g  ref=%.9g  rel=%.2e\n",
                    r.n, r.p, got, r.x, rel);
        NLF_CHECK(std::isfinite(got), "quantile is finite");
        NLF_CHECK(rel < kRelTol, "quantile matches scipy.stats.chi2.ppf to 1e-5 relative");
    }

    // For n = 2 the distribution is exponential with mean 2, so the quantile is
    // closed-form: x = -2 ln(1 - p). An independent check not routed through scipy.
    for (double p : {0.01, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99}) {
        const double got = chi2_quantile(static_cast<float>(p), 2);
        const double exact = -2.0 * std::log1p(-p);
        NLF_CHECK(std::abs(got - exact) / exact < kRelTol,
                  "n=2 quantile matches -2 ln(1-p)");
    }

    // Round trip: P(n/2, x/2) == p for the returned x, across dimensions used
    // by the benchmark suite and beyond.
    for (int n : {1, 2, 4, 6, 10, 40, 100, 400}) {
        for (double p : {0.025, 0.975}) {
            const double x = chi2_quantile(static_cast<float>(p), n);
            const double back = regularized_gamma_p(0.5 * n, 0.5 * x);
            // Float rounding of x moves P by ~ x * pdf(x) * 1e-7; well under 1e-5.
            NLF_CHECK(std::abs(back - p) < 1e-5, "CDF(quantile(p)) == p");
        }
    }

    // Monotone in p and in n.
    NLF_CHECK(chi2_quantile(0.025f, 4) < chi2_quantile(0.5f, 4) &&
              chi2_quantile(0.5f, 4) < chi2_quantile(0.975f, 4),
              "quantile increases with p");
    NLF_CHECK(chi2_quantile(0.975f, 4) < chi2_quantile(0.975f, 5),
              "quantile increases with n");

    // Out-of-domain inputs yield NaN rather than a silently wrong bound.
    NLF_CHECK(std::isnan(chi2_quantile(0.0f, 4)), "p=0 -> NaN");
    NLF_CHECK(std::isnan(chi2_quantile(1.0f, 4)), "p=1 -> NaN");
    NLF_CHECK(std::isnan(chi2_quantile(0.5f, 0)), "n=0 -> NaN");

    std::printf("test_chi2_quantile: all checks passed\n");
    return 0;
}
