// Regression tests for the 2026-11-15 PKF audit (CTest PKF_AuditRegressions).
//
//  1. get_mean()/get_covariance() must equal the exact moments right after
//     initialize() and after an all-dead (-inf) step. The removed CUDA
//     GPUParticleContext read a never-written / stale linear-weight buffer
//     here: mean 0 (exact 0.4995) and 0.871 after the reset.
//  2. Equal-weight stratified resampling of N = 4e6 particles must give almost
//     every particle exactly one offspring. With a float offset u, 7.7% got none.
// All checks are NLF_CHECK (survive NDEBUG).

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include "particle_filter.hpp"
#include "resampling.hpp"
#include "state_space_model.hpp"
#include "TestCheck.h"

namespace {

struct Static1D : PKF::StateSpaceModel<1, 1> {
    bool dead = false;
    State propagate(const State& x, float, const Eigen::Ref<const State>&) const override { return x; }
    Observation observe(const State& x, float) const override { return x; }
    State sample_process_noise(float, std::mt19937_64&) const override { return State::Zero(); }
    Observation sample_observation_noise(float, std::mt19937_64&) const override { return Observation::Zero(); }
    float observation_loglik(const Observation& y, const State& x, float) const override {
        if (dead) return -std::numeric_limits<float>::infinity();
        const float d = y(0) - x(0);
        return -0.5f * d * d / 0.01f;
    }
};

}  // namespace

int main() {
    // ---- 1. Moments through whatever path get_mean() takes ----------------
    {
        Static1D m;
        const size_t N = 1000;
        PKF::ParticleFilter<1, 1> pf(&m, N, 0.0f /* never resample */);
        pf.set_seed(1);
        size_t i = 0;
        pf.initialize([&](std::mt19937_64&) {
            return Eigen::Matrix<float, 1, 1>::Constant(static_cast<float>(i++) / N);
        });

        const float exact_mean = 0.4995f;           // mean of {0, 1/N, ..., (N-1)/N}
        const float exact_var = (static_cast<float>(N) * N - 1.0f) / (12.0f * N * N);
        NLF_CHECK(std::abs(pf.get_mean()(0) - exact_mean) < 1e-4f, "get_mean() after initialize()");

        Eigen::Matrix<float, 1, 1> y; y << 0.9f;
        const Eigen::Matrix<float, 1, 1> u = Eigen::Matrix<float, 1, 1>::Zero();
        pf.step(y, 1.0f, u);
        NLF_CHECK(std::abs(pf.get_mean()(0) - pf.compute_mean_cpu()(0)) < 1e-4f,
                  "get_mean() equals the CPU mean after an informative step");

        m.dead = true;
        pf.step(y, 2.0f, u);   // every log-likelihood -inf -> weights reset to uniform
        std::printf("after reset: get_mean %.5f get_cov %.5f (exact %.5f, %.5f)\n",
                    pf.get_mean()(0), pf.get_covariance()(0, 0), exact_mean, exact_var);
        NLF_CHECK(std::abs(pf.get_mean()(0) - exact_mean) < 1e-4f, "get_mean() after an all-dead reset");
        NLF_CHECK(std::abs(pf.get_covariance()(0, 0) - exact_var) < 1e-4f,
                  "get_covariance() after an all-dead reset");
    }

    // ---- 2. Resampling precision at large N --------------------------------
    {
        const size_t N = 4000000;
        std::mt19937_64 rng(7);
        const std::vector<float> wf(N, 1.0f / static_cast<float>(N));
        const auto parents = PKF::Resampling::stratified(wf, rng);
        std::vector<unsigned> count(N, 0);
        for (size_t p : parents) ++count[p];
        size_t orphans = 0;
        for (unsigned c : count) orphans += (c == 0);
        const double frac = static_cast<double>(orphans) / N;
        std::printf("stratified(float weights), N=4e6: %.3f%% without offspring\n", 100.0 * frac);
        // float(1/N) itself is inexact, so a few strata still shift; measured 0.5%
        // with a double offset, 7.7% with the old float offset.
        NLF_CHECK(frac < 0.02, "equal-weight stratified resampling keeps almost every particle");
    }

    std::printf("test_pkf_audit: all checks passed\n");
    return 0;
}
