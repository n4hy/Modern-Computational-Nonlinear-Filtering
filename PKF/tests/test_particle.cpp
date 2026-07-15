#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include "particle_filter.hpp"
#include "resampling.hpp"
#include "noise_models.hpp"

// assert() is compiled out under NDEBUG (Release), which would turn every check
// below into a no-op and make CTest pass regardless of correctness. CHECK stays
// live in all build types and exits non-zero so failures actually surface.
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":"      \
                      << __LINE__ << std::endl;                                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Simple 1D Linear Model for Testing
// x_k = x_{k-1} + u + w_k
// y_k = x_k + v_k
/** Simple 1D linear model for unit testing: x_k = x_{k-1} + u, y_k = x_k + noise. */
class TestModel1D : public PKF::StateSpaceModel<1, 1> {
public:
    /** Identity propagation plus control input. */
    State propagate(const State& x_prev, float t_k, const Eigen::Ref<const State>& u_k) const override {
        (void)t_k;
        return x_prev + u_k;
    }

    /** Direct observation of the scalar state. */
    Observation observe(const State& x_k, float t_k) const override {
        (void)t_k;
        return x_k;
    }

    /** Sample Gaussian process noise with std=0.1. */
    State sample_process_noise(float t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        std::normal_distribution<float> d(0.0f, 0.1f);
        return State::Constant(d(rng));
    }

    /** Sample Gaussian observation noise with std=0.5. */
    Observation sample_observation_noise(float t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        std::normal_distribution<float> d(0.0f, 0.5f);
        return Observation::Constant(d(rng));
    }

    /** Gaussian log-likelihood: N(y | x, 0.25). */
    float observation_loglik(const Observation& y_k, const State& x_k, float t_k) const override {
        (void)t_k;
        float diff = y_k(0) - x_k(0);
        return -0.5f * (std::log(2.0f * static_cast<float>(M_PI) * 0.25f) + diff * diff / 0.25f);
    }
};

/** Verify that systematic and stratified resampling actually follow the weights.
 *
 *  Checking only that N in-range indices come back is not a test of a resampler: an
 *  implementation that ignored the weights entirely and always returned {0,0,0,0}
 *  satisfies both of those checks and always picked the least likely particle. What
 *  defines these two algorithms is the distribution of offspring, so assert that.
 */
void test_resampling() {
    std::cout << "Testing Resampling..." << std::endl;
    const std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.4f};
    const size_t N = weights.size();
    const int kTrials = 4000;

    std::vector<double> share_sys(N, 0.0), share_str(N, 0.0);

    for (int t = 0; t < kTrials; ++t) {
        std::mt19937_64 rng(t + 1);

        auto sys = PKF::Resampling::systematic(weights, rng);
        CHECK(sys.size() == N);
        for (auto p : sys) CHECK(p < N);

        std::vector<int> count(N, 0);
        for (auto p : sys) count[p]++;

        // Systematic draws a single offset u0 ~ U(0, 1/N) and then steps by exactly
        // 1/N, so particle i's cumulative-weight interval -- of width w_i -- captures
        // either floor(N*w_i) or ceil(N*w_i) of the N evenly spaced probes. That holds
        // on EVERY draw, not merely in expectation. For these weights it pins the
        // counts to {0,1}, {0,1}, {1,2}, {1,2}: particles 2 and 3 must always be
        // selected at least once. No weight-insensitive implementation survives this.
        for (size_t i = 0; i < N; ++i) {
            const double expected = static_cast<double>(N) * weights[i];
            CHECK(count[i] >= static_cast<int>(std::floor(expected)));
            CHECK(count[i] <= static_cast<int>(std::ceil(expected)));
            share_sys[i] += count[i] / static_cast<double>(N);
        }

        // Stratified draws one uniform per stratum, so it admits a wider per-draw
        // spread than systematic and only the distributional claim below holds for it.
        auto str = PKF::Resampling::stratified(weights, rng);
        CHECK(str.size() == N);
        for (auto p : str) CHECK(p < N);
        std::vector<int> count_str(N, 0);
        for (auto p : str) count_str[p]++;
        for (size_t i = 0; i < N; ++i) share_str[i] += count_str[i] / static_cast<double>(N);
    }

    // Both schemes are unbiased -- E[count_i] = N * w_i -- so each particle's mean
    // share over kTrials seeded draws concentrates on its own weight. The 0.02 band is
    // many standard errors wide at this trial count (so it cannot flake) while still
    // being far tighter than the gap to any implementation that mis-weights particles.
    for (size_t i = 0; i < N; ++i) {
        share_sys[i] /= kTrials;
        share_str[i] /= kTrials;
        std::cout << "  w=" << weights[i]
                  << "  systematic share=" << share_sys[i]
                  << "  stratified share=" << share_str[i] << std::endl;
        CHECK(std::abs(share_sys[i] - weights[i]) < 0.02);
        CHECK(std::abs(share_str[i] - weights[i]) < 0.02);
    }

    std::cout << "Resampling tests passed." << std::endl;
}

/** Sanity-check the 1D particle filter: one step should produce a mean near 1.0. */
void test_particle_filter() {
    std::cout << "Testing Particle Filter..." << std::endl;
    TestModel1D model;
    PKF::ParticleFilter<1, 1> pf(&model, 100);
    pf.set_seed(42);  // deterministic run

    // Initialize
    pf.initialize([](std::mt19937_64& r) {
        std::normal_distribution<float> d(0.0f, 1.0f);
        return Eigen::Matrix<float, 1, 1>::Constant(d(r));
    });

    // Step
    Eigen::Matrix<float, 1, 1> u; u << 1.0f;
    Eigen::Matrix<float, 1, 1> y; y << 1.0f;

    pf.step(y, 1.0f, u);

    auto mean = pf.get_mean();
    std::cout << "Filtered Mean: " << mean(0) << std::endl;

    // Basic sanity check: mean should be somewhere near 1.0 (prior 0 + u 1)
    // and dragged towards y=1.
    CHECK(std::isfinite(mean(0)));
    CHECK(std::abs(mean(0) - 1.0f) < 1.0f);

    std::cout << "Particle Filter tests passed." << std::endl;
}

/** Run all particle filter unit tests. */
int main() {
    test_resampling();
    test_particle_filter();
    return 0;
}
