#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>
#include "particle_filter.hpp"
#include "resampling.hpp"
#include "noise_models.hpp"

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

/** Verify that systematic and stratified resampling produce correct-sized output. */
void test_resampling() {
    std::cout << "Testing Resampling..." << std::endl;
    std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.4f};
    std::mt19937_64 rng(42);

    auto parents = PKF::Resampling::systematic(weights, rng);
    assert(parents.size() == 4);

    parents = PKF::Resampling::stratified(weights, rng);
    assert(parents.size() == 4);

    std::cout << "Resampling tests passed." << std::endl;
}

/** Sanity-check the 1D particle filter: one step should produce a mean near 1.0. */
void test_particle_filter() {
    std::cout << "Testing Particle Filter..." << std::endl;
    TestModel1D model;
    PKF::ParticleFilter<1, 1> pf(&model, 100);

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
    assert(std::abs(mean(0) - 1.0f) < 1.0f);

    std::cout << "Particle Filter tests passed." << std::endl;
}

/** Run all particle filter unit tests. */
int main() {
    test_resampling();
    test_particle_filter();
    return 0;
}
