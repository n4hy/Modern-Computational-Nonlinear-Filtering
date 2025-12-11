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
class TestModel1D : public PKF::StateSpaceModel<1, 1> {
public:
    State propagate(const State& x_prev, double t_k, const Eigen::Ref<const State>& u_k) const override {
        (void)t_k;
        return x_prev + u_k;
    }

    Observation observe(const State& x_k, double t_k) const override {
        (void)t_k;
        return x_k;
    }

    State sample_process_noise(double t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        std::normal_distribution<double> d(0.0, 0.1);
        return State::Constant(d(rng));
    }

    Observation sample_observation_noise(double t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        std::normal_distribution<double> d(0.0, 0.5);
        return Observation::Constant(d(rng));
    }

    double observation_loglik(const Observation& y_k, const State& x_k, double t_k) const override {
        (void)t_k;
        double diff = y_k(0) - x_k(0);
        return -0.5 * (std::log(2 * M_PI * 0.25) + diff * diff / 0.25);
    }
};

void test_resampling() {
    std::cout << "Testing Resampling..." << std::endl;
    std::vector<double> weights = {0.1, 0.2, 0.3, 0.4};
    std::mt19937_64 rng(42);

    auto parents = PKF::Resampling::systematic(weights, rng);
    assert(parents.size() == 4);

    parents = PKF::Resampling::stratified(weights, rng);
    assert(parents.size() == 4);

    std::cout << "Resampling tests passed." << std::endl;
}

void test_particle_filter() {
    std::cout << "Testing Particle Filter..." << std::endl;
    TestModel1D model;
    PKF::ParticleFilter<1, 1> pf(&model, 100);

    // Initialize
    pf.initialize([](std::mt19937_64& r) {
        std::normal_distribution<double> d(0.0, 1.0);
        return Eigen::Matrix<double, 1, 1>::Constant(d(r));
    });

    // Step
    Eigen::Matrix<double, 1, 1> u; u << 1.0;
    Eigen::Matrix<double, 1, 1> y; y << 1.0;

    pf.step(y, 1.0, u);

    auto mean = pf.get_mean();
    std::cout << "Filtered Mean: " << mean(0) << std::endl;

    // Basic sanity check: mean should be somewhere near 1.0 (prior 0 + u 1)
    // and dragged towards y=1.
    assert(std::abs(mean(0) - 1.0) < 1.0);

    std::cout << "Particle Filter tests passed." << std::endl;
}

int main() {
    test_resampling();
    test_particle_filter();
    return 0;
}
