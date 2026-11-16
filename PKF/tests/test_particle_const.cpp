// Compile-time canary: ParticleFilter::get_mean(), get_covariance() and
// compute_mean_cpu() are const-callable. They were non-const while the CUDA
// GPU context (removed 2026-11-16) mutated device state inside them; with a
// pure CPU implementation a const filter must be readable without const_cast.

#include <iostream>
#include "TestCheck.h"  // NLF_REQUIRE: survives NDEBUG, unlike assert()
#include <cmath>
#include <type_traits>
#include "particle_filter.hpp"
#include "state_space_model.hpp"

namespace {

class TinyModel1D : public PKF::StateSpaceModel<1, 1> {
public:
    State propagate(const State& x_prev, float, const Eigen::Ref<const State>& u_k) const override {
        return x_prev + u_k;
    }
    Observation observe(const State& x_k, float) const override { return x_k; }
    State sample_process_noise(float, std::mt19937_64& rng) const override {
        std::normal_distribution<float> d(0.0f, 0.1f);
        return State::Constant(d(rng));
    }
    Observation sample_observation_noise(float, std::mt19937_64& rng) const override {
        std::normal_distribution<float> d(0.0f, 0.5f);
        return Observation::Constant(d(rng));
    }
    float observation_loglik(const Observation& y_k, const State& x_k, float) const override {
        float diff = y_k(0) - x_k(0);
        return -0.5f * (std::log(2.0f * static_cast<float>(M_PI) * 0.25f) + diff * diff / 0.25f);
    }
};

using PF = PKF::ParticleFilter<1, 1>;

// SFINAE probes: `has_const_get_mean<PF>::value` is true iff `pf.get_mean()`
// is well-formed when `pf` is `const PF`.
template<typename T, typename = void>
struct has_const_get_mean : std::false_type {};

template<typename T>
struct has_const_get_mean<
    T,
    std::void_t<decltype(std::declval<const T&>().get_mean())>>
    : std::true_type {};

template<typename T, typename = void>
struct has_const_get_covariance : std::false_type {};

template<typename T>
struct has_const_get_covariance<
    T,
    std::void_t<decltype(std::declval<const T&>().get_covariance())>>
    : std::true_type {};

template<typename T, typename = void>
struct has_const_compute_mean_cpu : std::false_type {};

template<typename T>
struct has_const_compute_mean_cpu<
    T,
    std::void_t<decltype(std::declval<const T&>().compute_mean_cpu())>>
    : std::true_type {};

static_assert(has_const_get_mean<PF>::value, "get_mean() must be const-callable.");
static_assert(has_const_get_covariance<PF>::value, "get_covariance() must be const-callable.");
static_assert(has_const_compute_mean_cpu<PF>::value,
              "compute_mean_cpu() must be const-callable.");

}

int main() {
    std::cout << "test_particle_const: compile-time canary passed." << std::endl;

    // Runtime sanity: the accessors produce sensible values.
    TinyModel1D model;
    PF pf(&model, 100);
    pf.set_seed(1234);
    pf.initialize([](std::mt19937_64& r) {
        std::normal_distribution<float> d(0.0f, 1.0f);
        return Eigen::Matrix<float, 1, 1>::Constant(d(r));
    });
    Eigen::Matrix<float, 1, 1> u; u << 1.0f;
    Eigen::Matrix<float, 1, 1> y; y << 1.0f;
    pf.step(y, 1.0f, u);

    auto mean = pf.get_mean();
    auto cov = pf.get_covariance();
    NLF_REQUIRE(std::isfinite(mean(0)), "std::isfinite(mean(0))");
    NLF_REQUIRE(std::isfinite(cov(0, 0)) && cov(0, 0) >= 0.0f, "std::isfinite(cov(0, 0)) && cov(0, 0) >= 0.0f");

    // Through a const reference: identical values.
    const PF& cpf = pf;
    NLF_REQUIRE(cpf.compute_mean_cpu()(0) == mean(0), "compute_mean_cpu() == get_mean()");
    NLF_REQUIRE(cpf.get_mean()(0) == mean(0), "const get_mean() == get_mean()");
    NLF_REQUIRE(cpf.get_covariance()(0, 0) == cov(0, 0), "const get_covariance() == get_covariance()");

    std::cout << "test_particle_const: runtime canary passed." << std::endl;
    return 0;
}
