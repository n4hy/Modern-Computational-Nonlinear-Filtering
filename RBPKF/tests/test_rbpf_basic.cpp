#include <iostream>
#include <cstdlib>
#include <cmath>
#include "rbpf/types.hpp"
#include "rbpf/rbpf_config.hpp"
#include "rbpf/rbpf_core.hpp"

// Live in Release too (assert() would be stripped under NDEBUG). Exits non-zero
// so a real failure fails CTest instead of silently printing "Test passed!".
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":"      \
                      << __LINE__ << std::endl;                                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Minimal instantiation test

// 1. Define Types
using TestTypes = rbpf::RbpfTypes<1, 2, 1>; // 1 NL, 2 LIN, 1 Obs

// 2. Define Dummy Models
class DummyNL : public rbpf::NonlinearModel<TestTypes> {
public:
    NonlinearState propagate(const NonlinearState& x, float, const NonlinearState&, std::mt19937_64&) const override {
        return x;
    }
    float log_proposal_density(const NonlinearState&, const NonlinearState&, float, const NonlinearState&) const override {
        return 0.0f;
    }
};

class DummyLin : public rbpf::ConditionalLinearGaussianModel<TestTypes> {
public:
    void get_dynamics(const NonlinearState&, float, Eigen::Ref<LinearState> bias, Eigen::Ref<Eigen::MatrixXf> A, Eigen::Ref<Eigen::MatrixXf> B, Eigen::Ref<LinearCov> Q) const override {
        bias.setZero();
        A.setIdentity();
        B.setZero();
        Q.setIdentity();
    }
    void get_observation(const NonlinearState&, float, Eigen::Ref<Observation> offset, Eigen::Ref<Eigen::MatrixXf> H, Eigen::Ref<ObsCov> R) const override {
        offset.setZero();
        // H is Matrix<float, Ny, Nlin> = 1x2.
        H.setZero(); H(0,0)=1.0f;
        R.setIdentity();
    }
};

/** Minimal smoke test: instantiate the RBPF template, initialize, step once,
 *  and retrieve filtered mean to verify compilation and basic functionality. */
int main() {
    rbpf::RbpfConfig config;
    config.num_particles = 10;

    DummyNL nl;
    DummyLin lin;

    rbpf::RaoBlackwellizedParticleFilter<TestTypes, DummyNL, DummyLin> filter(nl, lin, config);

    TestTypes::NonlinearState xnl; xnl.setZero();
    TestTypes::LinearState xlin; xlin.setZero();
    TestTypes::LinearCov Plin; Plin.setIdentity();

    filter.initialize(xnl, xlin, Plin);

    TestTypes::Observation y; y.setZero();
    TestTypes::NonlinearState u; u.setZero();

    filter.step(0.0f, y, u);

    TestTypes::NonlinearState est_nl;
    TestTypes::LinearState est_lin;
    filter.get_filtered_mean(est_nl, est_lin);

    // With this dummy model (NL state fixed at 0, y = 0, H = [1 0]) both the
    // nonlinear and linear filtered means must remain finite and near zero.
    CHECK(std::isfinite(est_nl(0)));
    CHECK(std::abs(est_nl(0)) < 1e-3f);
    for (int i = 0; i < TestTypes::Nlin; ++i) {
        CHECK(std::isfinite(est_lin(i)));
        CHECK(std::abs(est_lin(i)) < 1e-1f);
    }

    // Exercise the fixed-lag smoother ancestry path (covers the initialize/step
    // counter fix): with fixed_lag > 0 and several steps, smoothing at the max
    // lag must succeed and return finite estimates that reach the initial state.
    {
        rbpf::RbpfConfig cfg2;
        cfg2.num_particles = 20;
        cfg2.fixed_lag = 3;
        cfg2.seed = 7;
        rbpf::RaoBlackwellizedParticleFilter<TestTypes, DummyNL, DummyLin> f2(nl, lin, cfg2);
        f2.initialize(xnl, xlin, Plin);
        for (int k = 0; k < 5; ++k) f2.step(static_cast<float>(k), y, u);

        CHECK(f2.can_smooth(cfg2.fixed_lag));  // enough history accumulated
        TestTypes::NonlinearState s_nl;
        TestTypes::LinearState s_lin;
        f2.get_smoothed_mean(cfg2.fixed_lag, s_nl, s_lin);
        CHECK(std::isfinite(s_nl(0)));
        CHECK(std::abs(s_nl(0)) < 1e-3f);
        for (int i = 0; i < TestTypes::Nlin; ++i) CHECK(std::isfinite(s_lin(i)));
    }

    std::cout << "Test passed!" << std::endl;
    return 0;
}
