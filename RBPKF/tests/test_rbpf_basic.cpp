#include <iostream>
#include <cassert>
#include "rbpf/types.hpp"
#include "rbpf/rbpf_config.hpp"
#include "rbpf/rbpf_core.hpp"

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

    std::cout << "Test passed!" << std::endl;
    return 0;
}
