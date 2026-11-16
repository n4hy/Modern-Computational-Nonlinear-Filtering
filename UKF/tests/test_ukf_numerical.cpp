// Regression tests for the UKF Joseph-form covariance update + LLT/LDLT
// recovery ladder introduced in the audit-remediation pass. These would have
// caught the pre-fix behaviour where a rank-deficient P0 or an unusually-
// stressful measurement update could leave P non-PSD without recovery.

#include <iostream>
#include <cmath>
#include "TestCheck.h"  // NLF_REQUIRE: survives NDEBUG, unlike assert()
#include <Eigen/Dense>
#include "UKF.h"
#include "StateSpaceModel.h"

namespace {

// Minimal 2D linear model: identity dynamics + identity observation. Chosen to
// isolate covariance-update numerics from any nonlinearity.
class LinearModel2D : public UKFModel::StateSpaceModel<2, 2> {
public:
    State f(const State& x_prev, float, const Eigen::Ref<const State>&) const override {
        return x_prev;
    }
    Observation h(const State& x_k, float) const override {
        return x_k;
    }
    StateMat Q(float) const override {
        StateMat q = StateMat::Identity();
        q *= 1e-4f;
        return q;
    }
    ObsMat R(float) const override {
        ObsMat r = ObsMat::Identity();
        r *= 1e-2f;
        return r;
    }
};

// Check that a symmetric matrix M is PSD by requiring LLT to succeed on
// M + tiny*I; the tiny relative jitter tolerates float rounding on the eigenvalue
// that would otherwise sit right at zero.
bool is_psd(const Eigen::Matrix2f& M) {
    Eigen::Matrix2f M_sym = 0.5f * (M + M.transpose());
    float scale = std::max(M_sym.norm(), 1.0f);
    Eigen::LLT<Eigen::Matrix2f> llt(M_sym + 1e-6f * scale * Eigen::Matrix2f::Identity());
    return llt.info() == Eigen::Success;
}

}

// Case 1 — feed a rank-deficient P0 (one zero eigenvalue) and verify that after
// 10 predict/update cycles the filter output stays finite and P stays PSD.
void test_rank_deficient_p0() {
    std::cout << "test_rank_deficient_p0..." << std::endl;
    LinearModel2D model;
    UKFCore::UKF<2, 2> filter(model);

    // P0 has an obvious zero eigenvalue (rank 1). Pre-fix UKF would silently
    // propagate this into a permanently-degenerate covariance and eventually
    // return non-PSD values from update().
    Eigen::Matrix2f P0;
    P0 << 1.0f, 1.0f,
          1.0f, 1.0f;
    Eigen::Vector2f x0(0.0f, 0.0f);
    filter.initialize(x0, P0);

    Eigen::Vector2f u = Eigen::Vector2f::Zero();
    for (int k = 0; k < 10; ++k) {
        Eigen::Matrix2f P_before = filter.getCovariance();
        (void)P_before;

        filter.predict(static_cast<float>(k) * 0.1f, u);

        Eigen::Vector2f y(0.1f, -0.1f);
        filter.update(static_cast<float>(k) * 0.1f, y);

        Eigen::Matrix2f P = filter.getCovariance();
        Eigen::Vector2f x = filter.getState();

        NLF_REQUIRE(P.allFinite(), "P became NaN/Inf");
        NLF_REQUIRE(x.allFinite(), "state became NaN/Inf");
        NLF_REQUIRE(is_psd(P), "P is not PSD after update");
    }
    std::cout << "  OK" << std::endl;
}

// Case 2 — well-conditioned linear problem, run 20 update cycles, and require
// that trace(P) contracts monotonically toward the steady state. This is the
// behaviour the Joseph-consistent P = P - K S K^T update guarantees for a
// linear-Gaussian system, and would fail if the recovery ladder ever
// substituted an over-inflated fallback.
void test_well_conditioned_convergence() {
    std::cout << "test_well_conditioned_convergence..." << std::endl;
    LinearModel2D model;
    UKFCore::UKF<2, 2> filter(model);

    Eigen::Matrix2f P0 = 10.0f * Eigen::Matrix2f::Identity();
    Eigen::Vector2f x0(0.0f, 0.0f);
    filter.initialize(x0, P0);

    Eigen::Vector2f u = Eigen::Vector2f::Zero();
    float last_trace = P0.trace();
    for (int k = 0; k < 20; ++k) {
        filter.predict(static_cast<float>(k) * 0.1f, u);
        Eigen::Vector2f y(0.0f, 0.0f);
        filter.update(static_cast<float>(k) * 0.1f, y);

        Eigen::Matrix2f P = filter.getCovariance();
        NLF_REQUIRE(P.allFinite(), "P.allFinite()");
        NLF_REQUIRE(is_psd(P), "is_psd(P)");

        float tr = P.trace();
        // Allow small numerical wobble around steady state.
        NLF_REQUIRE(tr < last_trace + 1e-3f, "P grew without a corresponding measurement change");
        last_trace = tr;
    }
    // At steady state, trace(P) should be O(measurement noise), i.e. much
    // smaller than the initial 20.0.
    NLF_REQUIRE(last_trace < 1.0f, "trace(P) failed to contract into the measurement-noise regime");
    std::cout << "  OK (final trace = " << last_trace << ")" << std::endl;
}

int main() {
    test_rank_deficient_p0();
    test_well_conditioned_convergence();
    std::cout << "test_ukf_numerical: all cases passed." << std::endl;
    return 0;
}
