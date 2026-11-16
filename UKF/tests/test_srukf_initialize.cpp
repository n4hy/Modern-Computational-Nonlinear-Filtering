// Regression test for SRUKF::initialize() — must throw on obviously-broken P0
// rather than silently degrading to an identity square-root. Pre-fix behaviour
// would accept NaN/Inf/asymmetric/non-PSD P0 and produce garbage estimates
// several steps into the trajectory, making the failure diagnosis very hard.

#include <iostream>
#include "TestCheck.h"  // NLF_REQUIRE: survives NDEBUG, unlike assert()
#include <stdexcept>
#include <Eigen/Dense>
#include "SRUKF.h"
#include "StateSpaceModel.h"

namespace {

class LinearModel2D : public UKFModel::StateSpaceModel<2, 2> {
public:
    State f(const State& x_prev, float, const Eigen::Ref<const State>&) const override {
        return x_prev;
    }
    Observation h(const State& x_k, float) const override { return x_k; }
    StateMat Q(float) const override { return 1e-4f * StateMat::Identity(); }
    ObsMat R(float) const override { return 1e-2f * ObsMat::Identity(); }
};

template<typename F>
bool throws_runtime_error(F&& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}

void test_nan_p0_throws() {
    std::cout << "test_nan_p0_throws..." << std::endl;
    LinearModel2D model;
    UKFCore::SRUKF<2, 2> filter(model);
    Eigen::Vector2f x0(0.0f, 0.0f);
    Eigen::Matrix2f P0 = Eigen::Matrix2f::Identity();
    P0(0, 0) = std::nanf("");
    NLF_REQUIRE(throws_runtime_error([&] { filter.initialize(x0, P0); }), "throws_runtime_error([&] { filter.initialize(x0, P0); })");
    std::cout << "  OK" << std::endl;
}

void test_inf_p0_throws() {
    std::cout << "test_inf_p0_throws..." << std::endl;
    LinearModel2D model;
    UKFCore::SRUKF<2, 2> filter(model);
    Eigen::Vector2f x0(0.0f, 0.0f);
    Eigen::Matrix2f P0 = Eigen::Matrix2f::Identity();
    P0(1, 1) = std::numeric_limits<float>::infinity();
    NLF_REQUIRE(throws_runtime_error([&] { filter.initialize(x0, P0); }), "throws_runtime_error([&] { filter.initialize(x0, P0); })");
    std::cout << "  OK" << std::endl;
}

void test_asymmetric_p0_throws() {
    std::cout << "test_asymmetric_p0_throws..." << std::endl;
    LinearModel2D model;
    UKFCore::SRUKF<2, 2> filter(model);
    Eigen::Vector2f x0(0.0f, 0.0f);
    Eigen::Matrix2f P0;
    P0 << 1.0f, 0.5f,
          0.0f, 1.0f;   // asymmetric by 0.5 relative
    NLF_REQUIRE(throws_runtime_error([&] { filter.initialize(x0, P0); }), "throws_runtime_error([&] { filter.initialize(x0, P0); })");
    std::cout << "  OK" << std::endl;
}

void test_non_psd_p0_throws() {
    std::cout << "test_non_psd_p0_throws..." << std::endl;
    LinearModel2D model;
    UKFCore::SRUKF<2, 2> filter(model);
    Eigen::Vector2f x0(0.0f, 0.0f);
    // Symmetric but indefinite (one negative eigenvalue).
    Eigen::Matrix2f P0;
    P0 << 1.0f, 2.0f,
          2.0f, 1.0f;
    NLF_REQUIRE(throws_runtime_error([&] { filter.initialize(x0, P0); }), "throws_runtime_error([&] { filter.initialize(x0, P0); })");
    std::cout << "  OK" << std::endl;
}

void test_valid_identity_p0_does_not_throw() {
    std::cout << "test_valid_identity_p0_does_not_throw..." << std::endl;
    LinearModel2D model;
    UKFCore::SRUKF<2, 2> filter(model);
    Eigen::Vector2f x0(0.0f, 0.0f);
    Eigen::Matrix2f P0 = Eigen::Matrix2f::Identity();
    // Must not throw.
    filter.initialize(x0, P0);
    // And the filter must be functional afterward.
    Eigen::Vector2f u = Eigen::Vector2f::Zero();
    filter.predict(0.0f, u);
    Eigen::Vector2f y(0.0f, 0.0f);
    filter.update(0.0f, y);
    Eigen::Matrix2f P = filter.getCovariance();
    NLF_REQUIRE(P.allFinite(), "P.allFinite()");
    std::cout << "  OK" << std::endl;
}

int main() {
    test_nan_p0_throws();
    test_inf_p0_throws();
    test_asymmetric_p0_throws();
    test_non_psd_p0_throws();
    test_valid_identity_p0_does_not_throw();
    std::cout << "test_srukf_initialize: all cases passed." << std::endl;
    return 0;
}
