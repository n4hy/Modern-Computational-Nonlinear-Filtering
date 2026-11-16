// Regression test for the RBPF per-particle predictive log-likelihood,
// log N(innovation; 0, S) with S = H P H^T + R.
//
// It calls RaoBlackwellizedParticleFilter<...>::predictive_log_likelihood, the
// exact function step() uses. (An earlier version of this file re-implemented
// the algorithm in-file, so it stayed green whatever rbpf_core.hpp did; it also
// used assert(), which -DNDEBUG removes in Release.)
//
// Defects pinned here:
//  * A factorisation failure on a float-rounded S used to set log_det = -1e10,
//    i.e. log_lik ~ +5e9: the failed particle took all the weight. It must now
//    be a finite, sane value (jitter ladder) or -inf -- never a bonus.
//  * An indefinite S or a non-finite innovation must give -inf (zero weight).
// References are computed independently in double precision.

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <Eigen/Dense>
#include "rbpf/rbpf_core.hpp"
#include "TestCheck.h"

namespace {

using T3 = rbpf::RbpfTypes<1, 1, 3>;
using T2 = rbpf::RbpfTypes<1, 1, 2>;
template<typename T>
using RBPF = rbpf::RaoBlackwellizedParticleFilter<T, rbpf::NonlinearModel<T>,
                                                  rbpf::ConditionalLinearGaussianModel<T>>;

// Double-precision reference: -0.5 (z^T S^{-1} z + log|S| + n log 2 pi).
template<int N>
double ref_loglik(const Eigen::Matrix<float, N, N>& S, const Eigen::Matrix<float, N, 1>& z) {
    const Eigen::Matrix<double, N, N> Sd = S.template cast<double>();
    const Eigen::Matrix<double, N, 1> zd = z.template cast<double>();
    Eigen::LLT<Eigen::Matrix<double, N, N>> llt(Sd);
    double log_det = 0.0;
    for (int k = 0; k < N; ++k) log_det += 2.0 * std::log(llt.matrixLLT()(k, k));
    const double m = zd.dot(llt.solve(zd));
    return -0.5 * (m + log_det + N * std::log(2.0 * std::numbers::pi));
}

// SPD matrix with eigenvalues geometrically spaced in [1/cond, 1].
Eigen::Matrix3f make_conditioned(float cond) {
    Eigen::Matrix3f D = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) D(i, i) = std::pow(cond, -0.5f * static_cast<float>(i));
    Eigen::Matrix3f A;
    A << 0.3f, -1.1f, 0.7f,  0.9f, 0.2f, -0.4f,  -0.5f, 0.6f, 1.3f;  // fixed, full rank
    const Eigen::Matrix3f Q = Eigen::HouseholderQR<Eigen::Matrix3f>(A).householderQ();
    Eigen::Matrix3f S = Q * D * Q.transpose();
    return 0.5f * (S + S.transpose());
}

}  // namespace

int main() {
    const Eigen::Vector3f z(0.3f, -0.2f, 0.1f);

    // 1. Well-conditioned and moderately ill-conditioned S: agree with double.
    for (float cond : {1.0f, 1e3f, 1e6f}) {
        const Eigen::Matrix3f S = make_conditioned(cond);
        const float got = RBPF<T3>::predictive_log_likelihood(S, z);
        const double ref = ref_loglik<3>(S, z);
        const double err = std::abs(static_cast<double>(got) - ref) / std::max(std::abs(ref), 1.0);
        std::cout << "cond=" << cond << " loglik=" << got << " ref=" << ref << " rel=" << err << "\n";
        NLF_CHECK(std::isfinite(got), "log-likelihood is finite for an SPD S");
        // Float factorisation error grows like cond * eps (the reference factors
        // the same float S in double, so this is the float solve's own error):
        // measured 9e-8 at cond 1, 5e-6 at 1e3, 4.3e-3 at 1e6.
        const double tol = std::max(1e-5, 4.0 * cond * std::numeric_limits<float>::epsilon());
        NLF_CHECK(err < tol, "log-likelihood matches the double-precision reference");
    }

    // 2. The case that used to return ~+5e9: S = H P H^T + R with H = [1; 1.12553298],
    //    P = 199.577728, R = 1e-6 I. Its 1e-6 eigenvalue is below float resolution
    //    relative to ~450, so the float S fails LLT. Any answer within float noise
    //    is acceptable; a giant positive value (or NaN) is not.
    {
        Eigen::Vector2f H(1.0f, 1.12553298f);
        const float P = 199.577728f;
        const Eigen::Matrix2f S = H * P * H.transpose() + 1e-6f * Eigen::Matrix2f::Identity();
        const Eigen::Vector2f zz(0.01f, 0.012f);
        const float got = RBPF<T2>::predictive_log_likelihood(S, zz);
        std::cout << "near-singular S: loglik=" << got << "\n";
        NLF_CHECK(!std::isnan(got), "near-singular S does not give NaN");
        NLF_CHECK(got < 50.0f, "near-singular S is not rewarded with a huge log-likelihood");
    }

    // 3. Indefinite S (eigenvalues 3 and -1): no valid density -> zero weight.
    {
        Eigen::Matrix2f S;
        S << 1.0f, 2.0f, 2.0f, 1.0f;
        const float got = RBPF<T2>::predictive_log_likelihood(S, Eigen::Vector2f(0.1f, 0.1f));
        NLF_CHECK(std::isinf(got) && got < 0.0f, "indefinite S gives -inf");
    }

    // 4. Non-finite innovation -> zero weight, not NaN.
    {
        const Eigen::Matrix2f S = Eigen::Matrix2f::Identity();
        const Eigen::Vector2f bad(std::numeric_limits<float>::quiet_NaN(), 0.0f);
        const float got = RBPF<T2>::predictive_log_likelihood(S, bad);
        NLF_CHECK(std::isinf(got) && got < 0.0f, "NaN innovation gives -inf");
    }

    // 5. Closed form, S = s I (n = 2): -0.5 (|z|^2/s + 2 log s + 2 log 2 pi).
    {
        const float s = 0.25f;
        const Eigen::Vector2f zz(0.5f, -0.5f);
        const float got = RBPF<T2>::predictive_log_likelihood(s * Eigen::Matrix2f::Identity(), zz);
        const double exact = -0.5 * (0.5 / 0.25 + 2.0 * std::log(0.25) + 2.0 * std::log(2.0 * std::numbers::pi));
        NLF_CHECK(std::abs(got - exact) < 1e-5, "S = sI matches the closed form");
    }

    std::cout << "test_rbpf_logdet: all checks passed\n";
    return 0;
}
