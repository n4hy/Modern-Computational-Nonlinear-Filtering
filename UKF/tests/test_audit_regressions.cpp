// Regression tests for the 2026-11-15 audit of the UKF / SRUKF filters and
// smoothers. Each block pins one defect that was demonstrated before the fix,
// using that demonstration's input as the fixture. Every check is an NLF_CHECK
// (survives NDEBUG); this program is the CTest case UKF_AuditRegressions.

#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <Eigen/Dense>
#include "UKF.h"
#include "SRUKF.h"
#include "SigmaPoints.h"
#include "UKFSmoother.h"
#include "SRUKFSmoother.h"
#include "UnscentedFixedLagSmoother.h"
#include "SRUKFFixedLagSmoother.h"
#include "StateSpaceModel.h"
#include "TestCheck.h"

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// y = x^2, Q = 0: for x ~ N(0, 1) the exact predicted mean/variance are 1 and 2.
struct Square1D : UKFModel::StateSpaceModel<1, 1> {
    State f(const State& x, float, const Eigen::Ref<const State>&) const override {
        return State(x(0) * x(0));
    }
    Observation h(const State& x, float) const override { return x; }
    StateMat Q(float) const override { return StateMat::Zero(); }
    ObsMat R(float) const override { return ObsMat::Constant(1.0f); }
};

// 1-D bearing: state and observation both angular, h wraps to (-pi, pi].
struct Bearing1D : UKFModel::StateSpaceModel<1, 1> {
    State f(const State& x, float, const Eigen::Ref<const State>&) const override { return x; }
    Observation h(const State& x, float) const override {
        return Observation(std::atan2(std::sin(x(0)), std::cos(x(0))));
    }
    StateMat Q(float) const override { return StateMat::Zero(); }
    ObsMat R(float) const override { return ObsMat::Constant(0.04f); }
    bool isAngularState(int) const override { return true; }
    bool isAngularObservation(int) const override { return true; }
};

// x_{k} = x_{k-1} + t_prev, y = x: noise-free, time-varying. Truth with
// t_prev = 0,1,2,3: x = 0, 0, 1, 3, 6.
struct AddTime1D : UKFModel::StateSpaceModel<1, 1> {
    State f(const State& x, float t_prev, const Eigen::Ref<const State>&) const override {
        return State(x(0) + t_prev);
    }
    Observation h(const State& x, float) const override { return x; }
    StateMat Q(float) const override { return StateMat::Constant(1e-6f); }
    ObsMat R(float) const override { return ObsMat::Constant(1e-6f); }
};

// Model that returns NaN once armed, to exercise the smoothers' rollback.
struct Poison1D : UKFModel::StateSpaceModel<1, 1> {
    mutable bool armed = false;
    State f(const State& x, float, const Eigen::Ref<const State>&) const override {
        return armed ? State(std::numeric_limits<float>::quiet_NaN()) : x;
    }
    Observation h(const State& x, float) const override { return x; }
    StateMat Q(float) const override { return StateMat::Constant(0.01f); }
    ObsMat R(float) const override { return ObsMat::Constant(0.1f); }
};

using V1 = Eigen::Matrix<float, 1, 1>;
V1 v1(float a) { return V1::Constant(a); }

}  // namespace

int main() {
    // ---- 1. Wc0 under the spread floor (SigmaPoints.h) -------------------
    // Default UKF alpha = 1e-3 trips the n+lambda < 0.5 floor. The old floor
    // kept alpha in Wc0 (Wc0 ~ 3) and returned var(x^2) = 3.
    {
        Square1D m;
        UKFCore::UKF<1, 1> ukf(m);
        ukf.initialize(v1(0.0f), v1(1.0f));
        ukf.predict(0.0f, v1(0.0f));
        std::printf("Wc0: mean %.6f var %.6f (exact 1, 2)\n", ukf.getState()(0), ukf.getCovariance()(0, 0));
        NLF_CHECK(std::abs(ukf.getState()(0) - 1.0f) < 1e-5f, "UT mean of x^2 is exact");
        NLF_CHECK(std::abs(ukf.getCovariance()(0, 0) - 2.0f) < 1e-4f,
                  "UT variance of x^2 is exact at the default alpha (Wc0 floor fix)");
        UKFCore::SigmaPoints<1> sp;
        UKFCore::compute_sigma_weights<1>(0.0f, 2.0f, 0.0f, sp);  // alpha = 0 used to give NaN
        NLF_CHECK(sp.Wm.allFinite() && sp.Wc.allFinite(), "alpha = 0 gives finite weights");
    }

    // ---- 2. Pivot-aware LDLT square root ---------------------------------
    {
        Eigen::Matrix2f A;
        A << 1e4f, 2e4f, 2e4f, 4.0001e4f;   // pivots: Eigen picks the larger diagonal first
        Eigen::LDLT<Eigen::Matrix2f> ldlt(A);
        const Eigen::Matrix2f S = UKFCore::ldlt_sqrt_factor<2>(ldlt);
        const float rel = (S * S.transpose() - A).norm() / A.norm();
        std::printf("LDLT sqrt: rel err %.3e\n", rel);
        NLF_CHECK(rel < 1e-5f, "ldlt_sqrt_factor reconstructs A, not P A P^T");
    }

    // ---- 3. Non-finite measurement is a missed measurement ----------------
    {
        Square1D m;
        UKFCore::UKF<1, 1> ukf(m);
        ukf.initialize(v1(0.5f), v1(0.2f));
        ukf.update(0.0f, v1(std::numeric_limits<float>::quiet_NaN()));
        NLF_CHECK(ukf.getState()(0) == 0.5f && ukf.getCovariance()(0, 0) == 0.2f,
                  "UKF::update ignores a NaN measurement");
        UKFCore::SRUKF<1, 1> sr(m);
        sr.initialize(v1(0.5f), v1(0.2f));
        sr.update(0.0f, v1(std::numeric_limits<float>::quiet_NaN()));
        NLF_CHECK(sr.getState()(0) == 0.5f && sr.getSqrtCovariance().allFinite(),
                  "SRUKF::update ignores a NaN measurement");
    }

    // ---- 4. Angular observation straddling +-pi ---------------------------
    // The straddling problem and its rotation by pi must agree; the rotated one
    // is the plain Kalman answer x+ = 3.0862, P+ = 0.03448.
    {
        Bearing1D m;
        const float x0 = 3.0f, y = 3.1f;
        UKFCore::SRUKF<1, 1> a(m), b(m);
        a.initialize(v1(x0), v1(0.25f));
        b.initialize(v1(x0 - kPi), v1(0.25f));
        a.update(0.0f, v1(y));
        b.update(0.0f, v1(y - kPi));
        const float xa = a.getState()(0), xb = b.getState()(0) + kPi;
        const float Pa = a.getCovariance()(0, 0), Pb = b.getCovariance()(0, 0);
        std::printf("SRUKF angular: straddling x+ %.5f P+ %.5f | rotated x+ %.5f P+ %.5f\n", xa, Pa, xb, Pb);
        NLF_CHECK(std::abs(xa - xb) < 1e-3f && std::abs(Pa - Pb) < 1e-4f,
                  "SRUKF angular update is invariant to the branch cut");
        UKFCore::UKF<1, 1> c(m);
        c.initialize(v1(x0), v1(0.25f));
        c.update(0.0f, v1(y));
        std::printf("UKF angular:   straddling x+ %.5f P+ %.5f\n", c.getState()(0), c.getCovariance()(0, 0));
        NLF_CHECK(std::abs(c.getState()(0) - xb) < 1e-3f &&
                  std::abs(c.getCovariance()(0, 0) - Pb) < 1e-4f,
                  "UKF honours isAngularState/isAngularObservation");
    }

    // ---- 5. Full smoothers propagate FROM t_prev --------------------------
    {
        AddTime1D m;
        UKFCore::UKFSmoother<1, 1> us(m);
        UKFCore::SRUKFSmoother<1, 1> ss(m);
        us.initialize(v1(0.0f), v1(1e-6f));
        ss.initialize(v1(0.0f), v1(1e-6f));
        const float truth[] = {0.0f, 0.0f, 1.0f, 3.0f, 6.0f};
        for (int k = 1; k <= 4; ++k) {
            const float tp = static_cast<float>(k - 1), tk = static_cast<float>(k);
            us.step(tp, tk, v1(truth[k]), v1(0.0f));
            ss.step(tp, tk, v1(truth[k]), v1(0.0f));
        }
        us.smooth(); ss.smooth();
        float worst = 0.0f;
        for (int k = 0; k <= 4; ++k) {
            worst = std::max({worst, std::abs(us.filtered_state(k)(0) - truth[k]),
                              std::abs(ss.filtered_state(k)(0) - truth[k]),
                              std::abs(us.smoothed_state(k)(0) - truth[k]),
                              std::abs(ss.smoothed_state(k)(0) - truth[k])});
        }
        std::printf("smoother time convention: worst |err| %.3e\n", worst);
        NLF_CHECK(worst < 1e-2f, "full smoothers track a time-varying f with the t_prev convention");
        bool threw = false;
        try { (void)us.smoothed_state(99); } catch (const std::out_of_range&) { threw = true; }
        NLF_CHECK(threw, "smoothed_state() is bounds-checked");
    }

    // ---- 6. Fixed-lag smoother re-initialisation --------------------------
    {
        AddTime1D m;
        UKFCore::UnscentedFixedLagSmoother<1, 1> u(m, 3);
        UKFCore::SRUKFFixedLagSmoother<1, 1> s(m, 3);
        u.initialize(v1(0.0f), v1(1.0f));
        s.initialize(v1(0.0f), v1(1.0f));
        for (int k = 1; k <= 6; ++k) {
            u.step(k - 1.0f, static_cast<float>(k), v1(5.0f * k), v1(0.0f));
            s.step(k - 1.0f, static_cast<float>(k), v1(5.0f * k), v1(0.0f));
        }
        u.initialize(v1(-7.0f), v1(1.0f));
        s.initialize(v1(-7.0f), v1(1.0f));
        NLF_CHECK(u.get_smoothed_state(0)(0) == -7.0f, "UKF fixed-lag re-init clears the smoothed buffer");
        NLF_CHECK(s.get_smoothed_state(0)(0) == -7.0f, "SRUKF fixed-lag re-init clears the smoothed buffer");
    }

    // ---- 7. Non-finite model output: roll back and throw ------------------
    {
        Poison1D m;
        UKFCore::UnscentedFixedLagSmoother<1, 1> u(m, 2);
        UKFCore::SRUKFFixedLagSmoother<1, 1> s(m, 2);
        UKFCore::SRUKFSmoother<1, 1> f(m);
        u.initialize(v1(1.0f), v1(1.0f));
        s.initialize(v1(1.0f), v1(1.0f));
        f.initialize(v1(1.0f), v1(1.0f));
        u.step(0.0f, 1.0f, v1(1.0f), v1(0.0f));
        s.step(0.0f, 1.0f, v1(1.0f), v1(0.0f));
        f.step(0.0f, 1.0f, v1(1.0f), v1(0.0f));
        m.armed = true;
        int threw = 0;
        try { u.step(1.0f, 2.0f, v1(1.0f), v1(0.0f)); } catch (const std::runtime_error&) { ++threw; }
        try { s.step(1.0f, 2.0f, v1(1.0f), v1(0.0f)); } catch (const std::runtime_error&) { ++threw; }
        try { f.step(1.0f, 2.0f, v1(1.0f), v1(0.0f)); } catch (const std::runtime_error&) { ++threw; }
        NLF_CHECK(threw == 3, "non-finite model output throws in all three smoothers");
        NLF_CHECK(u.get_filtered_state().allFinite() && s.get_filtered_state().allFinite() &&
                  f.final_filtered_state().allFinite(), "the filter is restored after the throw");
        m.armed = false;
        u.step(1.0f, 2.0f, v1(1.0f), v1(0.0f));   // and keeps working
        NLF_CHECK(u.get_filtered_state().allFinite(), "the smoother continues after recovery");
    }

    // ---- 8. SRUKF keeps a caller-set kappa (incl. negative) ---------------
    {
        Square1D m;
        UKFCore::SRUKF<1, 1> sr(m);
        sr.alpha = 0.5f;
        sr.kappa = -0.5f;
        sr.initialize(v1(0.0f), v1(1.0f));
        NLF_CHECK(sr.alpha == 0.5f && sr.kappa == -0.5f, "SRUKF::initialize keeps user alpha/kappa");
        UKFCore::SRUKF<1, 1> d(m);
        d.initialize(v1(0.0f), v1(1.0f));
        NLF_CHECK(d.kappa == 2.0f && d.alpha == 1.0f, "SRUKF default kappa is 3 - n");
    }

    std::printf("test_audit_regressions: all checks passed\n");
    return 0;
}
