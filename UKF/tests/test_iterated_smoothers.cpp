// Regression tests for the iterated smoothers (2026-11-16):
//   EKFSmoother::smooth(n>0)   -- Gauss-Newton IEKS
//   UKFSmoother / SRUKFSmoother::smooth(n>0) -- iterated posterior linearisation (IPLS)
//
//  1. LINEAR model (with an initial observation and a missed measurement):
//     every iteration count must return the exact RTS posterior, computed
//     here by an independent double-precision KF + RTS. The replaced
//     iterations re-ran the filter from the smoothed x_0 with covariance P0
//     (data counted twice) and drifted away from it.
//  2. IEKS on a nonlinear model: the converged iterate must be a stationary
//     point of the MAP cost (derived property of a Gauss-Newton fixed point).
//  3. IPLS: UKF and SRUKF run the same algorithm -> same answer; iterations
//     converge; smooth(n) is idempotent and leaves the filtered pass alone.
//  4. Angular observation straddling +-pi: identical (up to the rotation) to
//     the same problem rotated away from the cut.
// All checks are NLF_CHECK (survive NDEBUG).

#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include <Eigen/Dense>

#include "EKF.h"
#include "EKFSmoother.h"
#include "SRUKFSmoother.h"
#include "StateSpaceModel.h"
#include "SystemModel.h"
#include "TestCheck.h"
#include "UKFSmoother.h"

namespace {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
constexpr float kDt = 1.0f;
constexpr double kPi = std::numbers::pi;

// ---------------------------------------------------------------- models ----
// Constant velocity, y = c * pos (linear when kNL == 0) or y = pos (1 + kNL pos).
struct LinCV : UKFModel::StateSpaceModel<2, 1> {
    float nl = 0.0f;
    State f(const State& x, float, const Eigen::Ref<const State>&) const override {
        return State(x(0) + kDt * x(1), x(1));
    }
    Observation h(const State& x, float) const override {
        return Observation::Constant(x(0) * (1.0f + nl * x(0)));
    }
    StateMat Q(float) const override { return (StateMat() << 0.05f, 0.0f, 0.0f, 0.01f).finished(); }
    ObsMat R(float) const override { return ObsMat::Constant(4.0f); }
};

struct EkfCV : SystemModel {
    float nl = 0.0f;
    Eigen::VectorXf f(const Eigen::VectorXf& x, const Eigen::VectorXf&, float) const override {
        Eigen::VectorXf o(2); o << x(0) + kDt * x(1), x(1); return o;
    }
    Eigen::VectorXf h(const Eigen::VectorXf& x, float) const override {
        Eigen::VectorXf y(1); y(0) = x(0) * (1.0f + nl * x(0)); return y;
    }
    Eigen::MatrixXf F(const Eigen::VectorXf&, const Eigen::VectorXf&, float) const override {
        Eigen::MatrixXf m(2, 2); m << 1.0f, kDt, 0.0f, 1.0f; return m;
    }
    Eigen::MatrixXf H(const Eigen::VectorXf& x, float) const override {
        Eigen::MatrixXf m(1, 2); m << 1.0f + 2.0f * nl * x(0), 0.0f; return m;
    }
    Eigen::MatrixXf Q(float) const override {
        Eigen::MatrixXf q = Eigen::MatrixXf::Zero(2, 2); q(0, 0) = 0.05f; q(1, 1) = 0.01f; return q;
    }
    Eigen::MatrixXf R(float) const override { return Eigen::MatrixXf::Constant(1, 1, 4.0f); }
    int getStateDim() const override { return 2; }
    int getObsDim() const override { return 1; }
};

// Angle + rate, y = angle (angular); linear apart from the wrap.
struct Bearing : UKFModel::StateSpaceModel<2, 1> {
    State f(const State& x, float, const Eigen::Ref<const State>&) const override {
        return State(std::remainder(x(0) + kDt * x(1), 2.0f * std::numbers::pi_v<float>), x(1));
    }
    Observation h(const State& x, float) const override { return Observation::Constant(x(0)); }
    StateMat Q(float) const override { return (StateMat() << 1e-3f, 0.0f, 0.0f, 1e-4f).finished(); }
    ObsMat R(float) const override { return ObsMat::Constant(4e-3f); }
    bool isAngularState(int i) const override { return i == 0; }
    bool isAngularObservation(int) const override { return true; }
};

// ------------------------------------------------- independent reference ----
// Double KF + RTS for x' = F x + w, y = c x + v; NaN y = missed. Index 0 is
// the prior (optionally updated by y0).
struct Ref { std::vector<Vec> m; std::vector<Mat> P; };
Ref exact_rts(const Vec& m0, const Mat& P0, double y0, const std::vector<double>& ys,
              const Mat& F, const Mat& Q, const Mat& c, double R) {
    const size_t T = ys.size();
    std::vector<Vec> mf(T + 1), mp(T + 1);
    std::vector<Mat> Pf(T + 1), Pp(T + 1);
    auto upd = [&](Vec& m, Mat& P, double y) {
        if (!std::isfinite(y)) return;
        const double S = (c * P * c.transpose())(0, 0) + R;
        const Vec K = P * c.transpose() / S;
        m += K * (y - (c * m)(0));
        P -= K * S * K.transpose();
    };
    mf[0] = m0; Pf[0] = P0; upd(mf[0], Pf[0], y0);
    for (size_t j = 0; j < T; ++j) {
        mp[j + 1] = F * mf[j]; Pp[j + 1] = F * Pf[j] * F.transpose() + Q;
        mf[j + 1] = mp[j + 1]; Pf[j + 1] = Pp[j + 1];
        upd(mf[j + 1], Pf[j + 1], ys[j]);
    }
    Ref r{mf, Pf};
    for (size_t j = T; j-- > 0;) {
        const Mat G = Pf[j] * F.transpose() * Pp[j + 1].inverse();
        r.m[j] = mf[j] + G * (r.m[j + 1] - mp[j + 1]);
        r.P[j] = Pf[j] + G * (r.P[j + 1] - Pp[j + 1]) * G.transpose();
    }
    return r;
}

// Worst |m - m_ref| in units of the reference smoothed sigma, and worst
// relative covariance error, over all epochs.
template<typename GetM, typename GetP>
std::pair<double, double> compare(const Ref& ref, GetM&& gm, GetP&& gp) {
    double em = 0, eP = 0;
    for (size_t k = 0; k < ref.m.size(); ++k) {
        const Vec m = gm(static_cast<int>(k));
        const Mat P = gp(static_cast<int>(k));
        for (int i = 0; i < 2; ++i) em = std::max(em, std::abs(m(i) - ref.m[k](i)) / std::sqrt(ref.P[k](i, i)));
        eP = std::max(eP, (P - ref.P[k]).norm() / ref.P[k].norm());
    }
    return {em, eP};
}

}  // namespace

int main() {
    const int T = 40;
    std::mt19937 rng(2026);
    std::normal_distribution<double> nz(0.0, 2.0);

    // ---- 1. Linear model: every iteration count = exact RTS ----------------
    {
        std::vector<double> ys(T);
        for (int k = 0; k < T; ++k) ys[k] = -40.0 + 2.0 * (k + 1) + nz(rng);
        ys[17] = std::numeric_limits<double>::quiet_NaN();     // missed measurement
        const double y0 = -40.0 + nz(rng);
        Vec m0(2); m0 << -20.0, 3.0;
        Mat P0(2, 2); P0 << 400.0, 0.0, 0.0, 4.0;
        Mat F(2, 2); F << 1.0, kDt, 0.0, 1.0;
        Mat Q(2, 2); Q << 0.05, 0.0, 0.0, 0.01;
        Mat c(1, 2); c << 1.0, 0.0;
        const Ref with_y0 = exact_rts(m0, P0, y0, ys, F, Q, c, 4.0);
        const Ref no_y0 = exact_rts(m0, P0, std::numeric_limits<double>::quiet_NaN(), ys, F, Q, c, 4.0);

        LinCV um;
        EkfCV em;
        const Eigen::Vector2f x0f = m0.cast<float>();
        const Eigen::Matrix2f P0f = P0.cast<float>();
        const Eigen::Vector2f u = Eigen::Vector2f::Zero();
        for (int n : {0, 1, 3, 10}) {
            UKFCore::UKFSmoother<2, 1> us(um);
            UKFCore::SRUKFSmoother<2, 1> ss(um);
            EKFSmoother es(&em);
            us.initialize(x0f, P0f); ss.initialize(x0f, P0f); es.initialize(x0f, P0f);
            us.observe_initial(0.0f, Eigen::Matrix<float, 1, 1>::Constant(static_cast<float>(y0)));
            ss.observe_initial(0.0f, Eigen::Matrix<float, 1, 1>::Constant(static_cast<float>(y0)));
            for (int k = 0; k < T; ++k) {
                const Eigen::Matrix<float, 1, 1> y = Eigen::Matrix<float, 1, 1>::Constant(static_cast<float>(ys[k]));
                us.step(k * kDt, (k + 1) * kDt, y, u);
                ss.step(k * kDt, (k + 1) * kDt, y, u);
                es.step(y.cast<float>(), Eigen::VectorXf::Zero(1), (k + 1) * kDt);
            }
            us.smooth(n); ss.smooth(n); es.smooth(n);
            const auto [eu, eUP] = compare(with_y0, [&](int k) { return Vec(us.smoothed_state(k).cast<double>()); },
                                           [&](int k) { return Mat(us.smoothed_covariance(k).cast<double>()); });
            const auto [es_, eSP] = compare(with_y0, [&](int k) { return Vec(ss.smoothed_state(k).cast<double>()); },
                                            [&](int k) { return Mat(ss.smoothed_covariance(k).cast<double>()); });
            const auto [ee, eEP] = compare(no_y0, [&](int k) { return Vec(es.smoothed_state(k).first.cast<double>()); },
                                           [&](int k) { return Mat(es.smoothed_state(k).second.cast<double>()); });
            std::printf("linear, %2d iterations: |dm|/sigma  UKF %.1e  SRUKF %.1e  EKF %.1e   "
                        "|dP|/|P|  %.1e %.1e %.1e\n", n, eu, es_, ee, eUP, eSP, eEP);
            // Float filters vs a double reference: measured <= 8.8e-6 sigma and
            // <= 9.4e-6 relative covariance for n = 0, 1, 3, 10. The replaced
            // iterations were already 2.9e-2 (UKF/SRUKF) / 3.1e-2 (EKF) sigma
            // off after ONE iteration on this problem.
            NLF_CHECK(eu < 1e-4 && es_ < 1e-4 && ee < 1e-4, "iterated smoother mean = exact RTS on a linear model");
            NLF_CHECK(eUP < 1e-4 && eSP < 1e-4 && eEP < 1e-4, "iterated smoother covariance = exact RTS on a linear model");
        }
    }

    // ---- Nonlinear data shared by 2 and 3 ----------------------------------
    const float nl = 0.02f;
    std::vector<float> ynl(T);
    for (int k = 0; k < T; ++k) {
        const double p = -30.0 + 1.5 * (k + 1);
        ynl[k] = static_cast<float>(p * (1.0 + nl * p) + nz(rng));
    }
    const Eigen::Vector2f x0n(-15.0f, 2.5f);
    const Eigen::Matrix2f P0n = (Eigen::Matrix2f() << 100.0f, 0.0f, 0.0f, 1.0f).finished();

    // ---- 2. IEKS fixed point = stationary point of the MAP cost ------------
    {
        EkfCV em; em.nl = nl;
        EKFSmoother es(&em);
        es.initialize(x0n, P0n);
        for (int k = 0; k < T; ++k)
            es.step(Eigen::VectorXf::Constant(1, ynl[k]), Eigen::VectorXf::Zero(1), (k + 1) * kDt);

        // Gradient of J = 1/2|x0-m0|^2_P0^-1 + 1/2 sum |x_k-f(x_{k-1})|^2_Q^-1
        //               + 1/2 sum |y_k-h(x_k)|^2_R^-1, in double, relative to
        // the largest single term (the scale float rounding acts on).
        auto grad_ratio = [&]() {
            std::vector<Vec> x(T + 1);
            for (int k = 0; k <= T; ++k) x[k] = es.smoothed_state(k).first.cast<double>();
            const Mat Qi = em.Q(0).cast<double>().inverse();
            const Mat P0i = P0n.cast<double>().inverse();
            Mat F(2, 2); F << 1.0, kDt, 0.0, 1.0;
            double gmax = 0, tmax = 0;
            for (int k = 0; k <= T; ++k) {
                std::vector<Vec> terms;
                if (k == 0) terms.push_back(P0i * (x[0] - x0n.cast<double>()));
                if (k >= 1) terms.push_back(Qi * (x[k] - F * x[k - 1]));
                if (k < T) terms.push_back(-F.transpose() * Qi * (x[k + 1] - F * x[k]));
                if (k >= 1) {
                    const double p = x[k](0);
                    Vec Hk(2); Hk << 1.0 + 2.0 * nl * p, 0.0;
                    terms.push_back(-Hk * (ynl[k - 1] - p * (1.0 + nl * p)) / 4.0);
                }
                Vec g = Vec::Zero(2);
                for (const Vec& t : terms) { g += t; tmax = std::max(tmax, t.cwiseAbs().maxCoeff()); }
                gmax = std::max(gmax, g.cwiseAbs().maxCoeff());
            }
            return gmax / tmax;
        };
        es.smooth(0);
        const double g0 = grad_ratio();
        es.smooth(30);
        const double g30 = grad_ratio();
        std::printf("IEKS MAP gradient / largest term: EKS %.2e -> 30 iterations %.2e\n", g0, g30);
        // Measured 9.5e-2 for the EKS (not a MAP stationary point for a
        // nonlinear h) and 2.6e-5 after 30 iterations (float means).
        NLF_CHECK(g30 < 1e-3, "converged IEKS is a stationary point of the MAP cost");
        NLF_CHECK(g30 < 0.1 * g0, "IEKS iterations reduce the MAP gradient");
    }

    // ---- 3. IPLS: UKF == SRUKF, convergence, idempotency -------------------
    {
        LinCV um; um.nl = nl;
        UKFCore::UKFSmoother<2, 1> us(um);
        UKFCore::SRUKFSmoother<2, 1> ss(um);
        us.initialize(x0n, P0n); ss.initialize(x0n, P0n);
        const Eigen::Vector2f u = Eigen::Vector2f::Zero();
        for (int k = 0; k < T; ++k) {
            const auto y = Eigen::Matrix<float, 1, 1>::Constant(ynl[k]);
            us.step(k * kDt, (k + 1) * kDt, y, u);
            ss.step(k * kDt, (k + 1) * kDt, y, u);
        }
        std::vector<Eigen::Vector2f> filt(T + 1);
        for (int k = 0; k <= T; ++k) filt[k] = us.filtered_state(k);

        auto snap = [&](auto& sm) { std::vector<Eigen::Vector2f> v(T + 1); for (int k = 0; k <= T; ++k) v[k] = sm.smoothed_state(k); return v; };
        auto dist = [&](const std::vector<Eigen::Vector2f>& a, const std::vector<Eigen::Vector2f>& b) {
            double d = 0;
            for (int k = 0; k <= T; ++k)
                for (int i = 0; i < 2; ++i)
                    d = std::max(d, std::abs(double(a[k](i)) - b[k](i)) / std::sqrt(double(us.smoothed_covariance(k)(i, i))));
            return d;
        };
        us.smooth(1); const auto u1 = snap(us);
        us.smooth(2); const auto u2 = snap(us);
        us.smooth(20); const auto u20 = snap(us);
        us.smooth(21); const auto u21 = snap(us);
        ss.smooth(20); const auto s20 = snap(ss);
        us.smooth(20); const auto u20b = snap(us);
        const double d12 = dist(u1, u2), d2021 = dist(u20, u21), dus = dist(u20, s20);
        std::printf("IPLS: step 1->2 %.2e sigma, 20->21 %.2e sigma, UKF vs SRUKF %.2e sigma\n", d12, d2021, dus);
        // Measured: 2.6e-2 sigma (1->2), 4.7e-6 sigma (20->21), UKF vs SRUKF
        // 2.2e-4 sigma (their iteration-0 passes differ by float rounding).
        NLF_CHECK(d2021 < 1e-3, "IPLS iterations converge");
        NLF_CHECK(d2021 < d12, "IPLS iteration changes shrink");
        NLF_CHECK(dus < 1e-3, "UKF and SRUKF IPLS agree");
        NLF_CHECK(dist(u20, u20b) == 0.0, "smooth(n) is idempotent");
        bool filt_same = true;
        for (int k = 0; k <= T; ++k) filt_same = filt_same && (us.filtered_state(k) == filt[k]);
        NLF_CHECK(filt_same, "smooth(n) leaves the forward filtered pass unchanged");
    }

    // ---- 4. Angular wrap: straddling +-pi == rotated away from it -----------
    {
        Bearing bm;
        std::normal_distribution<double> nb(0.0, std::sqrt(4e-3));
        std::vector<double> yb(T);
        for (int k = 0; k < T; ++k) yb[k] = kPi - 0.2 + 0.01 * (k + 1) + nb(rng);   // crosses +pi at k ~ 20
        const double rot = -kPi / 2.0;
        auto run = [&](auto& sm, double offset, int n) {
            sm.initialize(Eigen::Vector2f(static_cast<float>(std::remainder(kPi - 0.2 + offset, 2 * kPi)), 0.0f),
                          (Eigen::Matrix2f() << 0.05f, 0.0f, 0.0f, 1e-3f).finished());
            for (int k = 0; k < T; ++k)
                sm.step(k * kDt, (k + 1) * kDt,
                        Eigen::Matrix<float, 1, 1>::Constant(static_cast<float>(std::remainder(yb[k] + offset, 2 * kPi))),
                        Eigen::Vector2f::Zero());
            sm.smooth(n);
        };
        double worst = 0;
        for (int n : {0, 3}) {
            UKFCore::UKFSmoother<2, 1> a(bm), b(bm);
            UKFCore::SRUKFSmoother<2, 1> c(bm), d(bm);
            run(a, 0.0, n); run(b, rot, n); run(c, 0.0, n); run(d, rot, n);
            for (int k = 0; k <= T; ++k) {
                const double du = std::remainder(double(a.smoothed_state(k)(0)) + rot - b.smoothed_state(k)(0), 2 * kPi);
                const double ds = std::remainder(double(c.smoothed_state(k)(0)) + rot - d.smoothed_state(k)(0), 2 * kPi);
                const double sig = std::sqrt(double(b.smoothed_covariance(k)(0, 0)));
                worst = std::max({worst, std::abs(du) / sig, std::abs(ds) / sig,
                                  std::abs(double(a.smoothed_covariance(k)(0, 0)) - b.smoothed_covariance(k)(0, 0)) / sig / sig});
            }
        }
        std::printf("angular: straddling vs rotated, worst %.2e (sigma units)\n", worst);
        // Measured 1.25e-5. Before the URTS backward pass got circular means
        // (2026-11-16) the UKF smoother was 1.4 sigma off here at n = 0.
        NLF_CHECK(worst < 1e-3, "IPLS/URTS invariant to rotating an angular state across +-pi");
    }

    std::printf("test_iterated_smoothers: all checks passed\n");
    return 0;
}
