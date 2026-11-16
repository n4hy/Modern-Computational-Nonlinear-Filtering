// Regression test for the RBPF fixed-lag smoothed LINEAR mean (2026-11-16).
//
//  A. Deterministic nonlinear path (x_nl_t = t, so every particle carries the
//     same path), linear part with path-dependent A, bias, H. Conditional on
//     the path the model is linear-Gaussian, so get_smoothed_mean(lag) must
//     equal the exact E[x_lin_{k-lag} | y_{1:k}] of an independent double
//     KF + RTS, for every k and every lag in [0, fixed_lag]. The replaced
//     code returned the ancestor's FILTERED mean E[x_lin_{k-lag} | y_{1:k-lag}].
//  B. Random nonlinear path driving H, with resampling: lag 0 equals the
//     filtered mean, and the lag-L smoothed linear mean tracks the truth at
//     k - L better than the filtered estimate made at k - L did.
// All checks are NLF_CHECK (survive NDEBUG).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "rbpf/rbpf_core.hpp"
#include "TestCheck.h"

namespace {

using T = rbpf::RbpfTypes<1, 2, 1>;   // 1 nonlinear, 2 linear [pos, vel], 1 observation

float a_of(float xnl) { return 0.95f + 0.04f * std::sin(xnl); }
float h_of(float xnl) { return 1.0f + 0.3f * std::cos(xnl); }

struct NL : rbpf::NonlinearModel<T> {
    float sigma = 0.0f;   // 0: deterministic x_nl_t = x_nl_{t-1} + 1
    NonlinearState propagate(const NonlinearState& x, float, const NonlinearState&,
                             std::mt19937_64& rng) const override {
        std::normal_distribution<float> d(0.0f, 1.0f);
        return NonlinearState::Constant(x(0) + 1.0f + (sigma > 0.0f ? sigma * d(rng) : 0.0f));
    }
    float log_proposal_density(const NonlinearState&, const NonlinearState&, float,
                               const NonlinearState&) const override { return 0.0f; }
};

// x_lin' = A(x_nl_prev) x_lin + bias(x_nl_prev) + w,   y = H(x_nl) x_lin + v
struct Lin : rbpf::ConditionalLinearGaussianModel<T> {
    void get_dynamics(const NonlinearState& xp, float, Eigen::Ref<LinearState> bias,
                      Eigen::Ref<Eigen::MatrixXf> A, Eigen::Ref<Eigen::MatrixXf> B,
                      Eigen::Ref<LinearCov> Q) const override {
        A << 1.0f, 1.0f, 0.0f, a_of(xp(0));
        bias << 0.0f, 0.05f * std::cos(xp(0));
        B.setZero();
        Q << 0.02f, 0.0f, 0.0f, 0.01f;
    }
    void get_observation(const NonlinearState& xc, float, Eigen::Ref<Observation> offset,
                         Eigen::Ref<Eigen::MatrixXf> H, Eigen::Ref<ObsCov> R) const override {
        offset.setZero();
        H << h_of(xc(0)), 0.0f;
        R << 1.0f;
    }
};

}  // namespace

int main() {
    const int K = 30, L = 5;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> n01(0.0, 1.0);
    const T::NonlinearState u = T::NonlinearState::Zero();
    const T::LinearState x0(0.0f, 1.0f);
    const T::LinearCov P0 = (T::LinearCov() << 4.0f, 0.0f, 0.0f, 1.0f).finished();

    // ---- A. exact on a deterministic path ----------------------------------
    {
        // Truth and data (x_nl_t = t).
        std::vector<Eigen::Vector2d> xt(K + 1);
        std::vector<double> y(K + 1, 0.0);
        xt[0] = Eigen::Vector2d(0.0, 1.0);
        for (int t = 1; t <= K; ++t) {
            const double ap = a_of(float(t - 1));
            xt[t] = Eigen::Vector2d(xt[t - 1](0) + xt[t - 1](1) + std::sqrt(0.02) * n01(rng),
                                    ap * xt[t - 1](1) + 0.05 * std::cos(double(t - 1)) + 0.1 * n01(rng));
            y[t] = h_of(float(t)) * xt[t](0) + n01(rng);
        }

        NL nl; Lin lin;
        rbpf::RbpfConfig cfg;
        cfg.num_particles = 64; cfg.fixed_lag = L; cfg.seed = 3;
        rbpf::RaoBlackwellizedParticleFilter<T, NL, Lin> f(nl, lin, cfg);
        f.initialize(T::NonlinearState::Zero(), x0, P0);

        // Independent double KF; the smoothed reference at k is an RTS pass
        // over epochs k-L..k of the filter run so far.
        std::vector<Eigen::Vector2d> mf(K + 1), mp(K + 1);
        std::vector<Eigen::Matrix2d> Pf(K + 1), Pp(K + 1), Am(K + 1);
        mf[0] = x0.cast<double>(); Pf[0] = P0.cast<double>();
        double worst = 0.0, worst_old = 0.0;
        for (int k = 1; k <= K; ++k) {
            f.step(float(k), T::Observation::Constant(float(y[k])), u);
            Am[k] << 1.0, 1.0, 0.0, double(a_of(float(k - 1)));
            const Eigen::Vector2d b(0.0, 0.05 * double(std::cos(float(k - 1))));
            mp[k] = Am[k] * mf[k - 1] + b;
            Pp[k] = Am[k] * Pf[k - 1] * Am[k].transpose() + Eigen::Vector2d(0.02, 0.01).asDiagonal().toDenseMatrix();
            const Eigen::RowVector2d H(double(h_of(float(k))), 0.0);
            const double S = H * Pp[k] * H.transpose() + 1.0;
            const Eigen::Vector2d Kg = Pp[k] * H.transpose() / S;
            mf[k] = mp[k] + Kg * (y[k] - H * mp[k]);
            Pf[k] = Pp[k] - Kg * S * Kg.transpose();
            if (k < L) continue;
            Eigen::Vector2d ms = mf[k];
            for (int lag = 0; lag <= L; ++lag) {
                const int t = k - lag;
                if (lag > 0) {
                    const Eigen::Matrix2d G = Pf[t] * Am[t + 1].transpose() * Pp[t + 1].inverse();
                    ms = mf[t] + G * (ms - mp[t + 1]);
                }
                T::NonlinearState snl; T::LinearState slin;
                f.get_smoothed_mean(lag, snl, slin);
                const double sd = std::sqrt(Pf[t](0, 0));
                worst = std::max(worst, (slin.cast<double>() - ms).cwiseAbs().maxCoeff() / sd);
                worst_old = std::max(worst_old, (mf[t] - ms).cwiseAbs().maxCoeff() / sd);
                NLF_REQUIRE(std::abs(snl(0) - float(t)) < 1e-4f, "nonlinear ancestor sits at epoch k - lag");
            }
        }
        std::printf("A. deterministic path: worst |smoothed - exact RTS| = %.2e sigma "
                    "(filtered-at-k-lag answer: %.2e sigma)\n", worst, worst_old);
        // Measured 1.35e-5 sigma (float KF in the filter vs double reference);
        // the replaced code (filtered mean at k - lag) was 2.25 sigma off.
        NLF_CHECK(worst < 1e-4, "RBPF smoothed linear mean = exact RTS on a fixed nonlinear path");
        NLF_CHECK(worst_old > 0.1, "test discriminates smoothed from filtered");
    }

    // ---- B. random nonlinear path with resampling --------------------------
    {
        double se_s = 0.0, se_f = 0.0;
        int count = 0;
        double lag0_diff = 0.0;
        for (int run = 0; run < 20; ++run) {
            NL nl; nl.sigma = 0.3f;
            Lin lin;
            rbpf::RbpfConfig cfg;
            cfg.num_particles = 400; cfg.fixed_lag = L; cfg.seed = 100 + run;
            rbpf::RaoBlackwellizedParticleFilter<T, NL, Lin> f(nl, lin, cfg);
            f.initialize(T::NonlinearState::Zero(), x0, P0);
            double xnl = 0.0;
            Eigen::Vector2d xl(0.0, 1.0);
            std::vector<Eigen::Vector2d> truth(K + 1);
            std::vector<T::LinearState> filt(K + 1);
            truth[0] = xl;
            for (int k = 1; k <= K; ++k) {
                const double ap = a_of(float(xnl));
                xl = Eigen::Vector2d(xl(0) + xl(1) + std::sqrt(0.02) * n01(rng),
                                     ap * xl(1) + 0.05 * std::cos(xnl) + 0.1 * n01(rng));
                xnl += 1.0 + 0.3 * n01(rng);
                truth[k] = xl;
                f.step(float(k), T::Observation::Constant(float(h_of(float(xnl)) * xl(0) + n01(rng))), u);
                T::NonlinearState fnl; f.get_filtered_mean(fnl, filt[k]);
                T::NonlinearState s0nl; T::LinearState s0lin;
                f.get_smoothed_mean(0, s0nl, s0lin);
                lag0_diff = std::max(lag0_diff, double((s0lin - filt[k]).cwiseAbs().maxCoeff()));
                if (k >= L + 5) {
                    T::NonlinearState snl; T::LinearState slin;
                    f.get_smoothed_mean(L, snl, slin);
                    se_s += (slin.cast<double>() - truth[k - L]).squaredNorm();
                    se_f += (filt[k - L].cast<double>() - truth[k - L]).squaredNorm();
                    ++count;
                }
            }
        }
        const double rs = std::sqrt(se_s / count), rf = std::sqrt(se_f / count);
        std::printf("B. random path: lag-0 vs filtered max diff %.2e; RMSE at k-%d: smoothed %.4f, filtered %.4f\n",
                    lag0_diff, L, rs, rf);
        NLF_CHECK(lag0_diff < 1e-5, "lag-0 smoothed mean equals the filtered mean");
        // Measured over 20 runs x 21 epochs: smoothed 0.446, filtered 0.764.
        NLF_CHECK(rs < rf, "lag-L smoothing beats the filtered estimate at the same epoch");
    }

    std::printf("test_rbpf_smoother: all checks passed\n");
    return 0;
}
