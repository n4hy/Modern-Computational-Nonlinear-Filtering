// =============================================================================
// test_ekf_smoother.cpp -- regression test for the FULL-INTERVAL EKF RTS smoother
// (EKFSmoother) and its iterated (IEKS) mode.  Same moving target + mildly
// nonlinear observation as the UKF/SRUKF smoother tests, with analytic Jacobians.
// =============================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "SystemModel.h"
#include "EKF.h"
#include "EKFSmoother.h"

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } } while (0)

// State [pos, vel]; the time argument is the per-step dt.  h(x)=pos*(1+C*pos).
class EKFCVModel : public SystemModel {
public:
    static constexpr float C = 0.004f;
    Eigen::VectorXf f(const Eigen::VectorXf& x, const Eigen::VectorXf&, float dt) const override {
        Eigen::VectorXf o(2); o(0) = x(0) + x(1) * dt; o(1) = x(1); return o;
    }
    Eigen::VectorXf h(const Eigen::VectorXf& x, float) const override {
        Eigen::VectorXf y(1); y(0) = x(0) * (1.0f + C * x(0)); return y;
    }
    Eigen::MatrixXf F(const Eigen::VectorXf&, const Eigen::VectorXf&, float dt) const override {
        Eigen::MatrixXf Fm(2,2); Fm << 1.0f, dt, 0.0f, 1.0f; return Fm;
    }
    Eigen::MatrixXf H(const Eigen::VectorXf& x, float) const override {
        Eigen::MatrixXf Hm(1,2); Hm << 1.0f + 2.0f * C * x(0), 0.0f; return Hm;
    }
    Eigen::MatrixXf Q(float) const override {
        Eigen::MatrixXf q = Eigen::MatrixXf::Zero(2,2); q(0,0)=1e-3f; q(1,1)=1e-4f; return q;
    }
    Eigen::MatrixXf R(float) const override {
        Eigen::MatrixXf r(1,1); r(0,0)=4.0f; return r;
    }
    int getStateDim() const override { return 2; }
    int getObsDim() const override { return 1; }
};

int main() {
    const int N = 40; const float dt = 1.0f, pos0 = -40.0f, vel = 2.0f;
    EKFCVModel model;
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 2.0f);
    std::vector<float> truth_pos(N + 1);
    std::vector<Eigen::VectorXf> ys(N + 1);
    for (int k = 0; k <= N; ++k) {
        truth_pos[k] = pos0 + vel * (k * dt);
        Eigen::VectorXf xt(2); xt << truth_pos[k], vel;
        ys[k] = model.h(xt, 0.0f); ys[k](0) += noise(rng);
    }

    Eigen::VectorXf x0(2); x0 << pos0 + 20.0f, vel + 1.0f;
    Eigen::MatrixXf P0(2,2); P0 << 400.0f, 0.0f, 0.0f, 4.0f;
    Eigen::VectorXf u = Eigen::VectorXf::Zero(1);
    auto rmse = [&](auto&& get) {
        double s = 0; for (int k = 1; k <= N; ++k) { float e = get(k) - truth_pos[k]; s += e*e; }
        return std::sqrt(s / N);
    };
    // allFinite() accepts a negative variance and trace() hides one behind a
    // larger positive diagonal entry, so check the spectrum directly.
    auto min_eig = [](const Eigen::MatrixXf& P) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(P, Eigen::EigenvaluesOnly);
        return es.info() == Eigen::Success ? es.eigenvalues().minCoeff() : -1.0f;
    };

    EKFSmoother sm(&model);
    sm.initialize(x0, P0);
    for (int k = 1; k <= N; ++k) sm.step(ys[k], u, dt);
    CHECK(sm.size() == N + 1, "history size");
    sm.smooth(0);

    double filt   = rmse([&](int k){ return sm.filtered_state(k).first(0); });
    double smooth = rmse([&](int k){ return sm.smoothed_state(k).first(0); });
    for (int k = 0; k <= N; ++k) {
        auto s = sm.smoothed_state(k);
        CHECK(std::isfinite(s.first(0)) && s.second.allFinite(), "smoothed finite");
        CHECK(min_eig(s.second) >= -1e-4f, "smoothed covariance PSD");
    }
    std::printf("filtered RMSE = %.3f m | smoothed RMSE = %.3f m\n", filt, smooth);
    CHECK(smooth < filt, "smoothed RMSE < filtered RMSE");
    // The ratio gate above still passes when both estimates degrade together, so
    // bound each in absolute terms. Clean run is 1.207 / 0.717 (seed 12345,
    // deterministic); ~20% headroom absorbs FP-association drift across flags.
    CHECK(filt   < 1.45, "filtered RMSE within absolute bound");
    CHECK(smooth < 0.86, "smoothed RMSE within absolute bound");

    int mid = N / 2;
    double fv = sm.filtered_state(mid).second.trace();
    double sv = sm.smoothed_state(mid).second.trace();
    std::printf("interior cov trace: filtered %.3f -> smoothed %.3f\n", fv, sv);
    CHECK(sv >= 0.0 && sv <= fv + 1e-4, "smoothed covariance in [0, filtered] at interior");

    EKFSmoother smit(&model);
    smit.initialize(x0, P0);
    for (int k = 1; k <= N; ++k) smit.step(ys[k], u, dt);
    smit.smooth(3);
    double iter = rmse([&](int k){ return smit.smoothed_state(k).first(0); });
    std::printf("iterated (x3) smoothed RMSE = %.3f m\n", iter);
    CHECK(iter <= smooth + 1e-3, "iterated smoother no worse than single-pass");

    std::printf("PASS: EKF full-interval RTS smoothing + iteration\n");
    return 0;
}
