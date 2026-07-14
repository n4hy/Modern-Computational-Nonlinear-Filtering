// =============================================================================
// test_ukf_smoother.cpp -- regression test for the FULL-INTERVAL unscented RTS
// smoother (UKFSmoother) and its ITERATED mode.  Plain-covariance sibling of
// test_srukf_smoother.cpp; same moving target + mildly nonlinear observation.
// =============================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "StateSpaceModel.h"
#include "UKF.h"
#include "UKFSmoother.h"

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } } while (0)

// State [pos, vel]; the time argument is the per-step dt.
class CVNonlinRange : public UKFModel::StateSpaceModel<2, 1> {
public:
    static constexpr float C = 0.004f;
    State f(const State& x, float dt, const Eigen::Ref<const State>&) const override {
        State o; o(0) = x(0) + x(1) * dt; o(1) = x(1); return o;
    }
    Observation h(const State& x, float) const override {
        Observation y; y(0) = x(0) * (1.0f + C * x(0)); return y;
    }
    StateMat Q(float) const override {
        StateMat q = StateMat::Zero(); q(0,0) = 1e-3f; q(1,1) = 1e-4f; return q;
    }
    ObsMat R(float) const override { ObsMat r; r(0,0) = 4.0f; return r; }
};

int main() {
    using M = CVNonlinRange;
    using State = M::State; using Obs = M::Observation; using StateMat = M::StateMat;

    const int N = 40; const float dt = 1.0f, pos0 = -40.0f, vel = 2.0f;
    M truth_model;
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 2.0f);
    std::vector<float> truth_pos(N + 1);
    std::vector<Obs> ys(N + 1);
    for (int k = 0; k <= N; ++k) {
        truth_pos[k] = pos0 + vel * (k * dt);
        State xt; xt << truth_pos[k], vel;
        ys[k] = truth_model.h(xt, 0.0f); ys[k](0) += noise(rng);
    }

    State x0; x0 << pos0 + 20.0f, vel + 1.0f;
    StateMat P0; P0 << 400.0f, 0.0f, 0.0f, 4.0f;
    State u = State::Zero();
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

    M model;
    UKFCore::UKFSmoother<2, 1> sm(model);
    sm.initialize(x0, P0);
    for (int k = 1; k <= N; ++k) sm.step(dt, ys[k], u);
    CHECK(sm.size() == N + 1, "history size");
    sm.smooth(0);

    double filt   = rmse([&](int k){ return sm.filtered_state(k)(0); });
    double smooth = rmse([&](int k){ return sm.smoothed_state(k)(0); });
    for (int k = 0; k <= N; ++k) {
        CHECK(std::isfinite(sm.smoothed_state(k)(0)) && sm.smoothed_covariance(k).allFinite(),
              "smoothed finite");
        CHECK(min_eig(sm.smoothed_covariance(k)) >= -1e-4f, "smoothed covariance PSD");
    }
    std::printf("filtered RMSE = %.3f m | smoothed RMSE = %.3f m\n", filt, smooth);
    CHECK(smooth < filt, "smoothed RMSE < filtered RMSE");
    // The ratio gate above still passes when both estimates degrade together, so
    // bound each in absolute terms. Clean run is 1.218 / 0.680 (seed 12345,
    // deterministic); ~20% headroom absorbs FP-association drift across flags.
    CHECK(filt   < 1.46, "filtered RMSE within absolute bound");
    CHECK(smooth < 0.82, "smoothed RMSE within absolute bound");

    int mid = N / 2;
    double fv = sm.filtered_covariance(mid).trace(), sv = sm.smoothed_covariance(mid).trace();
    std::printf("interior cov trace: filtered %.3f -> smoothed %.3f\n", fv, sv);
    CHECK(sv >= 0.0 && sv <= fv + 1e-4, "smoothed covariance in [0, filtered] at interior");

    UKFCore::UKFSmoother<2, 1> smit(model);
    smit.initialize(x0, P0);
    for (int k = 1; k <= N; ++k) smit.step(dt, ys[k], u);
    smit.smooth(3);
    double iter = rmse([&](int k){ return smit.smoothed_state(k)(0); });
    std::printf("iterated (x3) smoothed RMSE = %.3f m\n", iter);
    CHECK(iter <= smooth + 1e-3, "iterated smoother no worse than single-pass");

    std::printf("PASS: UKF full-interval RTS smoothing + iteration\n");
    return 0;
}
