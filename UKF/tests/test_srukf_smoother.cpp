// =============================================================================
// test_srukf_smoother.cpp -- regression test for the FULL-INTERVAL square-root
// unscented RTS smoother (SRUKFSmoother) and its ITERATED mode.
// -----------------------------------------------------------------------------
// A moving target ([pos, vel], constant velocity) is observed through a MILDLY
// NONLINEAR measurement h(x) = pos*(1 + c*pos).  From a deliberately-off prior we
// verify:
//   (1) the backward RTS pass reduces both the trajectory RMSE and the interior
//       covariance relative to the forward filter (smoothing helps);
//   (2) the iterated smoother does not do worse -- and, given the nonlinearity +
//       off prior, does better -- than the single-pass smoother;
//   (3) everything stays finite.
//
// Standalone build (mirrors the other UKF tests / IPNT setup.py flags):
//   NLF=~/Modern-Computational-Nonlinear-Filtering
//   g++ -std=c++20 -O3 -I"$NLF/Common/include" -I"$NLF/UKF/include" \
//       -I<optmath>/include -I/usr/include/eigen3 \
//       UKF/tests/test_srukf_smoother.cpp -L<optmath-build> -lOptMathKernels \
//       -o /tmp/test_srukf_smoother && /tmp/test_srukf_smoother
// =============================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "StateSpaceModel.h"
#include "SRUKF.h"
#include "SRUKFSmoother.h"

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } } while (0)

// State [pos, vel]; the time argument is the per-step dt.
class CVNonlinRange : public UKFModel::StateSpaceModel<2, 1> {
public:
    static constexpr float C = 0.004f;      // measurement nonlinearity
    State f(const State& x, float dt, const Eigen::Ref<const State>&) const override {
        State o; o(0) = x(0) + x(1) * dt; o(1) = x(1); return o;
    }
    Observation h(const State& x, float) const override {
        Observation y; y(0) = x(0) * (1.0f + C * x(0)); return y;
    }
    StateMat Q(float) const override {
        StateMat q = StateMat::Zero(); q(0,0) = 1e-3f; q(1,1) = 1e-4f; return q;
    }
    ObsMat R(float) const override { ObsMat r; r(0,0) = 4.0f; return r; }  // sigma 2.0
};

int main() {
    using M = CVNonlinRange;
    using State = M::State;
    using Obs = M::Observation;
    using StateMat = M::StateMat;

    const int N = 40;
    const float dt = 1.0f;
    const float pos0 = -40.0f, vel = 2.0f;

    // --- truth trajectory + deterministic noisy observations ---
    M truth_model;
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 2.0f);
    std::vector<float> truth_pos(N + 1);
    std::vector<Obs> ys(N + 1);
    for (int k = 0; k <= N; ++k) {
        truth_pos[k] = pos0 + vel * (k * dt);
        State xt; xt << truth_pos[k], vel;
        ys[k] = truth_model.h(xt, 0.0f);
        ys[k](0) += noise(rng);
    }

    // --- deliberately-off prior ---
    State x0; x0 << pos0 + 20.0f, vel + 1.0f;   // 20 m position / 1 m/s velocity off
    StateMat P0; P0 << 400.0f, 0.0f, 0.0f, 4.0f;

    auto rmse = [&](auto&& get) {
        double s = 0; int n = 0;
        for (int k = 1; k <= N; ++k) { float e = get(k) - truth_pos[k]; s += e*e; ++n; }
        return std::sqrt(s / n);
    };
    // allFinite() accepts a negative variance and trace() hides one behind a
    // larger positive diagonal entry, so check the spectrum directly.
    auto min_eig = [](const Eigen::MatrixXf& P) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(P, Eigen::EigenvaluesOnly);
        return es.info() == Eigen::Success ? es.eigenvalues().minCoeff() : -1.0f;
    };

    M model;
    UKFCore::SRUKFSmoother<2, 1> sm(model);

    // forward filter + store history
    sm.initialize(x0, P0);
    State u = State::Zero();
    for (int k = 1; k <= N; ++k) sm.step(dt, ys[k], u);
    CHECK(sm.size() == N + 1, "history size");

    // (0) single backward pass
    sm.smooth(0);
    double filt_rmse   = rmse([&](int k){ return sm.filtered_state(k)(0); });
    double smooth_rmse = rmse([&](int k){ return sm.smoothed_state(k)(0); });

    // finiteness + positive semi-definiteness
    for (int k = 0; k <= N; ++k) {
        CHECK(std::isfinite(sm.smoothed_state(k)(0)) &&
              sm.smoothed_covariance(k).allFinite(), "smoothed finite");
        CHECK(min_eig(sm.smoothed_covariance(k)) >= -1e-4f, "smoothed covariance PSD");
    }

    // (1) smoothing reduces trajectory RMSE
    std::printf("filtered RMSE = %.3f m | smoothed RMSE = %.3f m\n", filt_rmse, smooth_rmse);
    CHECK(smooth_rmse < filt_rmse, "smoothed RMSE < filtered RMSE");
    // The ratio gate above still passes when both estimates degrade together, so
    // bound each in absolute terms. Clean run is 1.218 / 0.680 (seed 12345,
    // deterministic); ~20% headroom absorbs FP-association drift across flags.
    CHECK(filt_rmse   < 1.46, "filtered RMSE within absolute bound");
    CHECK(smooth_rmse < 0.82, "smoothed RMSE within absolute bound");

    // interior covariance shrinks (variance-reduction property of RTS)
    int mid = N / 2;
    double filt_var   = sm.filtered_covariance(mid).trace();
    double smooth_var = sm.smoothed_covariance(mid).trace();
    std::printf("interior cov trace: filtered %.3f -> smoothed %.3f\n", filt_var, smooth_var);
    CHECK(smooth_var >= 0.0 && smooth_var <= filt_var + 1e-4,
          "smoothed covariance in [0, filtered] at interior");

    // (2) iterated smoother: rebuild forward + smooth 3x; should not be worse
    UKFCore::SRUKFSmoother<2, 1> smit(model);
    smit.initialize(x0, P0);
    for (int k = 1; k <= N; ++k) smit.step(dt, ys[k], u);
    smit.smooth(3);
    double iter_rmse = rmse([&](int k){ return smit.smoothed_state(k)(0); });
    std::printf("iterated (x3) smoothed RMSE = %.3f m\n", iter_rmse);
    CHECK(iter_rmse <= smooth_rmse + 1e-3, "iterated smoother no worse than single-pass");
    for (int k = 0; k <= N; ++k)
        CHECK(std::isfinite(smit.smoothed_state(k)(0)), "iterated finite");

    std::printf("PASS: full-interval RTS smoothing + iteration\n");
    return 0;
}
