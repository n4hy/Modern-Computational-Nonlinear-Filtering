// SRUKF innovation-gate modes: down-scale (default) vs reject (setRejectOutliers(true)).
// CTest: SRUKF_GateModes. Every check is an NLF_CHECK (survives NDEBUG).
//
// Model: 2-D constant velocity, x = [px, vx, py, vy], dt = 1, y = (px, py) + v.
// f and h are linear, so the unscented transform is exact and the SRUKF update
// must equal the Kalman update with the gated gain s*K (derived):
//     x+ = x + s K nu,      P+ = P - (2s - s^2) K Pyy K^T,
// the Joseph form (I - sKH) P (I - sKH)^T + s^2 K R K^T reduced with
// K Pyy K^T = K H P. s = 1 below the gate, sqrt(T / NIS) above it in
// down-scale mode, 0 above it in reject mode.
//
//  A. One update, three regimes, against an independent double KF.
//  B. Reject mode: a rejected measurement is exactly a missed one (NaN y).
//  C. Monte Carlo with 5% gross outliers: accuracy and NEES per mode.
//  D. A genuine jump in the truth: steps until each mode re-acquires
//     (100 seeds x jumps of 10, 40, 200, 600 measurement sigmas), with and
//     without the reject-mode lock-out guard (setMaxConsecutiveRejections).
//  E. Lock-out guard semantics: rejections 1..N equal a missed measurement,
//     rejection N+1 equals a down-scaled update, a passing update or
//     initialize() ends the run, a missed measurement leaves it; a burst of
//     <= N outliers is handled bit-identically to the unguarded reject mode.
//
// C and D are empirical: thresholds sit well outside the measured values
// recorded next to them; they are not derived bounds.

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "SRUKF.h"
#include "StateSpaceModel.h"
#include "TestCheck.h"

namespace {

constexpr float kQ = 0.01f;   // white-acceleration intensity
constexpr float kR = 0.5f;    // measurement sigma per axis

struct CV2D : UKFModel::StateSpaceModel<4, 2> {
    State f(const State& x, float, const Eigen::Ref<const State>&) const override {
        return State(x(0) + x(1), x(1), x(2) + x(3), x(3));
    }
    Observation h(const State& x, float) const override { return Observation(x(0), x(2)); }
    StateMat Q(float) const override {
        StateMat Q = StateMat::Zero();
        const Eigen::Matrix2f b = (Eigen::Matrix2f() << 1.0f / 3, 0.5f, 0.5f, 1.0f).finished() * kQ;
        Q.block<2, 2>(0, 0) = b;
        Q.block<2, 2>(2, 2) = b;
        return Q;
    }
    ObsMat R(float) const override { return ObsMat::Identity() * (kR * kR); }
};

using SR = UKFCore::SRUKF<4, 2>;
using Vec4 = Eigen::Vector4d;
using Mat4 = Eigen::Matrix4d;

Mat4 F_d() {
    Mat4 F = Mat4::Identity();
    F(0, 1) = 1.0;
    F(2, 3) = 1.0;
    return F;
}

// Process noise drawn from the MODEL's Q (Cholesky factor), so the truth
// is exactly the model the filter assumes and NEES has mean 4 (derived).
// (A first version perturbed only the velocity, which is not N(0, Q):
// clean-data NEES read 4.71.)
Vec4 process_noise(std::mt19937_64& rng, std::normal_distribution<double>& n01) {
    static const Mat4 Lq = CV2D().Q(0.0f).cast<double>().llt().matrixL();
    Vec4 z;
    for (int i = 0; i < 4; ++i) z(i) = n01(rng);
    return Lq * z;
}

// Reject = reject mode with the default lock-out guard (N = 5);
// RejectPure = reject mode with the guard disabled (N = 0).
enum class Mode { None, DownScale, Reject, RejectPure };

void configure(SR& f, Mode m) {
    if (m == Mode::None) f.setInnovationGateChi2(std::numeric_limits<float>::infinity());
    if (m == Mode::Reject || m == Mode::RejectPure) f.setRejectOutliers(true);
    if (m == Mode::RejectPure) f.setMaxConsecutiveRejections(0);
}

}  // namespace

int main() {
    CV2D model;
    const Eigen::Matrix<double, 2, 4> H = (Eigen::Matrix<double, 2, 4>() << 1, 0, 0, 0, 0, 0, 1, 0).finished();
    const Eigen::Matrix2d Rd = Eigen::Matrix2d::Identity() * double(kR) * double(kR);

    // ---- A. single update vs exact gated KF --------------------------------
    {
        Eigen::Matrix4d L;
        L << 1.0, 0.0, 0.0, 0.0,
             0.3, 0.5, 0.0, 0.0,
             0.4, -0.2, 0.9, 0.0,
             -0.1, 0.2, 0.3, 0.4;
        const Mat4 P0 = L * L.transpose();
        const Vec4 x0(0.0, 1.0, 0.0, -1.0);
        const Eigen::Matrix2d Pyy = H * P0 * H.transpose() + Rd;
        const Eigen::Matrix<double, 4, 2> K = P0 * H.transpose() * Pyy.inverse();
        const Eigen::Vector2d e = Eigen::Vector2d(1.0, 0.3).normalized();
        const double T = double(SR(model).getInnovationGateChi2());   // 18.42 for NY = 2

        double worst = 0.0;
        for (double nis : {10.0, 40.0}) {
            const Eigen::Vector2d nu = std::sqrt(nis / e.dot(Pyy.ldlt().solve(e))) * e;
            const Eigen::Vector2d y = H * x0 + nu;
            for (Mode m : {Mode::DownScale, Mode::Reject}) {
                SR f(model);
                configure(f, m);
                f.initialize(x0.cast<float>(), P0.cast<float>());
                f.update(0.0f, y.cast<float>());

                const double s = nis <= T ? 1.0 : (m == Mode::Reject ? 0.0 : std::sqrt(T / nis));
                const Vec4 x_ref = x0 + s * K * nu;
                const Mat4 P_ref = P0 - (2.0 * s - s * s) * K * Pyy * K.transpose();
                // Independent Joseph form for the gain sK.
                const Mat4 IKH = Mat4::Identity() - s * K * H;
                const Mat4 P_jos = IKH * P0 * IKH.transpose() + s * s * K * Rd * K.transpose();
                NLF_CHECK((P_ref - P_jos).cwiseAbs().maxCoeff() < 1e-12, "reduced form == Joseph form");

                const double ex = (f.getState().cast<double>() - x_ref).cwiseAbs().maxCoeff();
                const double eP = (f.getCovariance().cast<double>() - P_ref).cwiseAbs().maxCoeff();
                worst = std::max({worst, ex, eP});
                NLF_CHECK(std::abs(f.getLastNIS() - nis) / nis < 1e-5, "NIS equals nu^T Pyy^-1 nu");
                NLF_CHECK(f.getGatedCount() == (nis > T ? 1u : 0u), "gated count");
                NLF_CHECK(ex < 1e-5 && eP < 1e-5, "SRUKF update = exact gated KF update");
                if (m == Mode::Reject && nis > T) {
                    NLF_CHECK(f.getState() == x0.cast<float>(), "rejected: state exactly unchanged");
                }
            }
        }
        std::printf("A. single update: worst |SRUKF - exact gated KF| = %.2e (threshold NY=2: %.4f)\n", worst, T);
    }

    // ---- B. rejection == missed measurement --------------------------------
    {
        std::mt19937_64 rng(5);
        std::normal_distribution<double> n01(0.0, 1.0);
        const int K = 200;
        SR fr(model), fm(model);
        fr.setRejectOutliers(true);
        fm.setRejectOutliers(true);
        const Eigen::Vector4f x0(0.0f, 1.0f, 0.0f, -1.0f);
        const Eigen::Matrix4f P0 = Eigen::Matrix4f::Identity();
        fr.initialize(x0, P0);
        fm.initialize(x0, P0);
        Vec4 x = x0.cast<double>();
        int outliers = 0;
        double worst = 0.0;
        for (int k = 1; k <= K; ++k) {
            x = F_d() * x;
            x += process_noise(rng, n01);
            Eigen::Vector2f y((x(0) + kR * n01(rng)), (x(2) + kR * n01(rng)));
            Eigen::Vector2f y_m = y;
            if (k % 20 == 7) {                      // 10 gross outliers
                y(0) += 50.0f;
                y_m.setConstant(std::numeric_limits<float>::quiet_NaN());
                ++outliers;
            }
            fr.predict(float(k - 1), Eigen::Vector4f::Zero());
            fm.predict(float(k - 1), Eigen::Vector4f::Zero());
            fr.update(float(k), y);
            fm.update(float(k), y_m);
            worst = std::max(worst, double((fr.getState() - fm.getState()).cwiseAbs().maxCoeff()));
            worst = std::max(worst, double((fr.getCovariance() - fm.getCovariance()).cwiseAbs().maxCoeff()));
        }
        std::printf("B. reject vs missed: max |difference| = %.2e over %d steps; gated %u vs %u (%d outliers)\n",
                    worst, K, fr.getGatedCount(), fm.getGatedCount(), outliers);
        NLF_CHECK(fr.getGatedCount() == fm.getGatedCount() + unsigned(outliers),
                  "every outlier is rejected, and nothing else differs");
        NLF_CHECK(worst == 0.0, "a rejected measurement leaves the filter exactly as a missed one");
    }

    // ---- C. Monte Carlo with 5% gross outliers -----------------------------
    {
        const int runs = 200, K = 200, burn = 10;
        const double p_out = 0.05, out_mag = 30.0 * kR;   // 30 sigma
        struct Acc { double se = 0.0, nees = 0.0; long n = 0; unsigned gated = 0; };
        // [contaminated?][mode]
        Acc acc[2][3];
        for (int run = 0; run < runs; ++run) {
            for (int contam = 0; contam < 2; ++contam) {
                std::mt19937_64 rng(1000 + run);            // same truth/noise for every mode
                std::normal_distribution<double> n01(0.0, 1.0);
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                std::vector<Vec4> xs(K + 1);
                std::vector<Eigen::Vector2f> ys(K + 1);
                xs[0] = Vec4(0.0, 1.0, 0.0, -1.0);
                for (int k = 1; k <= K; ++k) {
                    Vec4 x = F_d() * xs[k - 1];
                    x += process_noise(rng, n01);
                    xs[k] = x;
                    Eigen::Vector2d y(x(0) + kR * n01(rng), x(2) + kR * n01(rng));
                    const double u = u01(rng);
                    const double sgn = u01(rng) < 0.5 ? -1.0 : 1.0;
                    if (contam && u < p_out) y(u01(rng) < 0.5 ? 0 : 1) += sgn * out_mag;
                    ys[k] = y.cast<float>();
                }
                for (int mi = 0; mi < 3; ++mi) {
                    SR f(model);
                    configure(f, Mode(mi));
                    f.initialize(xs[0].cast<float>(), Eigen::Matrix4f::Identity() * 0.1f);
                    Acc& a = acc[contam][mi];
                    for (int k = 1; k <= K; ++k) {
                        f.predict(float(k - 1), Eigen::Vector4f::Zero());
                        f.update(float(k), ys[k]);
                        if (k < burn) continue;
                        const Vec4 e = f.getState().cast<double>() - xs[k];
                        a.se += e(0) * e(0) + e(2) * e(2);
                        a.nees += e.dot(f.getCovariance().cast<double>().ldlt().solve(e));
                        ++a.n;
                    }
                    a.gated += f.getGatedCount();
                }
            }
        }
        const char* names[3] = {"no gate", "down-scale", "reject"};
        double rmse[2][3], nees[2][3];
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < 3; ++m) {
                rmse[c][m] = std::sqrt(acc[c][m].se / double(acc[c][m].n));
                nees[c][m] = acc[c][m].nees / double(acc[c][m].n);
                std::printf("C. %-12s %-10s  pos RMSE %.4f   mean NEES %.3f (consistent: 4)   gated %u\n",
                            c ? "contaminated" : "clean", names[m], rmse[c][m], nees[c][m], acc[c][m].gated);
            }
        // Measured (GCC 13.3 / libstdc++, 200 runs x 191 scored steps):
        //   clean:        RMSE 0.4871 / 0.4871 / 0.4877, NEES 4.002 / 4.002 / 4.006,
        //                 8 gated of 40000 updates (4 expected at q = 1e-4)
        //   contaminated: RMSE 2.124 / 0.638 / 0.501, NEES 51.4 / 5.14 / 3.98,
        //                 gated 0 / 2136 / 1984 (about 2000 outliers injected)
        //   (no gate / down-scale / reject). The tolerances below were set
        //   after measuring; they are empirical checks, not derived bounds.
        NLF_CHECK(rmse[1][2] < rmse[1][1] && rmse[1][1] < rmse[1][0],
                  "with outliers: reject < down-scale < no gate (position RMSE)");
        NLF_CHECK(rmse[1][2] < 1.05 * rmse[0][2], "reject on contaminated data within 5% of clean");
        NLF_CHECK(std::abs(nees[1][2] - 4.0) < 0.4, "reject mode stays NEES-consistent with outliers");
        NLF_CHECK(std::abs(nees[0][0] - 4.0) < 0.2, "clean data, no gate: NEES-consistent (simulation sanity)");
        NLF_CHECK(std::abs(rmse[0][2] - rmse[0][1]) < 1e-2 * rmse[0][1],
                  "clean data: reject within 1% of down-scale (gate rarely fires)");
    }

    // ---- D. genuine jump: re-acquisition -----------------------------------
    // The truth's px jumps at k = 100 with no change to the model. In reject
    // mode every measurement is gated until the predicted covariance has grown
    // enough for the jump to pass the gate. Re-acquired = |px error| <
    // 3 sigma_px for 5 consecutive steps; median and maximum over 100 seeds;
    // lock-out = not re-acquired within 300 steps.
    //
    // Why the guard exists (measured 2026-11-18): WITHOUT it (RejectPure),
    // when the gate re-opens after a long coast the CV model explains the
    // accumulated position discrepancy as a velocity error, so one update
    // leaves the velocity many sigma wrong (seed 5002, jump 100: 2.8 vs
    // sigma 0.35 after 44 rejected steps); later measurements fail the gate
    // again and the track coasts away. Measured, RejectPure: lock-outs 0, 0,
    // 6, 33 of 100 at jumps 5, 20, 100, 300 (median 4, 20, 81, 227).
    // With the default guard (N = 5): 0 lock-outs at every jump, median 4,
    // 19, 41, 63, max 12, 24, 48, 73. Down-scale: max 9, 18, 39, 62.
    {
        const int K = 400, k_jump = 100, seeds = 100;
        const char* names[4] = {"no gate", "down-scale", "reject N=5", "reject N=0"};
        int worst_ok = 0;
        long pure_lockouts = 0;
        for (double jump : {5.0, 20.0, 100.0, 300.0}) {       // 10, 40, 200, 600 measurement sigmas
            std::vector<int> steps[4];
            for (int seed = 0; seed < seeds; ++seed) {
                for (int mi = 0; mi < 4; ++mi) {
                    std::mt19937_64 rng(5000 + seed);
                    std::normal_distribution<double> n01(0.0, 1.0);
                    SR f(model);
                    configure(f, Mode(mi));
                    Vec4 x(0.0, 1.0, 0.0, -1.0);
                    f.initialize(x.cast<float>(), Eigen::Matrix4f::Identity() * 0.1f);
                    int calm = 0, reacq = K;                    // K = never
                    for (int k = 1; k <= K && reacq == K; ++k) {
                        x = F_d() * x;
                        x += process_noise(rng, n01);
                        if (k == k_jump) x(0) += jump;
                        const Eigen::Vector2f y(float(x(0) + kR * n01(rng)), float(x(2) + kR * n01(rng)));
                        f.predict(float(k - 1), Eigen::Vector4f::Zero());
                        f.update(float(k), y);
                        if (k < k_jump) continue;
                        const double sd = std::sqrt(double(f.getCovariance()(0, 0)));
                        calm = std::abs(double(f.getState()(0)) - x(0)) < 3.0 * sd ? calm + 1 : 0;
                        if (calm == 5) reacq = k - 4 - k_jump;
                    }
                    steps[mi].push_back(reacq);
                }
            }
            std::printf("D. jump %3.0f:", jump);
            for (int mi = 0; mi < 4; ++mi) {
                std::sort(steps[mi].begin(), steps[mi].end());
                const int med = steps[mi][seeds / 2];
                const long never = std::count(steps[mi].begin(), steps[mi].end(), K);
                const int mx_ok = never < seeds ? *(steps[mi].end() - 1 - never) : -1;
                if (mi < 3) {
                    NLF_CHECK(never == 0, "no gate / down-scale / guarded reject always re-acquire");
                    worst_ok = std::max(worst_ok, mx_ok);
                } else {
                    pure_lockouts += never;
                }
                std::printf("  %s med %d max %d lock %ld;", names[mi], med, mx_ok, never);
            }
            std::printf("\n");
        }
        NLF_CHECK(worst_ok < 100, "asserted modes re-acquire within 100 steps");
        // Not a desired property: confirms this scenario still exercises the
        // failure the guard prevents (otherwise the checks above prove nothing).
        NLF_CHECK(pure_lockouts > 0, "unguarded reject mode locks out in this scenario");
    }

    // ---- E. lock-out guard semantics ---------------------------------------
    {
        const Eigen::Vector4f x0(0.0f, 1.0f, 0.0f, -1.0f);
        const Eigen::Matrix4f P0 = Eigen::Matrix4f::Identity();
        const unsigned N = 2;
        SR g(model);
        g.setRejectOutliers(true);
        g.setMaxConsecutiveRejections(N);
        NLF_CHECK(SR(model).getMaxConsecutiveRejections() == SR::kDefaultMaxConsecutiveRejections &&
                  SR::kDefaultMaxConsecutiveRejections == 5u, "default guard N = 5");
        g.initialize(x0, P0);
        const Eigen::Vector2f nan2 = Eigen::Vector2f::Constant(std::numeric_limits<float>::quiet_NaN());

        // One step of g against a reference filter started from g's predicted
        // state and run in `ref_mode` (bitwise comparison).
        auto step = [&](int k, Eigen::Vector2f dy, Mode ref_mode, unsigned run_after, const char* what) {
            g.predict(float(k - 1), Eigen::Vector4f::Zero());
            SR r(model);
            configure(r, ref_mode);
            r.initialize(x0, P0);
            r.restore(g.getState(), g.getSqrtCovariance());
            const Eigen::Vector2f y = dy.allFinite()
                ? Eigen::Vector2f(g.getState()(0) + dy(0), g.getState()(2) + dy(1)) : dy;
            g.update(float(k), y);
            r.update(float(k), y);
            NLF_REQUIRE(g.getState() == r.getState() && g.getSqrtCovariance() == r.getSqrtCovariance(), what);
            NLF_REQUIRE(g.getConsecutiveRejections() == run_after, "consecutive-rejection count");
        };
        const Eigen::Vector2f out(50.0f, 0.0f), ok(0.0f, 0.0f);
        step(1, out, Mode::RejectPure, 1, "rejection 1 = unguarded rejection");
        step(2, nan2, Mode::RejectPure, 1, "missed measurement leaves the run unchanged");
        step(3, out, Mode::RejectPure, 2, "rejection N = unguarded rejection");
        step(4, out, Mode::DownScale, 3, "gated update N+1 = down-scaled update");
        step(5, out, Mode::DownScale, 4, "gated update N+2 = down-scaled update");
        step(6, ok, Mode::RejectPure, 0, "passing update ends the run");
        step(7, out, Mode::RejectPure, 1, "after a pass: rejected again");
        g.initialize(x0, P0);
        NLF_CHECK(g.getConsecutiveRejections() == 0, "initialize() ends the run");

        // Burst of L consecutive 30-sigma outliers: L <= N (default 5) is
        // bit-identical to unguarded reject (derived: the run never exceeds
        // N, so the fallback never fires); L = N + 1 differs.
        auto burst_differs = [&](int L) {
            std::mt19937_64 rng(77);
            std::normal_distribution<double> n01(0.0, 1.0);
            SR a(model), b(model);
            configure(a, Mode::Reject);
            configure(b, Mode::RejectPure);
            a.initialize(x0, Eigen::Matrix4f::Identity() * 0.1f);
            b.initialize(x0, Eigen::Matrix4f::Identity() * 0.1f);
            Vec4 x = x0.cast<double>();
            bool differs = false;
            for (int k = 1; k <= 150; ++k) {
                x = F_d() * x;
                x += process_noise(rng, n01);
                Eigen::Vector2f y(float(x(0) + kR * n01(rng)), float(x(2) + kR * n01(rng)));
                if (k >= 100 && k < 100 + L) y(0) += 30.0f * kR;
                a.predict(float(k - 1), Eigen::Vector4f::Zero());
                b.predict(float(k - 1), Eigen::Vector4f::Zero());
                a.update(float(k), y);
                b.update(float(k), y);
                differs = differs || a.getState() != b.getState();
            }
            return differs;
        };
        const int Nd = int(SR::kDefaultMaxConsecutiveRejections);
        NLF_CHECK(!burst_differs(Nd), "burst of N outliers: guarded == unguarded reject, bit-identical");
        NLF_CHECK(burst_differs(Nd + 1), "burst of N+1 outliers: guard fires");
        std::printf("E. guard semantics: rejections 1..N = missed, N+1.. = down-scaled, reset on pass/initialize; "
                    "burst <= %d bit-identical to unguarded\n", Nd);
    }

    std::printf("test_srukf_gate_modes: all checks passed\n");
    return 0;
}
