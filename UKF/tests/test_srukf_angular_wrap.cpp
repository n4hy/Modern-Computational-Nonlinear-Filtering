// =============================================================================
// test_srukf_angular_wrap.cpp -- regression test for the angular-OBSERVATION
// innovation wrap (R32) added to SRUKF::update().
// -----------------------------------------------------------------------------
// The IPNT production models (PNT solver, TLECorrect) currently declare no
// angular observation, so this defect is not exercised through their bindings.
// This standalone test installs a minimal model whose single observation IS an
// angle and verifies the wrap directly.
//
// Build (standalone, mirrors IPNT setup.py flags):
//   NLF=~/Modern-Computational-Nonlinear-Filtering
//   g++ -std=c++20 -O3 -I"$NLF/Common/include" -I"$NLF/UKF/include" \
//       -I"$NLF/build/_deps/optimizedkernels-src/include" -I/usr/include/eigen3 \
//       UKF/tests/test_srukf_angular_wrap.cpp \
//       -L"$NLF/build/_deps/optimizedkernels-build/src" -lOptMathKernels \
//       -o /tmp/test_srukf_angular_wrap && /tmp/test_srukf_angular_wrap
// =============================================================================
#include <cassert>
#include <cstdio>
#include <cmath>
#include <Eigen/Dense>
#include "StateSpaceModel.h"
#include "SRUKF.h"

// Minimal 1-state, 1-observation model whose observation is an ANGLE.
//   f: identity (a static phase);  h(x) = x (the angle itself).
// The observation is flagged angular, so SRUKF::update() must wrap the
// innovation (y_k - y_hat) into [-pi, pi] before forming the correction.
class AngleModel : public UKFModel::StateSpaceModel<1, 1> {
public:
    State f(const State& x, float, const Eigen::Ref<const State>&) const override { return x; }
    Observation h(const State& x, float) const override { return x; }
    StateMat Q(float) const override { return StateMat::Identity() * 1e-6f; }
    ObsMat   R(float) const override { return ObsMat::Identity() * 0.04f; }   // sigma ~0.2 rad
    bool isAngularObservation(int) const override { return true; }
};

int main() {
    AngleModel m;
    UKFCore::SRUKF<1, 1> f(m);

    AngleModel::State    x0;  x0  << 3.0f;    // prior phase ~ +pi
    AngleModel::StateMat P0;  P0  << 0.25f;   // sigma ~0.5 rad
    f.initialize(x0, P0);

    AngleModel::Observation y;  y << -3.0f;   // measurement ~ -pi
    // The TRUE circular innovation is (-3) - (+3) = -6 rad, which wraps to
    // +0.283 rad.  WITH the wrap the state moves the SHORT way and stays near
    // the +/-pi branch (|x_new| ~ 3.2).  WITHOUT the wrap the raw -6 rad
    // innovation drags the state the long way toward 0 (|x_new| < 1).
    f.update(0.0f, y);
    const float x_new = f.getState()(0);
    std::printf("x_new = %.4f   NIS = %.4f   gated = %u\n",
                x_new, f.getLastNIS(), f.getGatedCount());

    assert(std::fabs(x_new) > 2.5f &&
           "angular-OBSERVATION innovation was not wrapped at +/-pi (R32)");

    // R33: the rigorous NIS is exposed and finite/positive.
    assert(f.getLastNIS() >= 0.0f && std::isfinite(f.getLastNIS()) &&
           "getLastNIS() must expose a finite NIS (R33)");

    std::printf("PASS: angular-observation innovation wrapped (R32); "
                "NIS exposed (R33)\n");
    return 0;
}
