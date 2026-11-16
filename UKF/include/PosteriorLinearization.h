#ifndef UKF_POSTERIOR_LINEARIZATION_H
#define UKF_POSTERIOR_LINEARIZATION_H

// =============================================================================
// Iterated posterior linearisation smoother (IPLS) step, shared by
// UKFSmoother::smooth(n>0) and SRUKFSmoother::smooth(n>0).
// -----------------------------------------------------------------------------
// Garcia-Fernandez, Svensson, Sarkka, "Iterated posterior linearization
// smoother", IEEE Trans. Automatic Control 62(4), 2017.
//
// Given the current smoothed posteriors N(xbar_k, Pbar_k), k = 0..T:
//
//   1. Statistical linear regression (SLR) of f(., t_prev_j, u_j) w.r.t.
//      N(xbar_j, Pbar_j), computed with the filter's own sigma points:
//         zbar = sum Wm Z_i,  Psi = sum Wc dX_i dZ_i^T,  Phi = sum Wc dZ_i dZ_i^T,
//         Pxx  = sum Wc dX_i dX_i^T  (= Pbar_j up to rounding),
//         A    = Psi^T Pxx^{-1},  Omega = Phi - A Pxx A^T,
//      giving  f(x) ~ zbar + A (x - xbar_j) + e,  e ~ N(0, Omega).
//      Likewise h(., t_j) w.r.t. N(xbar_{j+1}, Pbar_{j+1}) -> (ybar, C, Lambda).
//   2. Solve the affine model with Q + Omega and R + Lambda EXACTLY
//      (affine_rts::smooth) from the ORIGINAL prior N(x0, P0), each
//      measurement once.
//
// Properties (derived): for affine f and h the unscented transform is exact,
// so A = F, Omega = 0, C = H, Lambda = 0 and every iteration returns the exact
// RTS posterior. Linearising f w.r.t. the FILTERED densities and h w.r.t. the
// PREDICTED densities instead reproduces the UKF + URTS pass (iteration 0):
// A P A^T + Omega = Phi, so the predicted and innovation covariances, the
// gain P^- C^T S^{-1} = Psi_y S^{-1} and the RTS gain P A^T (P^-)^{-1} all
// coincide with their sigma-point counterparts.
//
// Omega and Lambda are Schur complements of the sigma-point joint covariance.
// With Wc_0 < 0 (the usual scaled-UT choice) that joint matrix need not be
// PSD, so the Schur complement can have negative eigenvalues; they are
// clipped to zero (eigen-decomposition, double precision).
// =============================================================================

#include <cmath>
#include <numbers>
#include <optional>
#include <vector>
#include <Eigen/Dense>
#include "AffineRTS.h"
#include "SigmaPoints.h"

namespace UKFCore {
namespace ipls {

inline Eigen::MatrixXd psd_clip(const Eigen::MatrixXd& M) {
    const Eigen::MatrixXd S = 0.5 * (M + M.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) return Eigen::MatrixXd::Zero(S.rows(), S.cols());
    const Eigen::VectorXd ev = es.eigenvalues().cwiseMax(0.0);
    return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

// SLR of g w.r.t. the sigma points `sig` about the mean xbar. ROWS is the
// output dimension; ang(i) says whether output component i is an angle.
template<int NX, int ROWS, typename G, typename AngFn>
void slr(const SigmaPoints<NX>& sig, const Eigen::Matrix<float, NX, 1>& xbar, G&& g, AngFn&& ang,
         Eigen::VectorXd& gbar, Eigen::MatrixXd& A, Eigen::MatrixXd& Omega) {
    constexpr int NS = SigmaPoints<NX>::NSIG;
    Eigen::Matrix<double, ROWS, NS> Z;
    for (int i = 0; i < NS; ++i) Z.col(i) = g(Eigen::Matrix<float, NX, 1>(sig.X.col(i))).template cast<double>();

    const Eigen::VectorXd Wm = sig.Wm.template cast<double>();
    const Eigen::VectorXd Wc = sig.Wc.template cast<double>();
    gbar = Z * Wm;
    for (int r = 0; r < ROWS; ++r) {
        if (!ang(r)) continue;
        double s = 0.0, c = 0.0;
        for (int i = 0; i < NS; ++i) { s += Wm(i) * std::sin(Z(r, i)); c += Wm(i) * std::cos(Z(r, i)); }
        gbar(r) = std::atan2(s, c);
    }

    Eigen::Matrix<double, NX, NS> dX;
    Eigen::Matrix<double, ROWS, NS> dZ;
    for (int i = 0; i < NS; ++i) {
        dX.col(i) = (sig.X.col(i) - xbar).template cast<double>();   // true sigma offsets
        dZ.col(i) = Z.col(i) - gbar;
        for (int r = 0; r < ROWS; ++r)
            if (ang(r)) dZ(r, i) = std::remainder(dZ(r, i), 2.0 * std::numbers::pi);
    }
    const Eigen::MatrixXd Pxx = dX * Wc.asDiagonal() * dX.transpose();
    const Eigen::MatrixXd Psi = dX * Wc.asDiagonal() * dZ.transpose();     // NX x ROWS
    const Eigen::MatrixXd Phi = dZ * Wc.asDiagonal() * dZ.transpose();     // ROWS x ROWS
    // A = Psi^T Pxx^{-1}; LDLT's solve zeroes directions with a zero pivot,
    // so a singular Pbar (no spread in some direction) gives A = 0 there.
    A = Eigen::LDLT<Eigen::MatrixXd>(0.5 * (Pxx + Pxx.transpose())).solve(Psi).transpose();
    Omega = psd_clip(Phi - A * Pxx * A.transpose());
}

// One IPLS iteration. Model is a UKFModel::StateSpaceModel<NX, NY>.
// make_sigma(k, out) fills the sigma points of the current posterior at k
// (UKF: from (xbar_k, Pbar_k); SRUKF: directly from its Cholesky factor).
// Returns the new smoothed posteriors (double precision).
template<int NX, int NY, typename Model, typename MeasT, typename InitObsT, typename MakeSigma>
affine_rts::Result iterate(const Model& model,
                           const Eigen::Matrix<float, NX, 1>& x0, const Eigen::Matrix<float, NX, NX>& P0,
                           const std::optional<InitObsT>& init_obs, const std::vector<MeasT>& meas,
                           const std::vector<Eigen::Matrix<float, NX, 1>>& xbar, MakeSigma&& make_sigma) {
    using State = Eigen::Matrix<float, NX, 1>;
    const size_t T = meas.size();
    const auto ang_x = [&](int i) { return model.isAngularState(i); };
    const auto ang_y = [&](int i) { return model.isAngularObservation(i); };

    affine_rts::Problem pr;
    pr.m0 = x0.template cast<double>();
    pr.P0 = P0.template cast<double>();
    pr.ang_x.resize(NX);
    pr.ang_y.resize(NY);
    for (int i = 0; i < NX; ++i) pr.ang_x[static_cast<size_t>(i)] = model.isAngularState(i);
    for (int i = 0; i < NY; ++i) pr.ang_y[static_cast<size_t>(i)] = model.isAngularObservation(i);
    pr.dyn.resize(T);
    pr.obs.resize(T);

    SigmaPoints<NX> sig;
    const auto linearise_obs = [&](affine_rts::Measurement& z, size_t k, float t,
                                   const Eigen::Matrix<float, NY, 1>& y) {
        z.active = y.allFinite();                       // as UKF::update / SRUKF::update
        if (!z.active) return;
        make_sigma(k, sig);
        Eigen::MatrixXd Lambda;
        slr<NX, NY>(sig, xbar[k], [&](const State& x) { return model.h(x, t); }, ang_y,
                    z.h0, z.C, Lambda);
        z.y = y.template cast<double>();
        z.zlin = xbar[k].template cast<double>();
        z.R = model.R(t).template cast<double>() + Lambda;
    };

    if (init_obs) linearise_obs(pr.obs0, 0, init_obs->t, init_obs->y);
    for (size_t j = 0; j < T; ++j) {
        const MeasT& m = meas[j];
        affine_rts::Dynamics& d = pr.dyn[j];
        make_sigma(j, sig);
        Eigen::MatrixXd Omega;
        slr<NX, NX>(sig, xbar[j], [&](const State& x) { return model.f(x, m.t_prev, m.u); }, ang_x,
                    d.f0, d.A, Omega);
        d.xlin = xbar[j].template cast<double>();
        d.Q = model.Q(m.t_prev).template cast<double>() + Omega;
        linearise_obs(pr.obs[j], j + 1, m.t, m.y);
    }
    return affine_rts::smooth(pr);
}

}  // namespace ipls
}  // namespace UKFCore

#endif  // UKF_POSTERIOR_LINEARIZATION_H
