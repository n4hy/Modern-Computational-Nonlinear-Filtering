#ifndef NLF_AFFINE_RTS_H
#define NLF_AFFINE_RTS_H

// =============================================================================
// affine_rts::smooth -- exact Kalman filter + Rauch-Tung-Striebel smoother for
// an AFFINE-Gaussian state-space model, in double precision.
// -----------------------------------------------------------------------------
// This is the inner solve of every iterated smoother in the library:
//
//   * EKFSmoother::smooth(n>0)      -- Gauss-Newton iterated EKS (IEKS):
//       the model is linearised by first-order Taylor expansion about the
//       previous smoothed means.
//   * UKFSmoother / SRUKFSmoother::smooth(n>0) -- iterated posterior
//       linearisation smoother (IPLS): the model is linearised by sigma-point
//       statistical linear regression w.r.t. the previous smoothed posteriors.
//
// Model, written about linearisation points (centred form, so angular
// components can be wrapped):
//
//   x_0       ~ N(m0, P0)
//   x_{j+1}   = f0_j + A_j (x_j - xlin_j)          + w_j,  w_j ~ N(0, Q_j)
//   y_{j+1}   = h0_j + C_j (x_{j+1} - zlin_j)      + v_j,  v_j ~ N(0, R_j)
//   y_0 (opt) = h0   + C   (x_0 - zlin)             + v,    v   ~ N(0, R)
//
// Every iteration starts from the ORIGINAL prior N(m0, P0) and every
// measurement is used exactly once; the result is the exact Gaussian posterior
// of this affine model (derived: Kalman 1960, Rauch-Tung-Striebel 1965). The
// iterated smoothers it replaced instead re-ran the nonlinear filter from the
// smoothed x_0 with the prior covariance P0, which counts the data twice
// (they moved away from the exact RTS answer on a LINEAR model).
//
// Angular components (ang_x / ang_y flags) are differenced with
// std::remainder(., 2 pi) and stored wrapped to [-pi, pi].
// =============================================================================

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

namespace affine_rts {

struct Dynamics {               // x_{j+1} = f0 + A (x_j - xlin) + w,  w ~ N(0, Q)
    Eigen::VectorXd xlin, f0;
    Eigen::MatrixXd A, Q;
};

struct Measurement {            // y = h0 + C (x - zlin) + v,  v ~ N(0, R)
    bool active = false;        // false: missed / non-finite measurement
    Eigen::VectorXd y, zlin, h0;
    Eigen::MatrixXd C, R;
};

struct Problem {
    Eigen::VectorXd m0;
    Eigen::MatrixXd P0;
    Measurement obs0;                   // optional measurement of x_0
    std::vector<Dynamics> dyn;          // dyn[j]: x_j -> x_{j+1}
    std::vector<Measurement> obs;       // obs[j]: measures x_{j+1}
    std::vector<bool> ang_x, ang_y;     // empty = no angular components
};

struct Result {
    std::vector<Eigen::VectorXd> m;     // smoothed means, size dyn.size() + 1
    std::vector<Eigen::MatrixXd> P;     // smoothed covariances
};

namespace detail {

inline double wrap_pi(double a) { return std::remainder(a, 2.0 * std::numbers::pi); }

inline Eigen::VectorXd diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
                            const std::vector<bool>& ang) {
    Eigen::VectorXd d = a - b;
    for (Eigen::Index i = 0; i < d.size() && static_cast<size_t>(i) < ang.size(); ++i)
        if (ang[static_cast<size_t>(i)]) d(i) = wrap_pi(d(i));
    return d;
}

inline void wrap(Eigen::VectorXd& x, const std::vector<bool>& ang) {
    for (Eigen::Index i = 0; i < x.size() && static_cast<size_t>(i) < ang.size(); ++i)
        if (ang[static_cast<size_t>(i)]) x(i) = wrap_pi(x(i));
}

inline Eigen::MatrixXd sym(const Eigen::MatrixXd& M) { return 0.5 * (M + M.transpose()); }

// Kalman update in place (Joseph form). Skips an inactive measurement.
inline void update(Eigen::VectorXd& m, Eigen::MatrixXd& P, const Measurement& z,
                   const std::vector<bool>& ang_x, const std::vector<bool>& ang_y) {
    if (!z.active) return;
    const Eigen::VectorXd yhat = z.h0 + z.C * diff(m, z.zlin, ang_x);
    const Eigen::VectorXd innov = diff(z.y, yhat, ang_y);
    const Eigen::MatrixXd S = sym(z.C * P * z.C.transpose() + z.R);
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error("affine_rts: innovation covariance factorisation failed");
    const Eigen::MatrixXd K = ldlt.solve(z.C * P).transpose();   // P C^T S^{-1}
    m += K * innov;
    wrap(m, ang_x);
    const Eigen::MatrixXd IKC = Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * z.C;
    P = sym(IKC * P * IKC.transpose() + K * z.R * K.transpose());
}

}  // namespace detail

inline Result smooth(const Problem& pr) {
    using namespace detail;
    const size_t T = pr.dyn.size();
    if (pr.obs.size() != T)
        throw std::invalid_argument("affine_rts: dyn and obs must have the same length");

    std::vector<Eigen::VectorXd> mf(T + 1), mp(T + 1);
    std::vector<Eigen::MatrixXd> Pf(T + 1), Pp(T + 1);

    // Forward Kalman filter from the ORIGINAL prior.
    mf[0] = pr.m0;
    Pf[0] = pr.P0;
    update(mf[0], Pf[0], pr.obs0, pr.ang_x, pr.ang_y);
    for (size_t j = 0; j < T; ++j) {
        const Dynamics& d = pr.dyn[j];
        mp[j + 1] = d.f0 + d.A * diff(mf[j], d.xlin, pr.ang_x);
        wrap(mp[j + 1], pr.ang_x);
        Pp[j + 1] = sym(d.A * Pf[j] * d.A.transpose() + d.Q);
        mf[j + 1] = mp[j + 1];
        Pf[j + 1] = Pp[j + 1];
        update(mf[j + 1], Pf[j + 1], pr.obs[j], pr.ang_x, pr.ang_y);
    }

    // Backward RTS:  G_j = P_j A_j^T (P^-_{j+1})^{-1}.
    Result r;
    r.m = mf;
    r.P = Pf;
    for (size_t jj = T; jj-- > 0;) {
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(Pp[jj + 1]);
        if (ldlt.info() != Eigen::Success)
            throw std::runtime_error("affine_rts: predicted covariance factorisation failed");
        const Eigen::MatrixXd G = ldlt.solve(pr.dyn[jj].A * Pf[jj]).transpose();
        r.m[jj] = mf[jj] + G * diff(r.m[jj + 1], mp[jj + 1], pr.ang_x);
        wrap(r.m[jj], pr.ang_x);
        r.P[jj] = sym(Pf[jj] + G * (r.P[jj + 1] - Pp[jj + 1]) * G.transpose());
    }
    return r;
}

}  // namespace affine_rts

#endif  // NLF_AFFINE_RTS_H
