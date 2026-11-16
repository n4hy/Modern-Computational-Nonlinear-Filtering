#ifndef EKF_SMOOTHER_H
#define EKF_SMOOTHER_H

// =============================================================================
// EKFSmoother -- FULL-INTERVAL (batch) Rauch-Tung-Striebel smoother for the EKF,
// with optional ITERATION.
// -----------------------------------------------------------------------------
// The full-interval counterpart of EKFFixedLag (same RTS math, using the stored
// state-transition Jacobian F for the smoothing gain), and the EKF sibling of
// UKFSmoother / SRUKFSmoother.  It keeps the ENTIRE forward trajectory and runs a
// single backward pass once all measurements are in, so every measurement informs
// every epoch -- the natural "revisit the whole capture after the pass" refinement.
//
// smooth(n_iterations>0) is the Gauss-Newton iterated EKS (IEKS; Bell 1994,
// "The iterated Kalman smoother as a Gauss-Newton method", SIAM J. Optim. 4).
// Each iteration relinearises f and h by their Jacobians about the previous
// smoothed means xbar_{0:T},
//     f(x) ~ f(xbar_j) + F(xbar_j) (x - xbar_j),   h(x) ~ h(xbar) + H(xbar) (x - xbar),
// and solves the resulting affine model exactly (affine_rts::smooth) from the
// ORIGINAL prior N(x0, P0), each measurement used once. That solve is one
// Gauss-Newton step on the MAP cost
//     J = |x_0-x0|^2_{P0^-1} + sum |x_k-f(x_{k-1})|^2_{Q^-1} + sum |y_k-h(x_k)|^2_{R^-1},
// so a fixed point is a stationary point of J (derived), and on a linear
// model every iteration returns the exact RTS answer. Iteration 0 is the EKS.
// The filtered accessors always report the original forward EKF pass.
//
// (Before 2026-11-16 iteration re-ran the forward EKF from the smoothed x_0
// with covariance P0 -- the data counted twice; on a linear model it moved up
// to 0.45 sigma away from the exact smoothed answer.)
// =============================================================================

#include <memory>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include "EKF.h"
#include "EKFFixedLag.h"   // reuse the FilterState history struct
#include "FilterMath.h"
#include "AffineRTS.h"

class EKFSmoother {
public:
    explicit EKFSmoother(SystemModel* model) : model_(model) {}

    /** Set the prior; clears any stored trajectory/measurements. */
    void initialize(const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0) {
        x0_ = x0; P0_ = P0; meas_.clear();
        x_smooth_.clear(); P_smooth_.clear();
        reset_forward(x0_, P0_);
    }

    /** Consume one measurement (retained for iteration replay); no smoothing yet. */
    void step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
        meas_.push_back(Meas{t_k, y_k, u_k});
        forward_step(y_k, u_k, t_k);
    }

    /**
     * Backward RTS over the full interval (the EKS), then n_iterations
     * Gauss-Newton IEKS iterations. Idempotent: always starts from the stored
     * forward pass, so calling smooth(n) twice gives the same answer.
     */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations && !x_smooth_.empty(); ++it) gauss_newton_iteration();
    }

    int size() const { return static_cast<int>(buffer_.size()); }
    // Bounds-checked: an index past size() throws std::out_of_range.
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> filtered_state(int k) const {
        const FilterState& s = buffer_.at(static_cast<size_t>(k));
        return {s.x_upd, s.P_upd};
    }
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> smoothed_state(int k) const {
        return {x_smooth_.at(static_cast<size_t>(k)), P_smooth_.at(static_cast<size_t>(k))};
    }
    Eigen::VectorXf smoothed_initial() const { return x_smooth_.at(0); }
    Eigen::VectorXf final_filtered_state() const { return buffer_.back().x_upd; }

private:
    struct Meas { float t; Eigen::VectorXf y; Eigen::VectorXf u; };

    SystemModel* model_;
    std::unique_ptr<EKF> ekf_;
    Eigen::VectorXf x0_;
    Eigen::MatrixXf P0_;
    std::vector<FilterState> buffer_;
    std::vector<Meas> meas_;
    std::vector<Eigen::VectorXf> x_smooth_;
    std::vector<Eigen::MatrixXf> P_smooth_;

    void reset_forward(const Eigen::VectorXf& x, const Eigen::MatrixXf& P) {
        ekf_ = std::make_unique<EKF>(model_, x, P);
        buffer_.clear();
        FilterState s;
        s.t = 0.0f; s.x_upd = x; s.P_upd = P;
        buffer_.push_back(s);
    }

    void forward_step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
        const Eigen::VectorXf x_before = ekf_->getState();
        const Eigen::MatrixXf P_before = ekf_->getCovariance();
        Eigen::MatrixXf F = ekf_->predict(u_k, t_k);
        FilterState s;
        s.t = t_k;
        s.x_pred = ekf_->getPredictedState();
        s.P_pred = ekf_->getPredictedCovariance();
        ekf_->update(y_k, t_k);
        s.x_upd = ekf_->getState();
        s.P_upd = ekf_->getCovariance();
        s.F = F;
        // A NaN measurement never gets here (EKF::update treats it as missed).
        // A model returning non-finite values cannot be skipped: the old guard
        // returned early but left the EKF NaN, so every later step was dropped
        // and buffer_ fell out of step with meas_. Roll back and report.
        if (!s.x_upd.allFinite() || !s.P_upd.allFinite()) {
            ekf_->setState(x_before);
            ekf_->setCovariance(P_before);
            throw std::runtime_error("EKFSmoother: model produced a non-finite state; "
                                     "filter restored to its pre-step estimate");
        }
        buffer_.push_back(s);
    }

    // One IEKS iteration: Jacobian linearisation about x_smooth_, exact affine solve.
    void gauss_newton_iteration() {
        const size_t T = meas_.size();
        if (x_smooth_.size() != T + 1) return;
        affine_rts::Problem pr;
        pr.m0 = x0_.cast<double>();
        pr.P0 = P0_.cast<double>();
        pr.dyn.resize(T);
        pr.obs.resize(T);
        for (size_t j = 0; j < T; ++j) {
            const Meas& m = meas_[j];
            const Eigen::VectorXf& xa = x_smooth_[j];       // linearise f at x_j
            const Eigen::VectorXf& xb = x_smooth_[j + 1];   // linearise h at x_{j+1}
            affine_rts::Dynamics& d = pr.dyn[j];
            d.xlin = xa.cast<double>();
            d.f0 = model_->f(xa, m.u, m.t).cast<double>();
            d.A = model_->F(xa, m.u, m.t).cast<double>();
            d.Q = model_->Q(m.t).cast<double>();
            affine_rts::Measurement& z = pr.obs[j];
            z.active = m.y.allFinite();                    // as EKF::update
            if (!z.active) continue;
            z.y = m.y.cast<double>();
            z.zlin = xb.cast<double>();
            z.h0 = model_->h(xb, m.t).cast<double>();
            z.C = model_->H(xb, m.t).cast<double>();
            z.R = model_->R(m.t).cast<double>();
        }
        const affine_rts::Result r = affine_rts::smooth(pr);
        for (size_t k = 0; k <= T; ++k) {
            x_smooth_[k] = r.m[k].cast<float>();
            P_smooth_[k] = r.P[k].cast<float>();
        }
    }

    // Backward RTS over the full buffer (identical math to EKFFixedLag::step).
    void backward_pass() {
        const int N = static_cast<int>(buffer_.size());
        x_smooth_.assign(N, Eigen::VectorXf());
        P_smooth_.assign(N, Eigen::MatrixXf());
        if (N == 0) return;
        x_smooth_[N-1] = buffer_[N-1].x_upd;
        P_smooth_[N-1] = buffer_[N-1].P_upd;
        for (int j = N - 2; j >= 0; --j) {
            // G_j = P_{j|j} F_{j+1}^T P_{j+1|j}^{-1}  (via SPD solve, no explicit inverse)
            Eigen::MatrixXf PFt = filtermath::gemm(buffer_[j].P_upd,
                                                   buffer_[j+1].F.transpose());
            Eigen::MatrixXf G = filtermath::kalman_gain(PFt, buffer_[j+1].P_pred);
            Eigen::VectorXf diff_x = x_smooth_[j+1] - buffer_[j+1].x_pred;
            x_smooth_[j] = buffer_[j].x_upd + filtermath::mat_vec_mul(G, diff_x);
            Eigen::MatrixXf diff_P = P_smooth_[j+1] - buffer_[j+1].P_pred;
            Eigen::MatrixXf t1 = filtermath::gemm(G, diff_P);
            P_smooth_[j] = buffer_[j].P_upd + filtermath::gemm(t1, G.transpose());
        }
    }
};

#endif // EKF_SMOOTHER_H
