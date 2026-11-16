#ifndef SRUKF_SMOOTHER_H
#define SRUKF_SMOOTHER_H

// =============================================================================
// SRUKFSmoother -- full-interval Square-Root Unscented RTS smoother, Sarkka
// form (Alg. 10.6 of "Bayesian Filtering and Smoothing", Cambridge 2013).
// -----------------------------------------------------------------------------
// Everything stays in Cholesky-factor form on the backward pass -- no full
// covariance is ever materialised, no P^{-1} is ever taken. At each backward
// epoch j we:
//
//   1. Regenerate sigma points chi_j^(i) directly from the filtered factor
//      (m_j, A_j), with A_j A_j^T = P_j.
//   2. Propagate through model_.f with SRUKF's angular-state conventions
//      (circular mean, wrap deviations by std::remainder to [-pi, pi]).
//   3. Build A_tilde_{j+1} -- the Cholesky factor of the predicted covariance
//      P_tilde_{j+1} + Q_{j+1} -- via QR of the weighted deviations for
//      i=1..2n plus sqrt(Q), then a rank-1 cholupdate (or downdate if Wc(0)<0)
//      for i=0.  Identical construction to SRUKF::predict.
//   4. Compute the cross-covariance D_{j+1} = sum Wc(i) (chi_j^(i)-m_j)
//      (chi_hat_{j+1}^(i)-m_tilde_{j+1})^T via GEMM.
//   5. Solve for the smoother gain by two triangular solves, keeping A_tilde
//      as the operational factor:
//         Step A:  A_tilde * U^T   = D^T           (forward substitution)
//         Step B:  A_tilde^T * G^T = U^T           (back substitution)
//      so U = D * A_tilde^{-T} and G = D * (A_tilde A_tilde^T)^{-1}.
//   6. Smoothed mean:  m_j^s = m_j + G (m_{j+1}^s - m_tilde_{j+1}).
//   7. Compose the smoothed factor.  Using U U^T = D * P_tilde^{-1} * D^T,
//      the URTS covariance identity rearranges as
//         P_j^s = (P_j - U U^T) + (G * A_{j+1}^s) (G * A_{j+1}^s)^T,
//      where the first bracket is the Schur-complement conditional covariance
//      Cov(x_j | x_{j+1}) -- PSD in exact arithmetic.  So:
//         a) Downdate A_j by each column of U to get A_cond, with A_cond
//            A_cond^T = P_j - U U^T.
//         b) QR the stacked matrix [A_cond | G * A_{j+1}^s]^T; the R factor
//            (transposed) is A_j^s.
//      If (a) fails a rank-1 downdate (near-singular P_tilde), fall back once
//      to full-covariance re-Cholesky with the same jitter ladder used
//      elsewhere in the SRUKF pipeline.
//
// smooth(n>0) then runs n iterations of the iterated posterior linearisation
// smoother (IPLS, PosteriorLinearization.h): sigma points are drawn directly
// from the current smoothed Cholesky factors, f and h are replaced by their
// statistical linear regressions, and the affine model is solved exactly from
// the ORIGINAL prior. That inner solve is in double-precision covariance form
// (affine_rts) and its covariances are re-factored by Cholesky (double) before
// returning to float factors, so the float SR guarantees are not weakened.
// On a linear model every iteration returns the exact RTS answer. The filtered
// accessors always report the original forward SRUKF pass. (Before 2026-11-16
// iteration re-ran the forward filter from the smoothed x_0 with covariance
// P0, counting the data twice.)
// =============================================================================

#include <optional>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numbers>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "SRUKF.h"
#include "SigmaPoints.h"
#include "FilterMath.h"
#include "PosteriorLinearization.h"

namespace UKFCore {

template<int NX, int NY>
class SRUKFSmoother {
public:
    using SRUKFType = SRUKF<NX, NY>;
    using Model = typename SRUKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;
    using SigmaPts = SigmaPoints<NX>;

    explicit SRUKFSmoother(Model& model) : model_(model), srukf_(model) {}

    void initialize(const State& x0, const StateMat& P0) {
        x0_ = x0;
        P0_ = P0;
        meas_.clear();
        init_obs_.reset();
        smoothed_states_.clear();
        smoothed_A_.clear();
        reset_forward(x0_, P0_);
    }

    /**
     * Fuse a measurement taken at the initialisation time t0, without
     * propagating (same contract as the fixed-lag smoothers). Refines
     * history_[0] in place; call at most once, before the first step().
     */
    void observe_initial(float t0, const Observation& y_0) {
        init_obs_ = InitObs{t0, y_0};
        apply_initial_observation();
    }

    /**
     * Consume one measurement (stored for iteration replay); no smoothing yet.
     * Propagates FROM t_prev (where the estimate is) TO t_k, then fuses y_k,
     * which observes the state at t_k -- the convention of the fixed-lag
     * smoothers and the benchmark (commit 3dbbf4c). A non-finite y_k is a
     * missed measurement (SRUKF::update skips it).
     * Throws std::runtime_error if the model makes the state non-finite; the
     * filter is restored to its pre-step state first.
     */
    void step(float t_prev, float t_k, const Observation& y_k,
              const Eigen::Ref<const State>& u_k) {
        meas_.push_back(Meas{t_prev, t_k, y_k, State(u_k)});
        forward_step(meas_.back());
    }

    /**
     * Backward SR-URTS over the full interval, then n_iterations IPLS
     * iterations. Idempotent: always starts from the stored forward pass.
     */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (smoothed_states_.size() != meas_.size() + 1) break;
            const auto r = ipls::iterate<NX, NY>(
                model_, x0_, P0_, init_obs_, meas_, smoothed_states_,
                [&](size_t k, SigmaPts& sig) {
                    generate_sigma_points_from_sqrt<NX>(smoothed_states_[k], smoothed_A_[k],
                                                        srukf_.alpha, srukf_.beta, srukf_.kappa, sig);
                });
            for (size_t k = 0; k < r.m.size(); ++k) {
                smoothed_states_[k] = r.m[k].template cast<float>();
                smoothed_A_[k] = sqrt_spd_double(r.P[k]);
            }
        }
    }

    int size() const { return static_cast<int>(history_.size()); }
    // Bounds-checked: an index past size() throws std::out_of_range.
    State  filtered_state(int k)      const { return history_.at(static_cast<size_t>(k)).x; }
    StateMat filtered_covariance(int k) const {
        const StateMat& A = history_.at(static_cast<size_t>(k)).A;
        return A * A.transpose();
    }
    State  smoothed_state(int k)      const { return smoothed_states_.at(static_cast<size_t>(k)); }
    StateMat smoothed_covariance(int k) const {
        const StateMat& A = smoothed_A_.at(static_cast<size_t>(k));
        return A * A.transpose();
    }
    const StateMat& smoothed_sqrt_covariance(int k) const { return smoothed_A_.at(static_cast<size_t>(k)); }
    State  smoothed_initial() const { return smoothed_states_.at(0); }
    State  final_filtered_state() const { return history_.back().x; }

private:
    struct Meas { float t_prev; float t; Observation y; State u; };
    struct InitObs { float t; Observation y; };
    struct FilteredSR { State x; StateMat A; };  // A * A^T = P_{k|k}

    Model& model_;
    SRUKFType srukf_;
    State x0_;
    StateMat P0_;
    std::vector<FilteredSR> history_;         // history_[0]=prior; history_[k>=1]=(m_k,A_k)
    std::vector<Meas> meas_;                  // meas_[j] drives history_[j] -> history_[j+1]
    std::optional<InitObs> init_obs_;         // replayed on every iteration
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_A_;        // A_s A_s^T = P_s

    // ---- setup / forward pass ----

    // Lower Cholesky factor of a double SPD matrix, as float. Falls back to the
    // pivot-aware LDLT factor (zero pivots floored) if P is only semidefinite.
    static StateMat sqrt_spd_double(const Eigen::MatrixXd& P) {
        const Eigen::LLT<Eigen::MatrixXd> llt(P);
        if (llt.info() == Eigen::Success) return Eigen::MatrixXd(llt.matrixL()).cast<float>();
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
        const Eigen::VectorXd d = ldlt.vectorD().cwiseMax(1e-20).cwiseSqrt();
        const Eigen::MatrixXd L = ldlt.matrixL();
        return Eigen::MatrixXd(ldlt.transpositionsP().transpose() * (L * d.asDiagonal())).cast<float>();
    }

    static StateMat sqrt_spd(const StateMat& P) {
        Eigen::MatrixXf L = filtermath::cholesky(P);
        if (L.size() == 0) L = filtermath::cholesky(P + 1e-6f * StateMat::Identity());
        if (L.size() == 0) {
            Eigen::LLT<StateMat> llt(P + 1e-8f * StateMat::Identity());
            return (llt.info() == Eigen::Success) ? StateMat(llt.matrixL())
                                                  : StateMat(StateMat::Identity());
        }
        return StateMat(L);
    }

    void reset_forward(const State& x, const StateMat& P) {
        srukf_.initialize(x, P);
        history_.clear();
        history_.push_back(FilteredSR{x, sqrt_spd(P)});
    }

    void apply_initial_observation() {
        if (!init_obs_ || history_.empty()) return;
        srukf_.update(init_obs_->t, init_obs_->y);
        require_finite(history_[0]);
        history_[0] = FilteredSR{srukf_.getState(), srukf_.getSqrtCovariance()};
    }

    void forward_step(const Meas& m) {
        if (history_.empty()) return;
        const FilteredSR before{srukf_.getState(), srukf_.getSqrtCovariance()};
        srukf_.predict(m.t_prev, m.u);  // returned cross-cov is not used by SR-URTS
        srukf_.update(m.t, m.y);
        require_finite(before);
        history_.push_back(FilteredSR{srukf_.getState(), srukf_.getSqrtCovariance()});
    }

    // A non-finite state cannot be skipped: it would desynchronise history_
    // from meas_ (the old guard returned early but left the SRUKF itself NaN,
    // so every later step was lost too). Restore and report instead.
    void require_finite(const FilteredSR& before) {
        if (srukf_.getState().allFinite() && srukf_.getSqrtCovariance().allFinite()) return;
        srukf_.restore(before.x, before.A);
        throw std::runtime_error("SRUKFSmoother: model produced a non-finite state; "
                                 "filter restored to its pre-step estimate");
    }

    // ---- backward pass helpers ----

    // Rank-1 Cholesky update:  S_new S_new^T = S S^T + alpha^2 v v^T .
    // Copy of the routine in SRUKF.h (kept local so the smoother is self-contained).
    static void cholupdate(StateMat& S, const State& v, float alpha) {
        State w = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) S(k,k) = eps;
            float r = std::sqrt(S(k,k)*S(k,k) + w(k)*w(k));
            float c = r / S(k,k);
            float s = w(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps)
                    S.block(k+1, k, NX-k-1, 1) =
                        (S.block(k+1, k, NX-k-1, 1) + s * w.segment(k+1, NX-k-1)) / c;
                w.segment(k+1, NX-k-1) = c * w.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
    }

    // Rank-1 Cholesky downdate:  S_new S_new^T = S S^T - alpha^2 v v^T .
    // Returns false if the downdate would produce a non-PSD result (near-singular
    // argument); caller then falls back to a full-covariance re-Cholesky.
    static bool cholupdate_downdate_safe(StateMat& S, const State& v, float alpha) {
        State w = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) S(k,k) = eps;
            float r_sq = S(k,k)*S(k,k) - w(k)*w(k);
            if (r_sq <= 0) return false;
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = w(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps)
                    S.block(k+1, k, NX-k-1, 1) =
                        (S.block(k+1, k, NX-k-1, 1) - s * w.segment(k+1, NX-k-1)) / c;
                w.segment(k+1, NX-k-1) = c * w.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
        return true;
    }

    // Wrap x into [-pi, pi] iff state-index j is declared angular by the model.
    void wrap_state_deviation(State& d) const {
        for (int j = 0; j < NX; ++j) {
            if (model_.isAngularState(j))
                d(j) = std::remainder(d(j), 2.0f * std::numbers::pi_v<float>);
        }
    }

    // Circular mean for angular components; ordinary weighted mean for the rest.
    State weighted_mean_angular(
        const Eigen::Matrix<float, NX, SigmaPts::NSIG>& Xp,
        const typename SigmaPts::Weights& Wm) const
    {
        State m = State::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i)
            m += Wm(i) * Xp.col(i);
        for (int j = 0; j < NX; ++j) {
            if (model_.isAngularState(j)) {
                float sin_sum = 0.0f, cos_sum = 0.0f;
                for (int i = 0; i < SigmaPts::NSIG; ++i) {
                    sin_sum += Wm(i) * std::sin(Xp(j, i));
                    cos_sum += Wm(i) * std::cos(Xp(j, i));
                }
                m(j) = std::atan2(sin_sum, cos_sum);
            }
        }
        return m;
    }

    // Build the SR predicted factor A_tilde with A_tilde A_tilde^T =
    // sum_i Wc(i) (chi_hat_i - m_tilde)(...)^T + Q .  Mirrors SRUKF::predict:
    // QR of weighted deviations for i=1..2n + sqrt(Q), then rank-1
    // update/downdate for i=0 depending on the sign of Wc(0).
    // Returns true on success; false only if the i=0 downdate fails AND the
    // full-cov Cholesky fallback fails too.
    bool build_predicted_factor(
        const Eigen::Matrix<float, NX, SigmaPts::NSIG>& Xp,
        const State& m_tilde,
        const SigmaPts& sig,
        const StateMat& Q,
        StateMat& A_tilde_out) const
    {
        // sqrt(Q) via the standard Cholesky-with-jitter ladder.
        StateMat S_Q = StateMat::Identity() * 1e-4f;
        {
            Eigen::MatrixXf L_Q = filtermath::cholesky(Q);
            if (L_Q.size() == 0)
                L_Q = filtermath::cholesky(Q + 1e-6f * StateMat::Identity());
            if (L_Q.size() > 0) {
                S_Q = L_Q;
            } else {
                Eigen::LDLT<StateMat> ldlt(Q + 1e-6f * StateMat::Identity());
                if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                    StateMat L = ldlt.matrixL();
                    Eigen::VectorXf D = ldlt.vectorD();
                    for (int i = 0; i < NX; ++i) D(i) = std::sqrt(std::max(D(i), 1e-10f));
                    S_Q = L * D.asDiagonal();
                }
            }
        }

        // Stack (2n)+n rows for QR: rows 0..2n-1 are weighted deviations for
        // i=1..2n (with the Wc sign folded into the row so a negative Wc(i)
        // subtracts on the associated rank-1 term); rows 2n..3n-1 are sqrt(Q)^T.
        Eigen::Matrix<float, 3*NX, NX> chi_diff_T;
        for (int i = 1; i < SigmaPts::NSIG; ++i) {
            State diff = Xp.col(i) - m_tilde;
            wrap_state_deviation(diff);
            float wc_sign = (sig.Wc(i) >= 0) ? 1.0f : -1.0f;
            chi_diff_T.row(i-1) =
                (std::sqrt(std::abs(sig.Wc(i))) * diff * wc_sign).transpose();
        }
        chi_diff_T.block(2*NX, 0, NX, NX) = S_Q.transpose();

        Eigen::HouseholderQR<Eigen::Matrix<float, 3*NX, NX>> qr(chi_diff_T);
        StateMat R = qr.matrixQR().template block<NX, NX>(0, 0)
                       .template triangularView<Eigen::Upper>();
        StateMat A = R.transpose();  // lower-triangular Cholesky factor so far

        // Fold in i=0.
        State diff_0 = Xp.col(0) - m_tilde;
        wrap_state_deviation(diff_0);
        float wc_0 = sig.Wc(0);
        if (wc_0 < 0) {
            if (!cholupdate_downdate_safe(A, diff_0, std::sqrt(std::abs(wc_0)))) {
                // Fallback: full-cov re-Cholesky with jitter ladder.
                StateMat P_full = StateMat::Zero();
                for (int i = 0; i < SigmaPts::NSIG; ++i) {
                    State d = Xp.col(i) - m_tilde;
                    wrap_state_deviation(d);
                    P_full += sig.Wc(i) * (d * d.transpose());
                }
                P_full += Q;
                P_full = 0.5f * (P_full + P_full.transpose()) + 1e-8f * StateMat::Identity();
                Eigen::MatrixXf L_fb = filtermath::cholesky(P_full);
                if (L_fb.size() == 0)
                    L_fb = filtermath::cholesky(P_full + 1e-6f * StateMat::Identity());
                if (L_fb.size() == 0) return false;
                A = L_fb;
            }
        } else if (wc_0 > 0) {
            cholupdate(A, diff_0, std::sqrt(wc_0));
        }
        // wc_0 == 0: nothing to fold.

        A_tilde_out = A;
        return true;
    }

    // Two triangular solves for G = D * (A A^T)^{-1}, and the by-product
    // U = D * A^{-T} which the smoothed-factor step needs.
    //   Step A:  solve A U^T = D^T   (forward substitution, A is lower)
    //   Step B:  solve A^T G^T = U^T (back substitution, A^T is upper)
    static void gain_from_factor(const StateMat& A_tilde,
                                 const StateMat& D_cross,
                                 StateMat& G_out,
                                 StateMat& U_out)
    {
        StateMat U_T = A_tilde.template triangularView<Eigen::Lower>()
                              .solve(D_cross.transpose());
        StateMat G_T = A_tilde.transpose()
                              .template triangularView<Eigen::Upper>()
                              .solve(U_T);
        U_out = U_T.transpose();
        G_out = G_T.transpose();
    }

    // Compose A_j^s such that A_j^s A_j^s^T = A_cond A_cond^T + (G A^s_{j+1}) (...)^T.
    // Achieved via QR of the transposed stacked matrix.
    static StateMat compose_smoothed_factor(const StateMat& A_cond,
                                            const StateMat& G_A_s_next)
    {
        Eigen::Matrix<float, 2*NX, NX> M_T;
        M_T.template block<NX, NX>(0,  0) = A_cond.transpose();
        M_T.template block<NX, NX>(NX, 0) = G_A_s_next.transpose();
        Eigen::HouseholderQR<Eigen::Matrix<float, 2*NX, NX>> qr(M_T);
        StateMat R = qr.matrixQR().template block<NX, NX>(0, 0)
                       .template triangularView<Eigen::Upper>();
        return R.transpose();
    }

    // Full-cov fallback for the smoothed factor when the rank-1 downdate chain
    // that would build A_cond fails.  Reconstructs P_j^s directly from the
    // available square-root pieces and re-Choleskies with the jitter ladder.
    static StateMat smoothed_factor_fallback(const StateMat& A_j,
                                             const StateMat& U,
                                             const StateMat& G_A_s_next)
    {
        StateMat P_cond = A_j * A_j.transpose() - U * U.transpose();
        StateMat P_s    = P_cond + G_A_s_next * G_A_s_next.transpose();
        P_s = 0.5f * (P_s + P_s.transpose()) + 1e-8f * StateMat::Identity();
        Eigen::MatrixXf L = filtermath::cholesky(P_s);
        if (L.size() == 0)
            L = filtermath::cholesky(P_s + 1e-6f * StateMat::Identity());
        if (L.size() == 0) {
            Eigen::LLT<StateMat> llt(P_s + 1e-6f * StateMat::Identity());
            if (llt.info() == Eigen::Success) return StateMat(llt.matrixL());
            return A_j;  // ultimate best-effort
        }
        return StateMat(L);
    }

    // ---- backward pass proper ----

    void backward_pass() {
        const int N = static_cast<int>(history_.size());
        smoothed_states_.assign(N, State::Zero());
        smoothed_A_.assign(N, StateMat::Identity());
        if (N == 0) return;
        smoothed_states_[N-1] = history_.back().x;
        smoothed_A_[N-1]      = history_.back().A;
        if (N == 1) return;
        if (static_cast<int>(meas_.size()) < N - 1) return;

        for (int j = N - 2; j >= 0; --j) {
            const State&    m_j   = history_[j].x;
            const StateMat& A_j   = history_[j].A;
            const float     t_j   = meas_[j].t_prev;  // propagate FROM t_j, as the forward step did
            const State&    u_jp1 = meas_[j].u;

            // 1) Sigma points from (m_j, A_j) directly.
            SigmaPts sig;
            generate_sigma_points_from_sqrt<NX>(
                m_j, A_j, srukf_.alpha, srukf_.beta, srukf_.kappa, sig);

            // 2) Propagate.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Xp;
            for (int i = 0; i < SigmaPts::NSIG; ++i)
                Xp.col(i) = model_.f(sig.X.col(i), t_j, u_jp1);

            // 3) Predicted mean (circular for angular states).
            State m_tilde = weighted_mean_angular(Xp, sig.Wm);

            // 4) Predicted factor A_tilde in Cholesky form.
            const StateMat Q_k = model_.Q(t_j);
            StateMat A_tilde;
            if (!build_predicted_factor(Xp, m_tilde, sig, Q_k, A_tilde)) {
                // Cannot form a valid factor; keep the smoothed = filtered here.
                smoothed_states_[j] = m_j;
                smoothed_A_[j]      = A_j;
                continue;
            }

            // 5) Cross-covariance D via GEMM.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Dp, Dx_w;
            for (int i = 0; i < SigmaPts::NSIG; ++i) {
                State dp = Xp.col(i)    - m_tilde;
                State dx = sig.X.col(i) - m_j;
                wrap_state_deviation(dp);
                wrap_state_deviation(dx);
                Dp.col(i)   = dp;
                Dx_w.col(i) = sig.Wc(i) * dx;
            }
            StateMat D_cross = filtermath::gemm(Dx_w, Dp.transpose());

            // 6) Gain and by-product U (kept for the covariance step).
            StateMat G, U;
            gain_from_factor(A_tilde, D_cross, G, U);

            // 7) Smoothed mean.  Wrap angular components of the innovation.
            State diff_m = smoothed_states_[j+1] - m_tilde;
            wrap_state_deviation(diff_m);
            smoothed_states_[j] = m_j + G * diff_m;

            // 8) Smoothed factor: downdate A_j by columns of U -> A_cond,
            //    then QR-compose with G * A_{j+1}^s.
            StateMat A_cond = A_j;
            bool downdate_ok = true;
            for (int c = 0; c < NX && downdate_ok; ++c)
                downdate_ok = cholupdate_downdate_safe(A_cond, U.col(c), 1.0f);

            StateMat G_A_s_next = G * smoothed_A_[j+1];

            if (downdate_ok) {
                smoothed_A_[j] = compose_smoothed_factor(A_cond, G_A_s_next);
            } else {
                smoothed_A_[j] = smoothed_factor_fallback(A_j, U, G_A_s_next);
            }
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_SMOOTHER_H
