#include "UKF.h"
#include <iostream>

using namespace Eigen;

UKF::UKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
    : model_(model), x_(x0), P_(P0) {

    n_ = model->getStateDim();

    // Standard UKF parameters
    alpha_ = 0.001;
    beta_ = 2.0; // Optimal for Gaussian
    kappa_ = 0.0;

    // Initialize Weights
    int lambda = alpha_ * alpha_ * (n_ + kappa_) - n_;
    int num_sigmas = 2 * n_ + 1;

    weights_m_ = VectorXd(num_sigmas);
    weights_c_ = VectorXd(num_sigmas);

    weights_m_(0) = (double)lambda / (n_ + lambda);
    weights_c_(0) = weights_m_(0) + (1 - alpha_ * alpha_ + beta_);

    for (int i = 1; i < num_sigmas; ++i) {
        double w = 0.5 / (n_ + lambda);
        weights_m_(i) = w;
        weights_c_(i) = w;
    }
}

void UKF::generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P, std::vector<Eigen::VectorXd>& out_sigmas) {
    out_sigmas.resize(2 * n_ + 1);

    double lambda = alpha_ * alpha_ * (n_ + kappa_) - n_;

    // Cholesky Decomposition: P = L * L^T
    // We need sqrt((n + lambda) * P)
    // Note: LLT might fail if P is not pos-def (numerical issues).
    LLT<MatrixXd> lltOfP(P);
    if(lltOfP.info() == Eigen::NumericalIssue) {
        // Fallback or error?
        // Simple regularization if needed
        LLT<MatrixXd> lltFix(P + MatrixXd::Identity(n_, n_) * 1e-9);
        if(lltFix.info() == Eigen::NumericalIssue) {
             std::cerr << "UKF Error: P is not positive definite!" << std::endl;
             // Don't crash, but result will be bad.
             return;
        }
        MatrixXd L = lltFix.matrixL();
        MatrixXd S = std::sqrt(n_ + lambda) * L;

        out_sigmas[0] = x;
        for (int i = 0; i < n_; ++i) {
            out_sigmas[i + 1]      = x + S.col(i);
            out_sigmas[i + 1 + n_] = x - S.col(i);
        }
        return;
    }

    MatrixXd L = lltOfP.matrixL();
    MatrixXd S = std::sqrt(n_ + lambda) * L;

    out_sigmas[0] = x;
    for (int i = 0; i < n_; ++i) {
        out_sigmas[i + 1]      = x + S.col(i);
        out_sigmas[i + 1 + n_] = x - S.col(i);
    }
}

Eigen::MatrixXd UKF::predict() {
    // Dummy values for u and t
    Eigen::VectorXd u;
    double t = 0.0;

    // 1. Generate Sigma Points X_k-1|k-1
    generateSigmaPoints(x_, P_, sigma_points_);

    // 2. Propagate Sigma Points through Process Model: X_k|k-1 = f(X_k-1|k-1)
    std::vector<VectorXd> pred_sigmas(2 * n_ + 1);
    for (size_t i = 0; i < sigma_points_.size(); ++i) {
        pred_sigmas[i] = model_->f(sigma_points_[i], u, t);
    }

    // 3. Compute Predicted Mean x_k|k-1
    VectorXd x_pred = VectorXd::Zero(n_);
    for (size_t i = 0; i < pred_sigmas.size(); ++i) {
        x_pred += weights_m_(i) * pred_sigmas[i];
    }

    // 4. Compute Predicted Covariance P_k|k-1
    MatrixXd P_pred = MatrixXd::Zero(n_, n_);
    for (size_t i = 0; i < pred_sigmas.size(); ++i) {
        VectorXd diff = pred_sigmas[i] - x_pred;
        P_pred += weights_c_(i) * diff * diff.transpose();
    }
    P_pred += model_->Q(t);

    // 5. Compute Cross Covariance P_{x_k-1, x_k} (needed for smoothing)
    // P_{x, z} logic but here z is x_k (next state).
    // D_{k+1} = sum Wc * (X_k-1 - x_k-1) * (X_k - x_pred)^T
    MatrixXd P_cross = MatrixXd::Zero(n_, n_);
    for (size_t i = 0; i < sigma_points_.size(); ++i) {
        VectorXd diff_old = sigma_points_[i] - x_;
        VectorXd diff_new = pred_sigmas[i] - x_pred;
        P_cross += weights_c_(i) * diff_old * diff_new.transpose();
    }

    // Update internal state
    x_ = x_pred;
    P_ = P_pred;

    // For update step we'll need new sigma points from the NEW P_pred?
    // Standard UKF usually regenerates sigma points after prediction
    // to capture the effect of Q (process noise).
    // Yes, we will regenerate in update().

    return P_cross;
}

void UKF::update(const Eigen::VectorXd& y) {
    int m = model_->getObsDim();
    double t = 0.0;

    // 1. Generate Sigma Points for predicted state x_k|k-1
    // (Augmentation with measurement noise not strictly needed for additive noise)
    generateSigmaPoints(x_, P_, sigma_points_);

    // 2. Propagate through Measurement Model: Y = h(X)
    std::vector<VectorXd> meas_sigmas(2 * n_ + 1);
    for (size_t i = 0; i < sigma_points_.size(); ++i) {
        meas_sigmas[i] = model_->h(sigma_points_[i], t);
    }

    // 3. Predicted Measurement Mean
    VectorXd y_pred = VectorXd::Zero(m);
    for (size_t i = 0; i < meas_sigmas.size(); ++i) {
        y_pred += weights_m_(i) * meas_sigmas[i];
    }

    // 4. Innovation Covariance S
    MatrixXd S = MatrixXd::Zero(m, m);
    for (size_t i = 0; i < meas_sigmas.size(); ++i) {
        VectorXd diff = meas_sigmas[i] - y_pred;
        S += weights_c_(i) * diff * diff.transpose();
    }
    S += model_->R(t);

    // 5. Cross Covariance P_xy
    MatrixXd P_xy = MatrixXd::Zero(n_, m);
    for (size_t i = 0; i < sigma_points_.size(); ++i) {
        VectorXd diff_x = sigma_points_[i] - x_;
        VectorXd diff_y = meas_sigmas[i] - y_pred;
        P_xy += weights_c_(i) * diff_x * diff_y.transpose();
    }

    // 6. Kalman Gain
    MatrixXd K = P_xy * S.inverse();

    // 7. Update State and Covariance
    VectorXd innov = y - y_pred;
    x_ = x_ + K * innov;
    P_ = P_ - K * S * K.transpose();
}
