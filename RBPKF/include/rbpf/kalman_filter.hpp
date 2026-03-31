#ifndef RBPF_KALMAN_FILTER_HPP
#define RBPF_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <iostream>
#include "FilterMath.h"

namespace rbpf {

/**
 * @brief Generic reusable Kalman Filter class.
 *
 * @tparam Types Typdef struct containing dimensions and matrix types.
 */
template<typename Types>
class LinearKalmanFilter {
public:
    using LinearState = typename Types::LinearState;
    using LinearCov   = typename Types::LinearCov;
    using Observation = typename Types::Observation;
    using ObsCov      = typename Types::ObsCov;

    LinearState x;
    LinearCov   P;

    /**
     * @brief Initialize the filter state and covariance.
     */
    void initialize(const LinearState& x0, const LinearCov& P0) {
        x = x0;
        P = P0;
    }

    /**
     * @brief Predict step: x = A*x + bias, P = A*P*A^T + Q
     */
    void predict(const Eigen::Ref<const Eigen::MatrixXf>& A,
                 const Eigen::Ref<const LinearState>& bias,
                 const LinearCov& Q) {
        // x_k|k-1 = A * x_{k-1|k-1} + bias
        Eigen::VectorXf Ax = filtermath::mat_vec_mul(A, Eigen::VectorXf(x));
        x = Ax + bias;

        // P_k|k-1 = A * P * A^T + Q
        Eigen::MatrixXf AP = filtermath::gemm(A, P);
        Eigen::MatrixXf APAt = filtermath::gemm(AP, A.transpose());

        P = APAt + Q;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }

    /**
     * @brief Update step using Joseph form for stability.
     */
    void update(const Eigen::Ref<const Observation>& y,
                const Eigen::Ref<const Eigen::MatrixXf>& H,
                const Eigen::Ref<const Observation>& offset,
                const ObsCov& R) {
        // Innovation: z = y - (H*x + offset)
        Eigen::VectorXf Hx = filtermath::mat_vec_mul(H, Eigen::VectorXf(x));
        Observation z = y - (Hx + offset);

        // Innovation covariance: S = H*P*H^T + R
        Eigen::MatrixXf HP = filtermath::gemm(H, P);
        Eigen::MatrixXf HPHt = filtermath::gemm(HP, H.transpose());
        ObsCov S = HPHt + R;

        // Kalman gain via SPD solve (more stable than explicit inverse)
        Eigen::MatrixXf PHt = filtermath::gemm(P, H.transpose());
        Eigen::Matrix<float, Types::Nlin, Types::Ny> K = filtermath::kalman_gain(PHt, S);

        // Update state: x = x + K*z
        Eigen::VectorXf Kz = filtermath::mat_vec_mul(K, Eigen::VectorXf(z));
        x = x + Kz;

        // Update covariance (Joseph form): P = (I - KH)P(I - KH)^T + KRK^T
        Eigen::Matrix<float, Types::Nlin, Types::Nlin> I = Eigen::Matrix<float, Types::Nlin, Types::Nlin>::Identity();

        Eigen::MatrixXf KH = filtermath::gemm(K, H);
        Eigen::Matrix<float, Types::Nlin, Types::Nlin> I_KH = I - KH;

        Eigen::MatrixXf P_I_KH_T = filtermath::gemm(P, I_KH.transpose());
        Eigen::MatrixXf Term1 = filtermath::gemm(I_KH, P_I_KH_T);

        Eigen::MatrixXf RKt = filtermath::gemm(R, K.transpose());
        Eigen::MatrixXf Term2 = filtermath::gemm(K, RKt);

        P = Term1 + Term2;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }
};

} // namespace rbpf

#endif // RBPF_KALMAN_FILTER_HPP
