#ifndef RBPF_KALMAN_FILTER_HPP
#define RBPF_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <iostream>

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
     *
     * @param A State transition matrix.
     * @param bias Deterministic input/bias term (B*u + etc).
     * @param Q Process noise covariance.
     */
    void predict(const Eigen::Ref<const Eigen::MatrixXd>& A,
                 const Eigen::Ref<const LinearState>& bias,
                 const LinearCov& Q) {
        // x_k|k-1 = A * x_{k-1|k-1} + bias
        x = A * x + bias;

        // P_k|k-1 = A * P_{k-1|k-1} * A^T + Q
        P = A * P * A.transpose() + Q;

        // Ensure symmetry
        P = 0.5 * (P + P.transpose());
    }

    /**
     * @brief Update step using Joseph form for stability.
     *
     * @param y Measurement vector.
     * @param H Observation matrix.
     * @param offset Measurement offset (y = Hx + offset + v).
     * @param R Measurement noise covariance.
     */
    void update(const Eigen::Ref<const Observation>& y,
                const Eigen::Ref<const Eigen::MatrixXd>& H,
                const Eigen::Ref<const Observation>& offset,
                const ObsCov& R) {
        // Innovation: z = y - (H*x + offset)
        Observation z = y - (H * x + offset);

        // Innovation covariance: S = H*P*H^T + R
        ObsCov S = H * P * H.transpose() + R;

        // Kalman gain: K = P * H^T * S^{-1}
        // Using LDLT for inverse
        Eigen::Matrix<double, Types::Nlin, Types::Ny> K;
        K = P * H.transpose() * S.ldlt().solve(Eigen::Matrix<double, Types::Ny, Types::Ny>::Identity());

        // Update state: x = x + K*z
        x = x + K * z;

        // Update covariance (Joseph form): P = (I - KH)P(I - KH)^T + KRK^T
        Eigen::Matrix<double, Types::Nlin, Types::Nlin> I = Eigen::Matrix<double, Types::Nlin, Types::Nlin>::Identity();
        Eigen::Matrix<double, Types::Nlin, Types::Nlin> I_KH = I - K * H;

        P = I_KH * P * I_KH.transpose() + K * R * K.transpose();

        // Ensure symmetry
        P = 0.5 * (P + P.transpose());
    }
};

} // namespace rbpf

#endif // RBPF_KALMAN_FILTER_HPP
