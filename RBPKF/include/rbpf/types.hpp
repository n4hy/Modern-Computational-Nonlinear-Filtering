#ifndef RBPF_TYPES_HPP
#define RBPF_TYPES_HPP

#include <Eigen/Dense>

namespace rbpf {

/**
 * @brief Template struct to define dimensions and types for the RBPF.
 *
 * @tparam N_NL  Dimension of the nonlinear state (particle part)
 * @tparam N_LIN Dimension of the linear state (Kalman filter part)
 * @tparam N_Y   Dimension of the observation vector
 */
template<int N_NL, int N_LIN, int N_Y>
struct RbpfTypes {
    static constexpr int Nnl = N_NL;
    static constexpr int Nlin = N_LIN;
    static constexpr int Ny = N_Y;

    // Vector types
    using NonlinearState    = Eigen::Matrix<float, Nnl, 1>;
    using LinearState       = Eigen::Matrix<float, Nlin, 1>;
    using Observation       = Eigen::Matrix<float, Ny, 1>;

    // Matrix types
    using LinearCov         = Eigen::Matrix<float, Nlin, Nlin>;
    using ObsCov            = Eigen::Matrix<float, Ny, Ny>;
    using CrossCov          = Eigen::Matrix<float, Nlin, Ny>;
    using NonlinearCov      = Eigen::Matrix<float, Nnl, Nnl>;

    // System matrices types (for linear part)
    using Matrix_A          = Eigen::Matrix<float, Nlin, Nlin>;
    using Matrix_B          = Eigen::Matrix<float, Nlin, Eigen::Dynamic>;
    using Matrix_H          = Eigen::Matrix<float, Ny, Nlin>;
};

} // namespace rbpf

#endif // RBPF_TYPES_HPP
