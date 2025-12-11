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
    using NonlinearState    = Eigen::Matrix<double, Nnl, 1>;
    using LinearState       = Eigen::Matrix<double, Nlin, 1>;
    using Observation       = Eigen::Matrix<double, Ny, 1>;

    // Matrix types
    using LinearCov         = Eigen::Matrix<double, Nlin, Nlin>;
    using ObsCov            = Eigen::Matrix<double, Ny, Ny>;
    using CrossCov          = Eigen::Matrix<double, Nlin, Ny>;
    using NonlinearCov      = Eigen::Matrix<double, Nnl, Nnl>;

    // System matrices types (for linear part)
    using Matrix_A          = Eigen::Matrix<double, Nlin, Nlin>;
    using Matrix_B          = Eigen::Matrix<double, Nlin, Eigen::Dynamic>;
    using Matrix_H          = Eigen::Matrix<double, Ny, Nlin>;
};

} // namespace rbpf

#endif // RBPF_TYPES_HPP
