#ifndef RBPF_STATE_SPACE_MODELS_HPP
#define RBPF_STATE_SPACE_MODELS_HPP

#include <Eigen/Dense>
#include <random>

namespace rbpf {

/**
 * @brief Interface for the nonlinear state evolution (particle part).
 */
template<typename Types>
class NonlinearModel {
public:
    using NonlinearState = typename Types::NonlinearState;
    using Observation    = typename Types::Observation;

    virtual ~NonlinearModel() = default;

    /**
     * @brief Propagate the nonlinear state.
     */
    virtual NonlinearState propagate(const NonlinearState& x_nl_prev,
                                     float t_k,
                                     const NonlinearState& u_k,
                                     std::mt19937_64& rng) const = 0;

    /**
     * @brief Log proposal density.
     */
    virtual float log_proposal_density(const NonlinearState& x_nl_curr,
                                        const NonlinearState& x_nl_prev,
                                        float t_k,
                                        const NonlinearState& u_k) const = 0;
};

/**
 * @brief Interface for the conditional linear-Gaussian model.
 */
template<typename Types>
class ConditionalLinearGaussianModel {
public:
    using NonlinearState = typename Types::NonlinearState;
    using LinearState    = typename Types::LinearState;
    using Observation    = typename Types::Observation;
    using LinearCov      = typename Types::LinearCov;
    using ObsCov         = typename Types::ObsCov;
    using CrossCov       = typename Types::CrossCov;

    virtual ~ConditionalLinearGaussianModel() = default;

    /**
     * @brief Get linear dynamics matrices: x_lin_k = A * x_lin_{k-1} + B * u_k + w_lin
     */
    virtual void get_dynamics(const NonlinearState& x_nl_prev,
                              float t_k,
                              Eigen::Ref<LinearState> bias,
                              Eigen::Ref<Eigen::MatrixXf> A,
                              Eigen::Ref<Eigen::MatrixXf> B,
                              Eigen::Ref<LinearCov> Q) const = 0;

    /**
     * @brief Get linear observation matrices: y_k = H * x_lin_k + offset + v
     */
    virtual void get_observation(const NonlinearState& x_nl_curr,
                                 float t_k,
                                 Eigen::Ref<Observation> offset,
                                 Eigen::Ref<Eigen::MatrixXf> H,
                                 Eigen::Ref<ObsCov> R) const = 0;
};

} // namespace rbpf

#endif // RBPF_STATE_SPACE_MODELS_HPP
