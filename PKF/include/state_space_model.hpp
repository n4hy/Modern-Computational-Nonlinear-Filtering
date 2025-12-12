#ifndef PKF_STATE_SPACE_MODEL_HPP
#define PKF_STATE_SPACE_MODEL_HPP

#include <Eigen/Dense>
#include <random>

namespace PKF {

/**
 * @class StateSpaceModel
 * @brief Abstract base class for general nonlinear non-Gaussian state-space models.
 *
 * @tparam NX State dimension (must be <= 20)
 * @tparam NY Observation dimension (must be <= 20)
 */
template<int NX, int NY>
class StateSpaceModel {
public:
    static_assert(NX <= 20, "State dimension NX must be <= 20");
    static_assert(NY <= 20, "Observation dimension NY must be <= 20");

    using State       = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat    = Eigen::Matrix<float, NX, NX>;
    using ObsMat      = Eigen::Matrix<float, NY, NY>;

    virtual ~StateSpaceModel() = default;

    /**
     * @brief Propagate state: x_k = f(x_{k-1}, t_k, u_k) + w_k
     *
     * @param x_prev Previous state x_{k-1}
     * @param t_k Current time
     * @param u_k Control input
     * @return State Propagated state (deterministic part)
     */
    virtual State propagate(const State& x_prev,
                            float t_k,
                            const Eigen::Ref<const State>& u_k) const = 0;

    /**
     * @brief Observe state: y_k = h(x_k, t_k) + v_k
     *
     * @param x_k Current state
     * @param t_k Current time
     * @return Observation Expected observation (deterministic part)
     */
    virtual Observation observe(const State& x_k,
                                float t_k) const = 0;

    /**
     * @brief Sample process noise w_k
     *
     * @param t_k Current time
     * @param rng Random number generator
     * @return State Process noise sample
     */
    virtual State sample_process_noise(float t_k, std::mt19937_64& rng) const = 0;

    /**
     * @brief Sample observation noise v_k
     *
     * @param t_k Current time
     * @param rng Random number generator
     * @return Observation Observation noise sample
     */
    virtual Observation sample_observation_noise(float t_k, std::mt19937_64& rng) const = 0;

    /**
     * @brief Compute log-likelihood of observation p(y_k | x_k)
     *
     * @param y_k Actual observation
     * @param x_k Particle state
     * @param t_k Current time
     * @return float Log-likelihood
     */
    virtual float observation_loglik(const Observation& y_k,
                                       const State& x_k,
                                       float t_k) const = 0;
};

} // namespace PKF

#endif // PKF_STATE_SPACE_MODEL_HPP
