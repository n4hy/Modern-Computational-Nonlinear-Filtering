#ifndef STATE_SPACE_MODEL_H
#define STATE_SPACE_MODEL_H

#include <Eigen/Dense>

namespace UKFModel {

/**
 * Templated Abstract Base Class for Nonlinear State-Space Models.
 *
 * NX: State Dimension
 * NY: Observation Dimension
 */
template<int NX, int NY>
class StateSpaceModel {
public:
    static constexpr int StateDim = NX;
    static constexpr int ObsDim = NY;

    using State       = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat    = Eigen::Matrix<float, NX, NX>;
    using ObsMat      = Eigen::Matrix<float, NY, NY>;

    virtual ~StateSpaceModel() = default;

    /**
     * Nonlinear State Transition Function: x_k = f(x_{k-1}, t_k, u_k) + w_k
     * Note: Process noise w_k is handled via Q().
     */
    virtual State f(const State& x_prev,
                    float t_k,
                    const Eigen::Ref<const State>& u_k) const = 0;

    /**
     * Nonlinear Observation Function: y_k = h(x_k, t_k) + v_k
     * Note: Measurement noise v_k is handled via R().
     */
    virtual Observation h(const State& x_k,
                          float t_k) const = 0;

    /**
     * Process Noise Covariance Matrix Q_k
     */
    virtual StateMat Q(float t_k) const = 0;

    /**
     * Measurement Noise Covariance Matrix R_k
     */
    virtual ObsMat R(float t_k) const = 0;

    /**
     * Returns true if state index i is an angular state (requires circular mean).
     * Default implementation returns false for all states.
     * Override for models with angular states (e.g., attitude).
     */
    virtual bool isAngularState(int i) const { return false; }

    /**
     * Returns true if observation index i is an angular observation.
     * Default implementation returns false for all observations.
     * Override for models with angular observations (e.g., AOA measurements).
     */
    virtual bool isAngularObservation(int i) const { return false; }
};

} // namespace UKFModel

#endif // STATE_SPACE_MODEL_H
