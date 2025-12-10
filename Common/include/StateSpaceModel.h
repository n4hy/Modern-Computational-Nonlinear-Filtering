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

    using State       = Eigen::Matrix<double, NX, 1>;
    using Observation = Eigen::Matrix<double, NY, 1>;
    using StateMat    = Eigen::Matrix<double, NX, NX>;
    using ObsMat      = Eigen::Matrix<double, NY, NY>;

    virtual ~StateSpaceModel() = default;

    /**
     * Nonlinear State Transition Function: x_k = f(x_{k-1}, t_k, u_k) + w_k
     * Note: Process noise w_k is handled via Q().
     */
    virtual State f(const State& x_prev,
                    double t_k,
                    const Eigen::Ref<const State>& u_k) const = 0;

    /**
     * Nonlinear Observation Function: y_k = h(x_k, t_k) + v_k
     * Note: Measurement noise v_k is handled via R().
     */
    virtual Observation h(const State& x_k,
                          double t_k) const = 0;

    /**
     * Process Noise Covariance Matrix Q_k
     */
    virtual StateMat Q(double t_k) const = 0;

    /**
     * Measurement Noise Covariance Matrix R_k
     */
    virtual ObsMat R(double t_k) const = 0;
};

} // namespace UKFModel

#endif // STATE_SPACE_MODEL_H
