#ifndef SYSTEM_MODEL_H
#define SYSTEM_MODEL_H

#include <Eigen/Dense>

/**
 * Abstract base class for a Nonlinear System Model.
 * Defines the interface for state transition (f), observation (h),
 * and their respective Jacobians (F, H).
 */
class SystemModel {
public:
    virtual ~SystemModel() = default;

    /**
     * Nonlinear State Transition Function f(x, u)
     * x_k+1 = f(x_k, u_k) + w_k
     */
    virtual Eigen::VectorXd f(const Eigen::VectorXd& x) const = 0;

    /**
     * Nonlinear Observation Function h(x)
     * y_k = h(x_k) + v_k
     */
    virtual Eigen::VectorXd h(const Eigen::VectorXd& x) const = 0;

    /**
     * Jacobian of f w.r.t x, evaluated at x.
     * F_k = df/dx | x_k
     */
    virtual Eigen::MatrixXd F(const Eigen::VectorXd& x) const = 0;

    /**
     * Jacobian of h w.r.t x, evaluated at x.
     * H_k = dh/dx | x_k
     */
    virtual Eigen::MatrixXd H(const Eigen::VectorXd& x) const = 0;

    /**
     * Process Noise Covariance Matrix Q
     */
    virtual Eigen::MatrixXd Q() const = 0;

    /**
     * Measurement Noise Covariance Matrix R
     */
    virtual Eigen::MatrixXd R() const = 0;

    // Helper to get state dimension
    virtual int getStateDim() const = 0;

    // Helper to get observation dimension
    virtual int getObsDim() const = 0;
};

#endif // SYSTEM_MODEL_H
