#ifndef SYSTEM_MODEL_H
#define SYSTEM_MODEL_H

#include <Eigen/Dense>

/**
 * Abstract base class for a Nonlinear System Model.
 * Defines the interface for state transition (f), observation (h),
 * and their respective Jacobians (F, H).
 *
 * Includes support for control inputs (u) and time-varying dynamics (t).
 */
class SystemModel {
public:
    virtual ~SystemModel() = default;

    /**
     * Nonlinear State Transition Function f(x, u, t)
     * x_k = f(x_{k-1}, u_k, t_k) + w_k
     */
    virtual Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const = 0;

    /**
     * Nonlinear Observation Function h(x, t)
     * y_k = h(x_k, t_k) + v_k
     */
    virtual Eigen::VectorXd h(const Eigen::VectorXd& x, double t) const = 0;

    /**
     * Jacobian of f w.r.t x, evaluated at (x, u, t).
     * F_k = df/dx | x_{k-1}, u_k, t_k
     */
    virtual Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const = 0;

    /**
     * Jacobian of h w.r.t x, evaluated at (x, t).
     * H_k = dh/dx | x_k, t_k
     */
    virtual Eigen::MatrixXd H(const Eigen::VectorXd& x, double t) const = 0;

    /**
     * Process Noise Covariance Matrix Q(t)
     */
    virtual Eigen::MatrixXd Q(double t) const = 0;

    /**
     * Measurement Noise Covariance Matrix R(t)
     */
    virtual Eigen::MatrixXd R(double t) const = 0;

    // Helper to get state dimension
    virtual int getStateDim() const = 0;

    // Helper to get observation dimension
    virtual int getObsDim() const = 0;
};

#endif // SYSTEM_MODEL_H
