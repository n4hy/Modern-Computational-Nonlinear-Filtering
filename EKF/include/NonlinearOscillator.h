#ifndef NONLINEAR_OSCILLATOR_H
#define NONLINEAR_OSCILLATOR_H

#include "SystemModel.h"
#include <cmath>

/**
 * Nonlinear Oscillator Model.
 *
 * State x = [pos, vel]^T
 * Dynamics:
 *   d(pos)/dt = vel
 *   d(vel)/dt = -omega^2 * sin(pos) - damping * vel
 *
 * Discretized using Euler integration.
 *
 * Observations:
 *   y = pos + v_k
 */
class NonlinearOscillator : public SystemModel {
public:
    NonlinearOscillator(double dt = 0.01)
        : dt_(dt), omega_sq_(2.0), damping_(0.5),
          Q_(Eigen::MatrixXd::Identity(2, 2) * 0.001),
          R_(Eigen::MatrixXd::Identity(1, 1) * 0.1) {}

    // State: [pos, vel]
    Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        Eigen::VectorXd x_next(2);
        double pos = x(0);
        double vel = x(1);

        // Euler integration
        // pos_k+1 = pos_k + vel_k * dt
        // vel_k+1 = vel_k + (-omega^2 * sin(pos_k) - damping * vel_k) * dt

        x_next(0) = pos + vel * dt_;
        x_next(1) = vel + (-omega_sq_ * std::sin(pos) - damping_ * vel) * dt_;

        // Add control u if provided (assume scalar force on velocity)
        if (u.size() > 0) {
            x_next(1) += u(0) * dt_;
        }

        return x_next;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& x, double t) const override {
        Eigen::VectorXd y(1);
        y(0) = x(0); // Measure position
        return y;
    }

    Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        Eigen::MatrixXd F(2, 2);
        double pos = x(0);

        // df_0/dpos = 1, df_0/dvel = dt
        F(0, 0) = 1.0;
        F(0, 1) = dt_;

        // df_1/dpos = -omega^2 * cos(pos) * dt
        // df_1/dvel = 1 - damping * dt
        F(1, 0) = -omega_sq_ * std::cos(pos) * dt_;
        F(1, 1) = 1.0 - damping_ * dt_;

        return F;
    }

    Eigen::MatrixXd H(const Eigen::VectorXd& x, double t) const override {
        Eigen::MatrixXd H(1, 2);
        H << 1.0, 0.0;
        return H;
    }

    Eigen::MatrixXd Q(double t) const override {
        return Q_;
    }

    Eigen::MatrixXd R(double t) const override {
        return R_;
    }

    int getStateDim() const override { return 2; }
    int getObsDim() const override { return 1; }

private:
    double dt_;
    double omega_sq_;
    double damping_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};

#endif // NONLINEAR_OSCILLATOR_H
