#ifndef DRAG_BALL_MODEL_H
#define DRAG_BALL_MODEL_H

#include "SystemModel.h"
#include <iostream>

/**
 * 6D Ball with Air Drag.
 * State: [x, y, z, vx, vy, vz]
 * Dynamics:
 *   p_dot = v
 *   v_dot = -0.5 * rho * Cd * A * |v| * v / m - [0, 0, g] + w_wind
 *
 *   Simplified: v_dot = -beta * |v| * v - g + w
 *   where beta = 0.5 * rho * Cd * A / m
 */
class DragBallModel : public SystemModel {
public:
    DragBallModel(double dt, double beta, double q_std, double r_std)
        : dt_(dt), beta_(beta) {

        // Q: Process noise (Wind / forcing)
        // Applied primarily to velocity states
        Q_ = Eigen::MatrixXd::Identity(6, 6) * 1e-6; // Base small noise on pos
        Q_.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * (q_std * q_std);

        // R: Measurement noise on Position
        R_ = Eigen::MatrixXd::Identity(3, 3) * (r_std * r_std);
    }

    Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        // x = [px, py, pz, vx, vy, vz]
        Eigen::VectorXd pos = x.head(3);
        Eigen::VectorXd vel = x.tail(3);

        double speed = vel.norm();
        double g = 9.81;

        // Acceleration due to drag: a_d = -beta * speed * v
        Eigen::VectorXd acc_drag = -beta_ * speed * vel;

        // Acceleration due to gravity
        Eigen::VectorXd acc_grav(3);
        acc_grav << 0, 0, -g;

        Eigen::VectorXd acc = acc_drag + acc_grav;

        // Euler Integration (Semi-Implicit or Standard)
        // Standard:
        // p_new = p + v * dt + 0.5 * a * dt^2
        // v_new = v + a * dt

        Eigen::VectorXd x_next(6);
        x_next.head(3) = pos + vel * dt_ + 0.5 * acc * dt_ * dt_;
        x_next.tail(3) = vel + acc * dt_;

        return x_next;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& x, double t) const override {
        // Measure position only
        return x.head(3);
    }

    // UKF doesn't need F and H, but we must implement the interface.
    // We'll throw or return zeros. Returning zeros is safer to avoid crashes if logged.
    Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        // Jacobian of f is complex due to |v|*v term.
        // We strictly use UKF, so this shouldn't be called for filtering.
        return Eigen::MatrixXd::Zero(6, 6);
    }

    Eigen::MatrixXd H(const Eigen::VectorXd& x, double t) const override {
        Eigen::MatrixXd H_ = Eigen::MatrixXd::Zero(3, 6);
        H_(0, 0) = 1;
        H_(1, 1) = 1;
        H_(2, 2) = 1;
        return H_;
    }

    Eigen::MatrixXd Q(double t) const override { return Q_; }
    Eigen::MatrixXd R(double t) const override { return R_; }

    int getStateDim() const override { return 6; }
    int getObsDim() const override { return 3; }

private:
    double dt_;
    double beta_; // Drag factor
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};

#endif // DRAG_BALL_MODEL_H
