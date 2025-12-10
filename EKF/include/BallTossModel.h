#ifndef BALL_TOSS_MODEL_H
#define BALL_TOSS_MODEL_H

#include "SystemModel.h"

/**
 * Ball Toss Model (Example).
 * Included for backward compatibility, updated to new interface.
 */
class BallTossModel : public SystemModel {
public:
    BallTossModel(double dt = 0.1) : dt_(dt) {
        Q_ = Eigen::MatrixXd::Identity(4, 4) * 0.01;
        R_ = Eigen::MatrixXd::Identity(2, 2) * 0.1;
    }

    Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        // Simple Physics: x = [px, py, vx, vy]
        Eigen::VectorXd x_next = x;
        x_next(0) += x(2) * dt_;
        x_next(1) += x(3) * dt_;
        x_next(3) -= 9.81 * dt_; // Gravity
        return x_next;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& x, double t) const override {
        // Measure position
        Eigen::VectorXd y(2);
        y << x(0), x(1);
        return y;
    }

    Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double t) const override {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(4, 4);
        F(0, 2) = dt_;
        F(1, 3) = dt_;
        return F;
    }

    Eigen::MatrixXd H(const Eigen::VectorXd& x, double t) const override {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        return H;
    }

    Eigen::MatrixXd Q(double t) const override { return Q_; }
    Eigen::MatrixXd R(double t) const override { return R_; }

    int getStateDim() const override { return 4; }
    int getObsDim() const override { return 2; }

private:
    double dt_;
    Eigen::MatrixXd Q_, R_;
};

#endif
