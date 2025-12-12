#ifndef BALL_TOSS_MODEL_H
#define BALL_TOSS_MODEL_H

#include "SystemModel.h"

/**
 * Ball Toss Model (Example).
 * Included for backward compatibility, updated to new interface.
 */
class BallTossModel : public SystemModel {
public:
    BallTossModel(float dt = 0.1f) : dt_(dt) {
        Q_ = Eigen::MatrixXf::Identity(4, 4) * 0.01f;
        R_ = Eigen::MatrixXf::Identity(2, 2) * 0.1f;
    }

    Eigen::VectorXf f(const Eigen::VectorXf& x, const Eigen::VectorXf& u, float t) const override {
        // Simple Physics: x = [px, py, vx, vy]
        Eigen::VectorXf x_next = x;
        x_next(0) += x(2) * dt_;
        x_next(1) += x(3) * dt_;
        x_next(3) -= 9.81f * dt_; // Gravity
        return x_next;
    }

    Eigen::VectorXf h(const Eigen::VectorXf& x, float t) const override {
        // Measure position
        Eigen::VectorXf y(2);
        y << x(0), x(1);
        return y;
    }

    Eigen::MatrixXf F(const Eigen::VectorXf& x, const Eigen::VectorXf& u, float t) const override {
        Eigen::MatrixXf F = Eigen::MatrixXf::Identity(4, 4);
        F(0, 2) = dt_;
        F(1, 3) = dt_;
        return F;
    }

    Eigen::MatrixXf H(const Eigen::VectorXf& x, float t) const override {
        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(2, 4);
        H(0, 0) = 1.0f;
        H(1, 1) = 1.0f;
        return H;
    }

    Eigen::MatrixXf Q(float t) const override { return Q_; }
    Eigen::MatrixXf R(float t) const override { return R_; }

    int getStateDim() const override { return 4; }
    int getObsDim() const override { return 2; }

private:
    float dt_;
    Eigen::MatrixXf Q_, R_;
};

#endif
