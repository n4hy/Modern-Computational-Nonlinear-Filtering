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
    NonlinearOscillator(float dt = 0.01f)
        : dt_(dt), omega_sq_(2.0f), damping_(0.5f),
          Q_(Eigen::MatrixXf::Identity(2, 2) * 0.001f),
          R_(Eigen::MatrixXf::Identity(1, 1) * 0.1f) {}

    // State: [pos, vel]
    Eigen::VectorXf f(const Eigen::VectorXf& x, const Eigen::VectorXf& u, float t) const override {
        Eigen::VectorXf x_next(2);
        float pos = x(0);
        float vel = x(1);

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

    Eigen::VectorXf h(const Eigen::VectorXf& x, float t) const override {
        Eigen::VectorXf y(1);
        y(0) = x(0); // Measure position
        return y;
    }

    Eigen::MatrixXf F(const Eigen::VectorXf& x, const Eigen::VectorXf& u, float t) const override {
        Eigen::MatrixXf F(2, 2);
        float pos = x(0);

        // df_0/dpos = 1, df_0/dvel = dt
        F(0, 0) = 1.0f;
        F(0, 1) = dt_;

        // df_1/dpos = -omega^2 * cos(pos) * dt
        // df_1/dvel = 1 - damping * dt
        F(1, 0) = -omega_sq_ * std::cos(pos) * dt_;
        F(1, 1) = 1.0f - damping_ * dt_;

        return F;
    }

    Eigen::MatrixXf H(const Eigen::VectorXf& x, float t) const override {
        Eigen::MatrixXf H(1, 2);
        H << 1.0f, 0.0f;
        return H;
    }

    Eigen::MatrixXf Q(float t) const override {
        return Q_;
    }

    Eigen::MatrixXf R(float t) const override {
        return R_;
    }

    int getStateDim() const override { return 2; }
    int getObsDim() const override { return 1; }

private:
    float dt_;
    float omega_sq_;
    float damping_;
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;
};

#endif // NONLINEAR_OSCILLATOR_H
