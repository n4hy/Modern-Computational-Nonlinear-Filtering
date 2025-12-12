#ifndef DRAG_BALL_MODEL_H
#define DRAG_BALL_MODEL_H

#include <cmath>
#include <random>
#include "StateSpaceModel.h"

/**
 * 4D State: [x, y, vx, vy]
 * 2D Obs: [x, y]
 *
 * Dynamics:
 *   x_k = x_{k-1} + vx_{k-1} * dt
 *   y_k = y_{k-1} + vy_{k-1} * dt
 *   vx_k = vx_{k-1} - (C * v * vx_{k-1}) * dt + w_vx
 *   vy_k = vy_{k-1} - (C * v * vy_{k-1}) * dt - g * dt + w_vy
 *
 *   v = sqrt(vx^2 + vy^2)
 */
class DragBallModel : public UKFModel::StateSpaceModel<4, 2> {
public:
    float dt;
    float drag_coeff; // C
    float g;          // Gravity

    // Noise parameters
    float q_std; // Process noise std dev (velocity)
    float r_std; // Measurement noise std dev (position)

    DragBallModel(float dt_val = 0.1f, float drag = 0.001f, float grav = 9.81f, float q = 0.1f, float r = 0.5f)
        : dt(dt_val), drag_coeff(drag), g(grav), q_std(q), r_std(r) {}

    State f(const State& x_prev, float t, const Eigen::Ref<const State>& u) const override {
        // x = [px, py, vx, vy]
        float px = x_prev(0);
        float py = x_prev(1);
        float vx = x_prev(2);
        float vy = x_prev(3);

        float v = std::sqrt(vx*vx + vy*vy);

        State x_next;
        x_next(0) = px + vx * dt;
        x_next(1) = py + vy * dt;
        x_next(2) = vx - (drag_coeff * v * vx) * dt;
        x_next(3) = vy - (drag_coeff * v * vy) * dt - g * dt;

        return x_next;
    }

    Observation h(const State& x_k, float t) const override {
        // Observe position only
        Observation y;
        y(0) = x_k(0);
        y(1) = x_k(1);
        return y;
    }

    StateMat Q(float t) const override {
        StateMat q_mat = StateMat::Zero();
        // Noise on velocity mainly
        float q_var = q_std * q_std;
        // Basic discrete noise model: Integral over dt...
        // For simplicity:
        // Position noise is small (integration of velocity noise), Velocity noise is q_var
        q_mat(0,0) = 0.01f * q_var;
        q_mat(1,1) = 0.01f * q_var;
        q_mat(2,2) = q_var * dt;
        q_mat(3,3) = q_var * dt;
        return q_mat;
    }

    ObsMat R(float t) const override {
        ObsMat r_mat = ObsMat::Identity();
        r_mat *= (r_std * r_std);
        return r_mat;
    }
};

#endif // DRAG_BALL_MODEL_H
