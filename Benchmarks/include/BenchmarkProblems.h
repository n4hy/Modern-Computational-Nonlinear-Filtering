#ifndef BENCHMARK_PROBLEMS_H
#define BENCHMARK_PROBLEMS_H

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "StateSpaceModel.h"

namespace Benchmark {

/**
 * Problem 1: High-Dimensional Coupled Oscillators (10D)
 * 5 coupled nonlinear oscillators with damping and coupling
 * Highly nonlinear with coupling between all oscillators
 * Only observe positions (not velocities)
 */
template<int NX = 10, int NY = 5>
class CoupledOscillators : public UKFModel::StateSpaceModel<NX, NY> {
public:
    static constexpr int STATE_DIM = NX;
    static constexpr int OBS_DIM = NY;

    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.01f;           // Integration timestep
    float omega = 2.0f;         // Natural frequency
    float damping = 0.1f;       // Damping coefficient
    float coupling = 0.5f;      // Coupling strength
    float nonlinearity = 0.3f;  // Nonlinear coupling term

    State f(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        State x_next = x;

        // RK4 integration for each oscillator
        // State: [pos1, vel1, pos2, vel2, ..., pos5, vel5]
        for (int step = 0; step < 1; ++step) {
            State k1 = dynamics(x_next, t);
            State k2 = dynamics(x_next + 0.5f * dt * k1, t + 0.5f * dt);
            State k3 = dynamics(x_next + 0.5f * dt * k2, t + 0.5f * dt);
            State k4 = dynamics(x_next + dt * k3, t + dt);
            x_next = x_next + (dt / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);
        }

        return x_next;
    }

    Observation h(const State& x, float t) const override {
        Observation y;
        // Observe only positions with nonlinear measurement
        for (int i = 0; i < NY; ++i) {
            // Nonlinear observation: y = pos + 0.1*sin(pos)
            float pos = x(2*i);
            y(i) = pos + 0.1f * std::sin(pos);
        }
        return y;
    }

    StateMat Q(float t) const override {
        StateMat q = StateMat::Identity();
        // Lower noise on velocities, higher on positions
        for (int i = 0; i < NY; ++i) {
            q(2*i, 2*i) = 0.001f;      // Position noise
            q(2*i+1, 2*i+1) = 0.01f;   // Velocity noise
        }
        return q;
    }

    ObsMat R(float t) const override {
        return 0.1f * ObsMat::Identity();
    }

private:
    State dynamics(const State& x, float t) const {
        State dx = State::Zero();

        for (int i = 0; i < NY; ++i) {
            int pos_idx = 2*i;
            int vel_idx = 2*i + 1;

            float pos = x(pos_idx);
            float vel = x(vel_idx);

            // Compute coupling force from other oscillators
            float coupling_force = 0.0f;
            for (int j = 0; j < NY; ++j) {
                if (i != j) {
                    float other_pos = x(2*j);
                    // Linear + nonlinear coupling
                    coupling_force += coupling * (other_pos - pos);
                    coupling_force += nonlinearity * std::sin(other_pos - pos);
                }
            }

            // dx/dt = v
            dx(pos_idx) = vel;

            // dv/dt = -omega^2*sin(x) - damping*v + coupling_force
            dx(vel_idx) = -omega*omega * std::sin(pos) - damping * vel + coupling_force;
        }

        return dx;
    }
};

/**
 * Problem 2: Lorenz96 Model (High-dimensional chaotic system)
 * Used in weather prediction, highly chaotic and nonlinear
 * NX = 40 (40 state variables)
 * Observe only every 4th variable (10 observations)
 */
template<int NX = 40, int NY = 10>
class Lorenz96 : public UKFModel::StateSpaceModel<NX, NY> {
public:
    static constexpr int STATE_DIM = NX;
    static constexpr int OBS_DIM = NY;

    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.01f;  // Integration timestep
    float F = 8.0f;    // Forcing parameter (F=8 gives chaotic behavior)

    State f(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        State x_next = x;

        // RK4 integration
        State k1 = lorenz96_derivative(x_next);
        State k2 = lorenz96_derivative(x_next + 0.5f * dt * k1);
        State k3 = lorenz96_derivative(x_next + 0.5f * dt * k2);
        State k4 = lorenz96_derivative(x_next + dt * k3);
        x_next = x_next + (dt / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);

        return x_next;
    }

    Observation h(const State& x, float t) const override {
        Observation y;
        // Observe every 4th variable
        for (int i = 0; i < NY; ++i) {
            y(i) = x(i * 4);
        }
        return y;
    }

    StateMat Q(float t) const override {
        return 0.1f * StateMat::Identity();
    }

    ObsMat R(float t) const override {
        return 0.5f * ObsMat::Identity();
    }

private:
    State lorenz96_derivative(const State& x) const {
        State dx = State::Zero();

        for (int i = 0; i < NX; ++i) {
            int im2 = (i - 2 + NX) % NX;
            int im1 = (i - 1 + NX) % NX;
            int ip1 = (i + 1) % NX;

            // dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            dx(i) = (x(ip1) - x(im2)) * x(im1) - x(i) + F;
        }

        return dx;
    }
};

/**
 * Problem 3: Van der Pol Oscillator with Discontinuous Forcing
 * Highly nonlinear oscillator with discontinuous control input
 * 2D system with strong nonlinearity parameter
 */
template<int NX = 2, int NY = 1>
class VanDerPolDiscontinuous : public UKFModel::StateSpaceModel<NX, NY> {
public:
    static constexpr int STATE_DIM = NX;
    static constexpr int OBS_DIM = NY;

    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.01f;
    float mu = 5.0f;  // Nonlinearity parameter (large mu = very stiff)

    State f(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        State x_next = x;

        // RK4 integration
        State k1 = vdp_derivative(x_next, t);
        State k2 = vdp_derivative(x_next + 0.5f * dt * k1, t + 0.5f * dt);
        State k3 = vdp_derivative(x_next + 0.5f * dt * k2, t + 0.5f * dt);
        State k4 = vdp_derivative(x_next + dt * k3, t + dt);
        x_next = x_next + (dt / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);

        return x_next;
    }

    Observation h(const State& x, float t) const override {
        Observation y;
        // Nonlinear observation
        y(0) = x(0) + 0.2f * x(0) * x(0);
        return y;
    }

    StateMat Q(float t) const override {
        StateMat q = StateMat::Identity();
        q(0, 0) = 0.001f;
        q(1, 1) = 0.01f;
        return q;
    }

    ObsMat R(float t) const override {
        return 0.2f * ObsMat::Identity();
    }

private:
    State vdp_derivative(const State& x, float t) const {
        State dx;

        // Discontinuous forcing
        float forcing = (std::fmod(t, 2.0f) < 1.0f) ? 1.0f : -1.0f;

        // dx1/dt = x2
        dx(0) = x(1);

        // dx2/dt = mu*(1 - x1^2)*x2 - x1 + forcing
        dx(1) = mu * (1.0f - x(0)*x(0)) * x(1) - x(0) + forcing;

        return dx;
    }
};

/**
 * Problem 4: Reentry Vehicle Tracking (6D)
 * Spacecraft reentry with altitude-dependent drag
 * Highly nonlinear due to exponential atmosphere model
 * State: [x, y, z, vx, vy, vz]
 * Observe: [range, azimuth, elevation]
 */
template<int NX = 6, int NY = 3>
class ReentryVehicle : public UKFModel::StateSpaceModel<NX, NY> {
public:
    static constexpr int STATE_DIM = NX;
    static constexpr int OBS_DIM = NY;

    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.1f;
    float R0 = 6371000.0f;    // Earth radius (m)
    float H0 = 9000.0f;        // Scale height (m)
    float rho0 = 1.225f;       // Sea level density (kg/m^3)
    float BC = 500.0f;         // Ballistic coefficient (kg/m^2)
    float mu = 3.986004418e14f; // Earth gravitational parameter (m^3/s^2)

    // Radar position on Earth's surface (tracking station)
    Eigen::Vector3f radar_pos = Eigen::Vector3f(6371000.0f, 0.0f, 0.0f);

    State f(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        State x_next = x;

        // RK4 integration
        State k1 = dynamics(x_next);
        State k2 = dynamics(x_next + 0.5f * dt * k1);
        State k3 = dynamics(x_next + 0.5f * dt * k2);
        State k4 = dynamics(x_next + dt * k3);
        x_next = x_next + (dt / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);

        return x_next;
    }

    Observation h(const State& x, float t) const override {
        Observation y;

        // Position relative to radar
        Eigen::Vector3f pos(x(0), x(1), x(2));
        Eigen::Vector3f rel_pos = pos - radar_pos;

        // Range
        float range = rel_pos.norm();
        y(0) = range;

        // Guard against zero range (vehicle at radar position)
        if (range < 1e-6f) {
            y(1) = 0.0f;
            y(2) = 0.0f;
            return y;
        }

        // Azimuth (angle in x-y plane from x-axis)
        y(1) = std::atan2(rel_pos(1), rel_pos(0));

        // Elevation (angle from x-y plane)
        // Clamp argument to [-1,1] to prevent NaN from floating-point overshoot
        y(2) = std::asin(std::clamp(rel_pos(2) / range, -1.0f, 1.0f));

        return y;
    }

    StateMat Q(float t) const override {
        StateMat q = StateMat::Zero();
        // Process noise tuned for highly dynamic reentry
        // Position noise accounts for unmodeled dynamics
        for (int i = 0; i < 3; ++i) {
            q(i, i) = 100.0f;         // Position noise (m^2) - 10m std dev
            q(i+3, i+3) = 1000.0f;    // Velocity noise ((m/s)^2) - ~30m/s std dev
        }
        return q;
    }

    ObsMat R(float t) const override {
        ObsMat r = ObsMat::Identity();
        r(0, 0) = 10000.0f;                  // Range noise (m^2) - 100m std dev
        r(1, 1) = 0.0001f;                   // Azimuth noise (rad^2) - 0.01 rad std dev
        r(2, 2) = 0.0001f;                   // Elevation noise (rad^2) - 0.01 rad std dev
        return r;
    }

    // Return appropriate divergence threshold for this problem's scale
    float getDivergenceThreshold() const {
        return 5000.0f;  // 5km position or 5km/s velocity error
    }

private:
    State dynamics(const State& x) const {
        State dx;

        // Position derivatives = velocity
        dx(0) = x(3);
        dx(1) = x(4);
        dx(2) = x(5);

        // Altitude above Earth surface
        Eigen::Vector3f pos(x(0), x(1), x(2));
        float r = pos.norm();
        float altitude = r - R0;

        // Atmospheric density (exponential model)
        float rho = rho0 * std::exp(-altitude / H0);

        // Velocity vector
        Eigen::Vector3f vel(x(3), x(4), x(5));
        float speed = vel.norm();

        // Drag acceleration: -0.5 * rho * |v| / BC * v
        Eigen::Vector3f drag_acc = Eigen::Vector3f::Zero();
        if (speed > 1e-6f) {
            drag_acc = -0.5f * rho * speed / BC * vel;
        }

        // Gravity acceleration using gravitational parameter (altitude-dependent)
        // a_g = -mu / r^2 * (r_hat)
        // Reuse r from altitude computation above
        Eigen::Vector3f gravity_acc = Eigen::Vector3f::Zero();
        if (r > 1e-6f) {
            gravity_acc = -mu / (r * r * r) * pos;
        }

        // Total acceleration
        Eigen::Vector3f acc = drag_acc + gravity_acc;

        dx(3) = acc(0);
        dx(4) = acc(1);
        dx(5) = acc(2);

        return dx;
    }
};

/**
 * Problem 5: Bearing-Only Tracking (4D)
 * Track target using only bearing measurements
 * Highly nonlinear and challenging for observability
 * State: [x, y, vx, vy]
 * Observe: [bearing]
 */
template<int NX = 4, int NY = 1>
class BearingOnlyTracking : public UKFModel::StateSpaceModel<NX, NY> {
public:
    static constexpr int STATE_DIM = NX;
    static constexpr int OBS_DIM = NY;

    using State = Eigen::Matrix<float, NX, 1>;
    using Observation = Eigen::Matrix<float, NY, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using ObsMat = Eigen::Matrix<float, NY, NY>;

    float dt = 0.1f;
    float turn_rate = 0.1f;  // rad/s - observer platform turn rate

    // Observer position (moving in a circle)
    Eigen::Vector2f get_observer_pos(float t) const {
        float radius = 100.0f;
        return Eigen::Vector2f(radius * std::cos(turn_rate * t),
                               radius * std::sin(turn_rate * t));
    }

    State f(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        State x_next;
        // Constant velocity model with slight acceleration noise
        x_next(0) = x(0) + dt * x(2);
        x_next(1) = x(1) + dt * x(3);
        x_next(2) = x(2);
        x_next(3) = x(3);
        return x_next;
    }

    Observation h(const State& x, float t) const override {
        Observation y;

        // Observer position
        Eigen::Vector2f obs_pos = get_observer_pos(t);

        // Relative position
        Eigen::Vector2f rel_pos(x(0) - obs_pos(0), x(1) - obs_pos(1));

        // Bearing measurement
        y(0) = std::atan2(rel_pos(1), rel_pos(0));

        return y;
    }

    StateMat Q(float t) const override {
        StateMat q = StateMat::Zero();
        // Small acceleration noise
        q(2, 2) = 0.1f;
        q(3, 3) = 0.1f;
        return q;
    }

    ObsMat R(float t) const override {
        ObsMat r;
        r(0, 0) = 0.01f;  // Bearing noise (rad^2)
        return r;
    }
};

} // namespace Benchmark

#endif // BENCHMARK_PROBLEMS_H
