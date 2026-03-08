/**
 * @file AircraftDynamicsModel.h
 * @brief 6-DOF Aircraft Dynamics with Dryden Turbulence
 *
 * Simplified aircraft dynamics model for navigation simulation.
 * Includes:
 * - Position, velocity, and attitude integration
 * - Gravity and Coriolis compensation
 * - Dryden wind turbulence effects
 * - Earth rotation effects (WGS84)
 *
 * Reference frames:
 * - NED (North-East-Down) for navigation
 * - Body frame aligned with aircraft axes
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "DrydenTurbulenceModel.h"
#include "INSErrorModel.h"

namespace AircraftNav {

// WGS84 constants
namespace WGS84 {
    constexpr double A = 6378137.0;              // Semi-major axis [m]
    constexpr double F = 1.0 / 298.257223563;    // Flattening
    constexpr double B = A * (1.0 - F);          // Semi-minor axis [m]
    constexpr double E2 = F * (2.0 - F);         // First eccentricity squared
    constexpr double OMEGA_E = 7.292115e-5;      // Earth rotation rate [rad/s]
    constexpr double GM = 3.986004418e14;        // Gravitational constant [m^3/s^2]
    constexpr double G0 = 9.7803253359;          // Gravity at equator [m/s^2]
}

/**
 * @brief Aircraft dynamics configuration
 */
struct AircraftDynamicsConfig {
    // Initial state
    double latitude = 40.0 * M_PI / 180.0;   // [rad]
    double longitude = -105.0 * M_PI / 180.0; // [rad]
    double altitude = 3048.0;                 // [m] (10,000 ft)

    double heading = 0.0;     // [rad] (North)
    double airspeed = 103.0;  // [m/s] (200 knots)

    // Aircraft parameters
    double mass = 5000.0;     // [kg]
    float max_bank_angle = 30.0f * M_PI / 180.0f;  // [rad]

    // Turbulence configuration
    bool enable_turbulence = true;
    DrydenConfig::Severity turbulence_severity = DrydenConfig::Severity::MODERATE;

    // INS configuration
    INSErrorConfig::Grade ins_grade = INSErrorConfig::Grade::TACTICAL;
};

/**
 * @brief Aircraft state vector for truth model
 */
struct AircraftState {
    // Position (geodetic)
    double lat;      // Latitude [rad]
    double lon;      // Longitude [rad]
    double alt;      // Altitude [m]

    // Velocity (NED)
    double v_n;      // North velocity [m/s]
    double v_e;      // East velocity [m/s]
    double v_d;      // Down velocity [m/s]

    // Attitude (Euler angles)
    double roll;     // Roll [rad]
    double pitch;    // Pitch [rad]
    double yaw;      // Yaw/Heading [rad]

    // Angular rates (body frame)
    double p;        // Roll rate [rad/s]
    double q;        // Pitch rate [rad/s]
    double r;        // Yaw rate [rad/s]

    /**
     * @brief Convert to 15-element navigation state vector
     * State order: [lat, lon, alt, vN, vE, vD, roll, pitch, yaw, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
     */
    Eigen::Matrix<float, 15, 1> toNavState(const Eigen::Vector3f& gyro_bias,
                                            const Eigen::Vector3f& accel_bias) const {
        Eigen::Matrix<float, 15, 1> x;
        x << static_cast<float>(lat),
             static_cast<float>(lon),
             static_cast<float>(alt),
             static_cast<float>(v_n),
             static_cast<float>(v_e),
             static_cast<float>(v_d),
             static_cast<float>(roll),
             static_cast<float>(pitch),
             static_cast<float>(yaw),
             gyro_bias,
             accel_bias;
        return x;
    }

    static AircraftState fromNavState(const Eigen::Matrix<float, 15, 1>& x) {
        AircraftState s;
        s.lat = x(0);
        s.lon = x(1);
        s.alt = x(2);
        s.v_n = x(3);
        s.v_e = x(4);
        s.v_d = x(5);
        s.roll = x(6);
        s.pitch = x(7);
        s.yaw = x(8);
        s.p = 0.0;
        s.q = 0.0;
        s.r = 0.0;
        return s;
    }
};

/**
 * @brief 6-DOF Aircraft Dynamics Simulator
 *
 * Integrates aircraft equations of motion with:
 * - Strapdown INS mechanization equations
 * - WGS84 Earth model
 * - Dryden turbulence
 * - Simplified autopilot for level flight
 */
class AircraftDynamicsModel {
public:
    explicit AircraftDynamicsModel(const AircraftDynamicsConfig& config, uint64_t seed = 0)
        : config_(config)
        , turbulence_(DrydenConfig::create(
              static_cast<float>(config.altitude),
              static_cast<float>(config.airspeed),
              config.turbulence_severity), seed)
        , ins_errors_(INSErrorConfig::create(config.ins_grade), seed + 1)
    {
        initializeState();
    }

    /**
     * @brief Propagate aircraft state forward by dt
     * @param dt Time step [s]
     */
    void propagate(double dt) {
        // Update turbulence
        if (config_.enable_turbulence) {
            wind_body_ = turbulence_.update(static_cast<float>(dt));
        }

        // Update INS biases
        ins_errors_.update(static_cast<float>(dt));

        // Simple autopilot: maintain level flight at constant airspeed
        // Generates commanded angular rates and accelerations
        double roll_cmd = 0.0;
        double pitch_cmd = 0.0;

        // Roll/pitch control (simple P controller)
        double roll_error = roll_cmd - state_.roll;
        double pitch_error = pitch_cmd - state_.pitch;
        state_.p = 0.5 * roll_error;
        state_.q = 0.5 * pitch_error;
        state_.r = 0.0;  // No commanded yaw rate

        // RK4 integration of equations of motion
        AircraftState k1, k2, k3, k4;
        k1 = computeDerivatives(state_);

        AircraftState s2 = addScaled(state_, k1, 0.5 * dt);
        k2 = computeDerivatives(s2);

        AircraftState s3 = addScaled(state_, k2, 0.5 * dt);
        k3 = computeDerivatives(s3);

        AircraftState s4 = addScaled(state_, k3, dt);
        k4 = computeDerivatives(s4);

        // Combine: x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        state_ = addScaled(state_, k1, dt / 6.0);
        state_ = addScaled(state_, k2, dt / 3.0);
        state_ = addScaled(state_, k3, dt / 3.0);
        state_ = addScaled(state_, k4, dt / 6.0);

        // Normalize angles
        normalizeAngles();

        time_ += dt;
    }

    /**
     * @brief Get true IMU measurements (before error corruption)
     *
     * For strapdown INS: IMU measures specific force f = a_inertial - g
     * This must include actual aircraft accelerations from dynamics.
     */
    void getTrueIMU(Eigen::Vector3f& gyro, Eigen::Vector3f& accel) const {
        // True angular rates in body frame
        gyro << static_cast<float>(state_.p),
                static_cast<float>(state_.q),
                static_cast<float>(state_.r);

        // getRotationNEDToBody() returns body-to-NED (naming is misleading)
        Eigen::Matrix3f R_body_to_ned = getRotationNEDToBody();
        Eigen::Matrix3f R_ned_to_body = R_body_to_ned.transpose();

        // Get wind contribution in NED
        Eigen::Vector3f wind_ned = R_body_to_ned * wind_body_;

        // Velocity magnitude
        double V = std::sqrt(state_.v_n * state_.v_n +
                             state_.v_e * state_.v_e +
                             state_.v_d * state_.v_d);
        double V_target = config_.airspeed;
        double drag_accel = 0.1 * (V - V_target);

        // Actual aircraft acceleration in NED (must match computeDerivatives)
        Eigen::Vector3f a_ned;
        a_ned(0) = static_cast<float>(-drag_accel * state_.v_n / (V + 1e-6) + wind_ned(0) * 0.01);
        a_ned(1) = static_cast<float>(-drag_accel * state_.v_e / (V + 1e-6) + wind_ned(1) * 0.01);
        // Level flight: v_d returns to zero, plus turbulence
        // ds.v_d = -0.5 * s.v_d + wind_ned(2) * 0.1
        a_ned(2) = static_cast<float>(-0.5 * state_.v_d + wind_ned(2) * 0.1);

        // Gravity in NED (pointing down = positive z)
        float g = computeGravity(static_cast<float>(state_.lat),
                                  static_cast<float>(state_.alt));
        Eigen::Vector3f g_ned(0.0f, 0.0f, g);

        // Specific force: f = a_inertial - g (in body frame)
        // IMU measures what's needed to produce the acceleration
        // Must rotate NED to body frame
        accel = R_ned_to_body * (a_ned - g_ned);
    }

    /**
     * @brief Get corrupted IMU measurements (with errors)
     */
    void getCorruptedIMU(Eigen::Vector3f& gyro, Eigen::Vector3f& accel) const {
        Eigen::Vector3f true_gyro, true_accel;
        getTrueIMU(true_gyro, true_accel);

        gyro = ins_errors_.corruptGyro(true_gyro);
        accel = ins_errors_.corruptAccel(true_accel);
    }

    /**
     * @brief Get rotation matrix from NED to body frame
     */
    Eigen::Matrix3f getRotationNEDToBody() const {
        float cr = std::cos(static_cast<float>(state_.roll));
        float sr = std::sin(static_cast<float>(state_.roll));
        float cp = std::cos(static_cast<float>(state_.pitch));
        float sp = std::sin(static_cast<float>(state_.pitch));
        float cy = std::cos(static_cast<float>(state_.yaw));
        float sy = std::sin(static_cast<float>(state_.yaw));

        Eigen::Matrix3f C_bn;
        C_bn << cp*cy,                cp*sy,                -sp,
                sr*sp*cy - cr*sy,     sr*sp*sy + cr*cy,     sr*cp,
                cr*sp*cy + sr*sy,     cr*sp*sy - sr*cy,     cr*cp;
        return C_bn;
    }

    /**
     * @brief Compute gravity magnitude at given position
     */
    static float computeGravity(float lat, float alt) {
        // Somigliana formula for gravity on ellipsoid
        float sin_lat = std::sin(lat);
        float sin2_lat = sin_lat * sin_lat;

        float g_0 = static_cast<float>(WGS84::G0) *
                    (1.0f + 0.0019318514f * sin2_lat) /
                    std::sqrt(1.0f - 0.00669438f * sin2_lat);

        // Free-air correction for altitude
        float g = g_0 * (1.0f - 2.0f * alt / static_cast<float>(WGS84::A));

        return g;
    }

    /**
     * @brief Compute meridian and transverse radii of curvature
     */
    static void computeRadii(float lat, float& R_M, float& R_N) {
        float sin_lat = std::sin(lat);
        float denom = std::sqrt(1.0f - static_cast<float>(WGS84::E2) * sin_lat * sin_lat);

        // Meridian radius
        R_M = static_cast<float>(WGS84::A * (1.0 - WGS84::E2)) / (denom * denom * denom);

        // Transverse radius
        R_N = static_cast<float>(WGS84::A) / denom;
    }

    // Accessors
    const AircraftState& state() const { return state_; }
    AircraftState& state() { return state_; }
    double time() const { return time_; }

    const INSErrorModel& insErrors() const { return ins_errors_; }
    const DrydenTurbulenceModel& turbulence() const { return turbulence_; }

    Eigen::Vector3f getWindBody() const { return wind_body_; }

    void reset() {
        initializeState();
        turbulence_.reset();
        ins_errors_.reset();
        time_ = 0.0;
    }

private:
    AircraftDynamicsConfig config_;
    AircraftState state_;
    DrydenTurbulenceModel turbulence_;
    INSErrorModel ins_errors_;
    Eigen::Vector3f wind_body_ = Eigen::Vector3f::Zero();
    double time_ = 0.0;

    void initializeState() {
        state_.lat = config_.latitude;
        state_.lon = config_.longitude;
        state_.alt = config_.altitude;

        // Initialize velocity from airspeed and heading
        state_.v_n = config_.airspeed * std::cos(config_.heading);
        state_.v_e = config_.airspeed * std::sin(config_.heading);
        state_.v_d = 0.0;

        state_.roll = 0.0;
        state_.pitch = 0.0;
        state_.yaw = config_.heading;

        state_.p = 0.0;
        state_.q = 0.0;
        state_.r = 0.0;
    }

    AircraftState computeDerivatives(const AircraftState& s) const {
        AircraftState ds;

        float lat = static_cast<float>(s.lat);
        float alt = static_cast<float>(s.alt);

        // Compute radii of curvature
        float R_M, R_N;
        computeRadii(lat, R_M, R_N);

        // Position derivatives
        ds.lat = s.v_n / (R_M + alt);
        ds.lon = s.v_e / ((R_N + alt) * std::cos(lat));
        ds.alt = -s.v_d;

        // Velocity derivatives (simplified - level flight with turbulence)
        // Add wind effects
        Eigen::Matrix3f C_nb = getRotationNEDToBody().transpose();
        Eigen::Vector3f wind_ned = C_nb * wind_body_;

        // Simple drag model to maintain airspeed
        double V = std::sqrt(s.v_n * s.v_n + s.v_e * s.v_e + s.v_d * s.v_d);
        double V_target = config_.airspeed;
        double drag_accel = 0.1 * (V - V_target);  // Simple speed regulation

        // Gravity
        float g = computeGravity(lat, alt);

        // Velocity derivatives (simplified)
        // For level flight: lift = weight, so net vertical force = 0
        // Aircraft maintains altitude through lift, not by accelerating
        ds.v_n = -drag_accel * s.v_n / (V + 1e-6) + wind_ned(0) * 0.01;
        ds.v_e = -drag_accel * s.v_e / (V + 1e-6) + wind_ned(1) * 0.01;
        // Level flight: v_d returns to zero (lift cancels gravity)
        // Add turbulence-induced vertical perturbations
        ds.v_d = -0.5 * s.v_d + wind_ned(2) * 0.1;

        // Attitude derivatives (kinematic equations)
        double cr = std::cos(s.roll);
        double sr = std::sin(s.roll);
        double cp = std::cos(s.pitch);
        double tp = std::tan(s.pitch);

        ds.roll = s.p + sr * tp * s.q + cr * tp * s.r;
        ds.pitch = cr * s.q - sr * s.r;
        ds.yaw = (sr * s.q + cr * s.r) / cp;

        // Angular rate derivatives (simplified - autopilot maintains level)
        ds.p = -0.5 * s.p + 0.1 * wind_body_(1);  // Roll damping + turbulence
        ds.q = -0.5 * s.q + 0.1 * wind_body_(2);  // Pitch damping + turbulence
        ds.r = -0.2 * s.r;  // Yaw damping

        return ds;
    }

    AircraftState addScaled(const AircraftState& s,
                            const AircraftState& ds,
                            double scale) const {
        AircraftState result;
        result.lat = s.lat + scale * ds.lat;
        result.lon = s.lon + scale * ds.lon;
        result.alt = s.alt + scale * ds.alt;
        result.v_n = s.v_n + scale * ds.v_n;
        result.v_e = s.v_e + scale * ds.v_e;
        result.v_d = s.v_d + scale * ds.v_d;
        result.roll = s.roll + scale * ds.roll;
        result.pitch = s.pitch + scale * ds.pitch;
        result.yaw = s.yaw + scale * ds.yaw;
        result.p = s.p + scale * ds.p;
        result.q = s.q + scale * ds.q;
        result.r = s.r + scale * ds.r;
        return result;
    }

    void normalizeAngles() {
        // Wrap longitude to [-π, π]
        while (state_.lon > M_PI) state_.lon -= 2.0 * M_PI;
        while (state_.lon < -M_PI) state_.lon += 2.0 * M_PI;

        // Wrap yaw to [0, 2π]
        while (state_.yaw >= 2.0 * M_PI) state_.yaw -= 2.0 * M_PI;
        while (state_.yaw < 0.0) state_.yaw += 2.0 * M_PI;

        // Clamp roll and pitch
        state_.roll = std::clamp(state_.roll, -M_PI / 2.0, M_PI / 2.0);
        state_.pitch = std::clamp(state_.pitch, -M_PI / 4.0, M_PI / 4.0);
    }
};

} // namespace AircraftNav
