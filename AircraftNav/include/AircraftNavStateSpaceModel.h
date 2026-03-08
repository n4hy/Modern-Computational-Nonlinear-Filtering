/**
 * @file AircraftNavStateSpaceModel.h
 * @brief 15-State Aircraft Navigation State-Space Model for SRUKF
 *
 * State vector (15 states):
 *   [0-2]  Position: lat, lon, alt [rad, rad, m]
 *   [3-5]  Velocity: vN, vE, vD [m/s]
 *   [6-8]  Attitude: roll, pitch, yaw [rad]
 *   [9-11] Gyro bias: bg_x, bg_y, bg_z [rad/s]
 *   [12-14] Accel bias: ba_x, ba_y, ba_z [m/s^2]
 *
 * Measurement models:
 *   GPS (NY=6): [lat, lon, alt, vN, vE, vD]
 *   Iridium (NY=3 per satellite): [azimuth, elevation, doppler]
 */

#pragma once

#include <Eigen/Dense>
#include "StateSpaceModel.h"
#include "AircraftDynamicsModel.h"

namespace AircraftNav {

// State indices
enum StateIdx {
    LAT = 0, LON = 1, ALT = 2,
    VN = 3, VE = 4, VD = 5,
    ROLL = 6, PITCH = 7, YAW = 8,
    BG_X = 9, BG_Y = 10, BG_Z = 11,
    BA_X = 12, BA_Y = 13, BA_Z = 14
};

/**
 * @brief Common IMU measurement struct used by all models
 */
struct IMUMeasurement {
    Eigen::Vector3f gyro;   // Angular rate [rad/s]
    Eigen::Vector3f accel;  // Specific force [m/s^2]
};

/**
 * @brief GPS measurement model (6 observations)
 *
 * Direct observation of position and velocity in navigation frame.
 * R_gps: diag([4m, 4m, 6m, 0.1m/s, 0.1m/s, 0.15m/s])^2
 */
class GPSMeasurementModel : public UKFModel::StateSpaceModel<15, 6> {
public:
    using State = Eigen::Matrix<float, 15, 1>;
    using Observation = Eigen::Matrix<float, 6, 1>;
    using StateMat = Eigen::Matrix<float, 15, 15>;
    using ObsMat = Eigen::Matrix<float, 6, 6>;

    GPSMeasurementModel() {
        // GPS measurement noise covariance
        R_gps_ = ObsMat::Zero();

        // Position noise (convert from meters to radians for lat/lon)
        float R_M = 6371000.0f;  // Approximate Earth radius
        float sigma_lat = 4.0f / R_M;  // 4m in radians
        float sigma_lon = 4.0f / R_M;  // 4m in radians
        // Altitude: GPS + topographic map gives accurate aerial height (AGL)
        float sigma_alt = 1.0f;         // 1m (GPS + topo map)

        R_gps_(0, 0) = sigma_lat * sigma_lat;
        R_gps_(1, 1) = sigma_lon * sigma_lon;
        R_gps_(2, 2) = sigma_alt * sigma_alt;

        // Velocity noise
        R_gps_(3, 3) = 0.1f * 0.1f;   // 0.1 m/s North
        R_gps_(4, 4) = 0.1f * 0.1f;   // 0.1 m/s East
        R_gps_(5, 5) = 0.15f * 0.15f; // 0.15 m/s Down
    }

    State f(const State& x_prev, float t_k,
            const Eigen::Ref<const State>& u_k) const override {
        // Process model is handled by the navigation model
        // This shouldn't be called directly for GPS-only model
        return x_prev;
    }

    Observation h(const State& x_k, float t_k) const override {
        Observation y;
        y(0) = x_k(LAT);
        y(1) = x_k(LON);
        y(2) = x_k(ALT);
        y(3) = x_k(VN);
        y(4) = x_k(VE);
        y(5) = x_k(VD);
        return y;
    }

    StateMat Q(float t_k) const override {
        // Return zero - Q is handled by navigation model
        return StateMat::Zero();
    }

    ObsMat R(float t_k) const override {
        return R_gps_;
    }

private:
    ObsMat R_gps_;
};

/**
 * @brief Iridium AOA+Doppler measurement model (3 observations per satellite)
 *
 * Observation: [azimuth, elevation, doppler]
 * - Azimuth: atan2(sat_E, sat_N) in aircraft ENU frame [rad]
 * - Elevation: asin(sat_U / range) [rad]
 * - Doppler: -v_radial / λ [Hz] where λ = c/1626e6 = 0.1844m
 *
 * Noise (SNR-dependent):
 * - σ_az: 2-5° (0.035-0.087 rad)
 * - σ_el: 3-8° (0.052-0.14 rad)
 * - σ_doppler: 10-30 Hz
 */
class IridiumMeasurementModel : public UKFModel::StateSpaceModel<15, 3> {
public:
    using State = Eigen::Matrix<float, 15, 1>;
    using Observation = Eigen::Matrix<float, 3, 1>;
    using StateMat = Eigen::Matrix<float, 15, 15>;
    using ObsMat = Eigen::Matrix<float, 3, 3>;

    // Iridium constants
    static constexpr float IRIDIUM_FREQ = 1626e6f;      // Hz
    static constexpr float IRIDIUM_WAVELENGTH = 0.1844f; // m
    static constexpr float IRIDIUM_ALTITUDE = 780000.0f; // m

    /**
     * @brief Satellite position for measurement computation
     */
    struct SatellitePosition {
        float lat;   // [rad]
        float lon;   // [rad]
        float alt;   // [m]
        float v_lat; // [rad/s]
        float v_lon; // [rad/s]
        float v_alt; // [m/s]
    };

    IridiumMeasurementModel(float snr_db = 15.0f) {
        updateNoiseFromSNR(snr_db);
    }

    /**
     * @brief Set IMU measurements for next propagation
     */
    void setIMU(const IMUMeasurement& imu) {
        imu_ = imu;
    }

    /**
     * @brief Set timestep for process model
     */
    void setTimestep(float dt) {
        dt_ = dt;
    }

    /**
     * @brief Update measurement noise based on SNR
     */
    void updateNoiseFromSNR(float snr_db) {
        // Higher SNR = lower noise
        // Typical values: SNR 10-20 dB
        float snr_factor = std::pow(10.0f, -snr_db / 20.0f);

        // Base noise at 15 dB SNR: 3°, 5°, 15 Hz
        float sigma_az = (3.0f * M_PI / 180.0f) * snr_factor * 10.0f;
        float sigma_el = (5.0f * M_PI / 180.0f) * snr_factor * 10.0f;
        float sigma_doppler = 15.0f * snr_factor * 10.0f;

        // Clamp to reasonable ranges
        sigma_az = std::clamp(sigma_az, 0.035f, 0.087f);  // 2-5°
        sigma_el = std::clamp(sigma_el, 0.052f, 0.14f);   // 3-8°
        sigma_doppler = std::clamp(sigma_doppler, 10.0f, 30.0f);

        R_iridium_ = ObsMat::Zero();
        R_iridium_(0, 0) = sigma_az * sigma_az;
        R_iridium_(1, 1) = sigma_el * sigma_el;
        R_iridium_(2, 2) = sigma_doppler * sigma_doppler;

        snr_db_ = snr_db;
    }

    /**
     * @brief Set satellite position for measurement computation
     */
    void setSatellitePosition(const SatellitePosition& sat) {
        satellite_ = sat;
    }

    State f(const State& x_prev, float t_k,
            const Eigen::Ref<const State>& u_k) const override {
        State x = x_prev;

        float lat = x(LAT);
        float lon = x(LON);
        float alt = x(ALT);
        float v_n = x(VN);
        float v_e = x(VE);
        float v_d = x(VD);
        float roll = x(ROLL);
        float pitch = x(PITCH);
        float yaw = x(YAW);

        // Extract biases
        Eigen::Vector3f b_g(x(BG_X), x(BG_Y), x(BG_Z));
        Eigen::Vector3f b_a(x(BA_X), x(BA_Y), x(BA_Z));

        // Correct IMU measurements
        Eigen::Vector3f omega = imu_.gyro - b_g;
        Eigen::Vector3f f_b = imu_.accel - b_a;

        // Compute radii of curvature
        float R_M, R_N;
        AircraftDynamicsModel::computeRadii(lat, R_M, R_N);

        // Position rate
        float lat_dot = v_n / (R_M + alt);
        float lon_dot = v_e / ((R_N + alt) * std::cos(lat));
        float alt_dot = -v_d;

        // Rotation matrix body to NED
        Eigen::Matrix3f C_bn = getRotationBodyToNED(roll, pitch, yaw);

        // Specific force in NED
        Eigen::Vector3f f_n = C_bn * f_b;

        // Gravity
        float g = AircraftDynamicsModel::computeGravity(lat, alt);

        // Earth rotation rate in NED
        float omega_e = static_cast<float>(WGS84::OMEGA_E);
        Eigen::Vector3f omega_ie_n(omega_e * std::cos(lat), 0.0f, -omega_e * std::sin(lat));

        // Transport rate
        Eigen::Vector3f omega_en_n(
            v_e / (R_N + alt),
            -v_n / (R_M + alt),
            -v_e * std::tan(lat) / (R_N + alt)
        );

        // Velocity rate (with Coriolis)
        Eigen::Vector3f v_dot = f_n - (2.0f * omega_ie_n + omega_en_n).cross(
            Eigen::Vector3f(v_n, v_e, v_d)) + Eigen::Vector3f(0.0f, 0.0f, g);

        // Integrate with simple Euler (small dt assumed)
        x(LAT) = lat + lat_dot * dt_;
        x(LON) = lon + lon_dot * dt_;
        x(ALT) = alt + alt_dot * dt_;
        x(VN) = v_n + v_dot(0) * dt_;
        x(VE) = v_e + v_dot(1) * dt_;
        x(VD) = v_d + v_dot(2) * dt_;

        // Attitude integration using Euler kinematic equations
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cr = std::cos(roll);
        float sr = std::sin(roll);

        // Protect against gimbal lock (pitch = ±90°)
        if (std::abs(cp) > 1e-6f) {
            float tp = sp / cp;
            float roll_dot = omega(0) + omega(1) * sr * tp + omega(2) * cr * tp;
            float pitch_dot = omega(1) * cr - omega(2) * sr;
            float yaw_dot = (omega(1) * sr + omega(2) * cr) / cp;

            x(ROLL) = roll + roll_dot * dt_;
            x(PITCH) = pitch + pitch_dot * dt_;
            x(YAW) = yaw + yaw_dot * dt_;
        } else {
            // Near gimbal lock, use simplified integration
            x(ROLL) = roll + omega(0) * dt_;
            x(PITCH) = pitch + omega(1) * dt_;
            x(YAW) = yaw + omega(2) * dt_;
        }

        // Biases: random walk (no change in prediction)
        // x(BG_X), x(BG_Y), x(BG_Z), x(BA_X), x(BA_Y), x(BA_Z) unchanged

        return x;
    }

    Observation h(const State& x_k, float t_k) const override {
        Observation y;

        // Aircraft position and attitude
        float ac_lat = x_k(LAT);
        float ac_lon = x_k(LON);
        float ac_alt = x_k(ALT);
        float roll = x_k(ROLL);
        float pitch = x_k(PITCH);
        float yaw = x_k(YAW);

        // Convert both to ECEF for accurate geometry
        Eigen::Vector3f ac_ecef = geodeticToECEF(ac_lat, ac_lon, ac_alt);
        Eigen::Vector3f sat_ecef = geodeticToECEF(satellite_.lat, satellite_.lon, satellite_.alt);

        // Line of sight vector (aircraft to satellite)
        Eigen::Vector3f los = sat_ecef - ac_ecef;
        float range = los.norm();
        Eigen::Vector3f los_hat = los / range;

        // Transform ECEF to NED
        Eigen::Matrix3f R_ecef_to_ned = getECEFtoNEDRotation(ac_lat, ac_lon);
        Eigen::Vector3f los_ned = R_ecef_to_ned * los_hat;

        // Transform NED to body frame (matching antenna model)
        Eigen::Matrix3f R_ned_to_body = getNEDtoBodyRotation(roll, pitch, yaw);
        Eigen::Vector3f los_body = R_ned_to_body * los_ned;

        // Azimuth in body frame (from nose/x-axis, clockwise positive)
        float az = std::atan2(los_body(1), los_body(0));
        // Keep in [-π, π] to match measurement
        while (az > M_PI) az -= 2.0f * M_PI;
        while (az < -M_PI) az += 2.0f * M_PI;

        // Elevation in body frame (from horizontal plane)
        float el = std::asin(std::clamp(-los_body(2), -1.0f, 1.0f));

        // Doppler: f_d = -v_radial / λ
        // Compute radial velocity
        float R_M = 6371000.0f;
        float v_sat_n = satellite_.v_lat * (R_M + satellite_.alt);
        float v_sat_e = satellite_.v_lon * (R_M + satellite_.alt) * std::cos(satellite_.lat);
        float v_sat_d = -satellite_.v_alt;

        float v_ac_n = x_k(VN);
        float v_ac_e = x_k(VE);
        float v_ac_d = x_k(VD);

        // Relative velocity
        float dv_n = v_sat_n - v_ac_n;
        float dv_e = v_sat_e - v_ac_e;
        float dv_d = v_sat_d - v_ac_d;

        // Transform to ECEF (R_ned_to_ecef = R_ecef_to_ned^T)
        Eigen::Vector3f dv_ned(dv_n, dv_e, dv_d);
        Eigen::Matrix3f R_ned_to_ecef = R_ecef_to_ned.transpose();
        Eigen::Vector3f dv_ecef = R_ned_to_ecef * dv_ned;

        // Radial velocity (dot product with LOS)
        float v_radial = dv_ecef.dot(los_hat);

        // Doppler (positive = approaching)
        float doppler = -v_radial / IRIDIUM_WAVELENGTH;

        y(0) = az;
        y(1) = el;
        y(2) = doppler;

        return y;
    }

    StateMat Q(float t_k) const override {
        return StateMat::Zero();
    }

    ObsMat R(float t_k) const override {
        return R_iridium_;
    }

    /**
     * @brief Returns true for attitude states (ROLL, PITCH, YAW)
     * These require circular mean in UKF sigma point averaging.
     */
    bool isAngularState(int i) const override {
        return (i == ROLL || i == PITCH || i == YAW);
    }

    float getSnrDb() const { return snr_db_; }

private:
    ObsMat R_iridium_;
    SatellitePosition satellite_;
    float snr_db_ = 15.0f;
    IMUMeasurement imu_;
    float dt_ = 0.01f;

    static Eigen::Matrix3f getRotationBodyToNED(float roll, float pitch, float yaw) {
        float cr = std::cos(roll);
        float sr = std::sin(roll);
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cy = std::cos(yaw);
        float sy = std::sin(yaw);

        Eigen::Matrix3f C;
        C(0, 0) = cp * cy;
        C(0, 1) = sr * sp * cy - cr * sy;
        C(0, 2) = cr * sp * cy + sr * sy;
        C(1, 0) = cp * sy;
        C(1, 1) = sr * sp * sy + cr * cy;
        C(1, 2) = cr * sp * sy - sr * cy;
        C(2, 0) = -sp;
        C(2, 1) = sr * cp;
        C(2, 2) = cr * cp;
        return C;
    }

    static Eigen::Vector3f geodeticToECEF(float lat, float lon, float alt) {
        float sin_lat = std::sin(lat);
        float cos_lat = std::cos(lat);
        float sin_lon = std::sin(lon);
        float cos_lon = std::cos(lon);

        float N = static_cast<float>(WGS84::A) /
                  std::sqrt(1.0f - static_cast<float>(WGS84::E2) * sin_lat * sin_lat);

        Eigen::Vector3f ecef;
        ecef(0) = (N + alt) * cos_lat * cos_lon;
        ecef(1) = (N + alt) * cos_lat * sin_lon;
        ecef(2) = (N * (1.0f - static_cast<float>(WGS84::E2)) + alt) * sin_lat;

        return ecef;
    }

    static Eigen::Matrix3f getECEFtoNEDRotation(float lat, float lon) {
        // R_ecef_ned transforms ECEF vectors to NED frame
        // NED: x=North, y=East, z=Down
        float sin_lat = std::sin(lat);
        float cos_lat = std::cos(lat);
        float sin_lon = std::sin(lon);
        float cos_lon = std::cos(lon);

        Eigen::Matrix3f R;
        R << -sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat,
             -sin_lon,            cos_lon,             0.0f,
             -cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat;
        return R;
    }

    static Eigen::Matrix3f getNEDtoBodyRotation(float roll, float pitch, float yaw) {
        // R_ned_body transforms NED vectors to body frame
        // Body: x=forward, y=right, z=down
        // Rotation order: yaw -> pitch -> roll (3-2-1 Euler)
        float cr = std::cos(roll);
        float sr = std::sin(roll);
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cy = std::cos(yaw);
        float sy = std::sin(yaw);

        Eigen::Matrix3f R;
        R << cp*cy,                cp*sy,                -sp,
             sr*sp*cy - cr*sy,     sr*sp*sy + cr*cy,     sr*cp,
             cr*sp*cy + sr*sy,     cr*sp*sy - sr*cy,     cr*cp;
        return R;
    }
};

/**
 * @brief 15-State Strapdown INS Navigation Model
 *
 * Process model implements strapdown INS mechanization:
 * - Position integration with Earth curvature
 * - Velocity integration with Coriolis and transport rate
 * - Attitude integration from corrected gyro
 * - Bias random walk
 */
class AircraftNavStateSpaceModel : public UKFModel::StateSpaceModel<15, 6> {
public:
    using State = Eigen::Matrix<float, 15, 1>;
    using Observation = Eigen::Matrix<float, 6, 1>;
    using StateMat = Eigen::Matrix<float, 15, 15>;
    using ObsMat = Eigen::Matrix<float, 6, 6>;

    // Use common IMUMeasurement from namespace level

    AircraftNavStateSpaceModel() {
        initializeProcessNoise();
        gps_model_ = std::make_unique<GPSMeasurementModel>();
    }

    /**
     * @brief Set IMU measurements for next propagation
     */
    void setIMU(const IMUMeasurement& imu) {
        imu_ = imu;
    }

    /**
     * @brief Set timestep for process noise scaling
     */
    void setTimestep(float dt) {
        dt_ = dt;
    }

    State f(const State& x_prev, float t_k,
            const Eigen::Ref<const State>& u_k) const override {
        State x = x_prev;

        float lat = x(LAT);
        float lon = x(LON);
        float alt = x(ALT);
        float v_n = x(VN);
        float v_e = x(VE);
        float v_d = x(VD);
        float roll = x(ROLL);
        float pitch = x(PITCH);
        float yaw = x(YAW);

        // Extract biases
        Eigen::Vector3f b_g(x(BG_X), x(BG_Y), x(BG_Z));
        Eigen::Vector3f b_a(x(BA_X), x(BA_Y), x(BA_Z));

        // Correct IMU measurements
        Eigen::Vector3f omega = imu_.gyro - b_g;
        Eigen::Vector3f f_b = imu_.accel - b_a;

        // Compute radii of curvature
        float R_M, R_N;
        AircraftDynamicsModel::computeRadii(lat, R_M, R_N);

        // Position rate
        float lat_dot = v_n / (R_M + alt);
        float lon_dot = v_e / ((R_N + alt) * std::cos(lat));
        float alt_dot = -v_d;

        // Rotation matrix body to NED
        Eigen::Matrix3f C_bn = getRotationBodyToNED(roll, pitch, yaw);

        // Specific force in NED
        Eigen::Vector3f f_n = C_bn * f_b;

        // Gravity
        float g = AircraftDynamicsModel::computeGravity(lat, alt);

        // Earth rotation rate in NED
        float omega_e = static_cast<float>(WGS84::OMEGA_E);
        Eigen::Vector3f omega_ie_n(omega_e * std::cos(lat), 0.0f, -omega_e * std::sin(lat));

        // Transport rate
        Eigen::Vector3f omega_en_n(
            v_e / (R_N + alt),
            -v_n / (R_M + alt),
            -v_e * std::tan(lat) / (R_N + alt)
        );

        // Velocity rate (with Coriolis)
        Eigen::Vector3f v_dot = f_n - (2.0f * omega_ie_n + omega_en_n).cross(
            Eigen::Vector3f(v_n, v_e, v_d)) + Eigen::Vector3f(0.0f, 0.0f, g);

        // Integrate with simple Euler (small dt assumed)
        x(LAT) = lat + lat_dot * dt_;
        x(LON) = lon + lon_dot * dt_;
        x(ALT) = alt + alt_dot * dt_;
        x(VN) = v_n + v_dot(0) * dt_;
        x(VE) = v_e + v_dot(1) * dt_;
        x(VD) = v_d + v_dot(2) * dt_;

        // Attitude integration using Euler kinematic equations
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cr = std::cos(roll);
        float sr = std::sin(roll);

        // Protect against gimbal lock (pitch = ±90°)
        if (std::abs(cp) > 1e-6f) {
            float tp = sp / cp;
            float roll_dot = omega(0) + omega(1) * sr * tp + omega(2) * cr * tp;
            float pitch_dot = omega(1) * cr - omega(2) * sr;
            float yaw_dot = (omega(1) * sr + omega(2) * cr) / cp;

            x(ROLL) = roll + roll_dot * dt_;
            x(PITCH) = pitch + pitch_dot * dt_;
            x(YAW) = yaw + yaw_dot * dt_;
        } else {
            // Near gimbal lock, use simplified integration
            x(ROLL) = roll + omega(0) * dt_;
            x(PITCH) = pitch + omega(1) * dt_;
            x(YAW) = yaw + omega(2) * dt_;
        }

        // Biases unchanged (random walk handled in Q)
        // x(BG_X), x(BG_Y), x(BG_Z), x(BA_X), x(BA_Y), x(BA_Z) stay the same

        // Normalize angles
        // Use [-π, π] for yaw to avoid wrap issues at 0/2π boundary
        while (x(LON) > M_PI) x(LON) -= 2.0f * M_PI;
        while (x(LON) < -M_PI) x(LON) += 2.0f * M_PI;
        while (x(YAW) > M_PI) x(YAW) -= 2.0f * M_PI;
        while (x(YAW) < -M_PI) x(YAW) += 2.0f * M_PI;

        return x;
    }

    Observation h(const State& x_k, float t_k) const override {
        return gps_model_->h(x_k, t_k);
    }

    StateMat Q(float t_k) const override {
        StateMat Q = Q_base_;
        // Scale by timestep
        Q *= dt_;
        return Q;
    }

    ObsMat R(float t_k) const override {
        return gps_model_->R(t_k);
    }

    /**
     * @brief Returns true for attitude states (ROLL, PITCH, YAW)
     * These require circular mean in UKF sigma point averaging.
     */
    bool isAngularState(int i) const override {
        return (i == ROLL || i == PITCH || i == YAW);
    }

    /**
     * @brief Get initial covariance matrix
     */
    static StateMat getInitialCovariance() {
        StateMat P0 = StateMat::Zero();

        float R_M = 6371000.0f;

        // Position uncertainty (10m)
        P0(LAT, LAT) = std::pow(10.0f / R_M, 2);
        P0(LON, LON) = std::pow(10.0f / R_M, 2);
        P0(ALT, ALT) = std::pow(15.0f, 2);

        // Velocity uncertainty (1 m/s - trust GPS)
        P0(VN, VN) = std::pow(1.0f, 2);
        P0(VE, VE) = std::pow(1.0f, 2);
        P0(VD, VD) = std::pow(1.0f, 2);

        // Attitude uncertainty (5 degrees)
        float att_std = 5.0f * M_PI / 180.0f;
        P0(ROLL, ROLL) = std::pow(att_std, 2);
        P0(PITCH, PITCH) = std::pow(att_std, 2);
        P0(YAW, YAW) = std::pow(att_std, 2);

        // Gyro bias uncertainty (0.5 deg/hr)
        float bg_std = 0.5f * M_PI / 180.0f / 3600.0f;
        P0(BG_X, BG_X) = std::pow(bg_std, 2);
        P0(BG_Y, BG_Y) = std::pow(bg_std, 2);
        P0(BG_Z, BG_Z) = std::pow(bg_std, 2);

        // Accel bias uncertainty (0.1 mg)
        float ba_std = 0.1f * 9.81f / 1000.0f;
        P0(BA_X, BA_X) = std::pow(ba_std, 2);
        P0(BA_Y, BA_Y) = std::pow(ba_std, 2);
        P0(BA_Z, BA_Z) = std::pow(ba_std, 2);

        return P0;
    }

private:
    std::unique_ptr<GPSMeasurementModel> gps_model_;
    IMUMeasurement imu_;
    StateMat Q_base_;
    float dt_ = 0.01f;

    void initializeProcessNoise() {
        Q_base_ = StateMat::Zero();

        float R_M = 6371000.0f;

        // Position process noise - accounts for velocity uncertainty
        // ~1m per sqrt(s) in each direction
        Q_base_(LAT, LAT) = std::pow(1.0f / R_M, 2);
        Q_base_(LON, LON) = std::pow(1.0f / R_M, 2);
        Q_base_(ALT, ALT) = std::pow(2.0f, 2);

        // Velocity process noise - accounts for unmodeled accelerations
        // Aircraft turbulence/maneuvering can cause ~5 m/s² unmodeled accel
        Q_base_(VN, VN) = std::pow(2.0f, 2);
        Q_base_(VE, VE) = std::pow(2.0f, 2);
        Q_base_(VD, VD) = std::pow(5.0f, 2);  // Higher for vertical (turbulence)

        // Attitude process noise (gyro noise effect)
        float att_noise = 0.001f * M_PI / 180.0f;  // 0.001 deg/sqrt(s)
        Q_base_(ROLL, ROLL) = std::pow(att_noise, 2);
        Q_base_(PITCH, PITCH) = std::pow(att_noise, 2);
        Q_base_(YAW, YAW) = std::pow(att_noise, 2);

        // Gyro bias random walk
        float bg_rw = 0.01f * M_PI / 180.0f / 60.0f;  // 0.01 deg/sqrt(hr)
        Q_base_(BG_X, BG_X) = std::pow(bg_rw, 2);
        Q_base_(BG_Y, BG_Y) = std::pow(bg_rw, 2);
        Q_base_(BG_Z, BG_Z) = std::pow(bg_rw, 2);

        // Accel bias random walk
        float ba_rw = 0.01f / 60.0f;  // 0.01 m/s/sqrt(hr)
        Q_base_(BA_X, BA_X) = std::pow(ba_rw, 2);
        Q_base_(BA_Y, BA_Y) = std::pow(ba_rw, 2);
        Q_base_(BA_Z, BA_Z) = std::pow(ba_rw, 2);
    }

    static Eigen::Matrix3f getRotationBodyToNED(float roll, float pitch, float yaw) {
        float cr = std::cos(roll);
        float sr = std::sin(roll);
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cy = std::cos(yaw);
        float sy = std::sin(yaw);

        Eigen::Matrix3f C_bn;
        C_bn << cp*cy,  sr*sp*cy - cr*sy,  cr*sp*cy + sr*sy,
                cp*sy,  sr*sp*sy + cr*cy,  cr*sp*sy - sr*cy,
                -sp,    sr*cp,             cr*cp;
        return C_bn;
    }
};

} // namespace AircraftNav
