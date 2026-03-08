/**
 * @file INSErrorModel.h
 * @brief INS Error Model for Gyroscope and Accelerometer Bias Drift
 *
 * Models the error characteristics of a tactical-grade INS:
 * - Gyroscope bias drift (random walk)
 * - Accelerometer bias drift (random walk)
 * - Scale factor errors
 * - Misalignment errors
 *
 * Typical tactical-grade INS specifications:
 * - Gyro bias stability: 0.5-10 deg/hr
 * - Gyro random walk: 0.01-0.1 deg/sqrt(hr)
 * - Accel bias stability: 0.05-1 mg
 * - Accel random walk: 0.01-0.1 m/s/sqrt(hr)
 */

#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

namespace AircraftNav {

/**
 * @brief INS sensor error configuration
 */
struct INSErrorConfig {
    // Gyroscope parameters
    float gyro_bias_stability_rad_s = 0.5f * M_PI / 180.0f / 3600.0f;  // 0.5 deg/hr in rad/s
    float gyro_random_walk_rad_sqrt_s = 0.05f * M_PI / 180.0f / 60.0f; // 0.05 deg/sqrt(hr) in rad/sqrt(s)
    float gyro_noise_rad_s = 0.001f * M_PI / 180.0f;  // ARW in rad/s

    // Accelerometer parameters
    float accel_bias_stability_m_s2 = 0.1f * 9.81f / 1000.0f;  // 0.1 mg in m/s^2
    float accel_random_walk_m_s_sqrt_s = 0.05f / 60.0f;  // 0.05 m/s/sqrt(hr) in m/s/sqrt(s)
    float accel_noise_m_s2 = 0.001f * 9.81f;  // VRW in m/s^2

    // Initial bias values (can be set based on calibration)
    Eigen::Vector3f initial_gyro_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f initial_accel_bias = Eigen::Vector3f::Zero();

    // Time constants for first-order Gauss-Markov process
    float gyro_correlation_time_s = 3600.0f;   // 1 hour
    float accel_correlation_time_s = 3600.0f;  // 1 hour

    /**
     * @brief Create configuration for different INS grades
     */
    enum class Grade { NAVIGATION, TACTICAL, CONSUMER };

    static INSErrorConfig create(Grade grade) {
        INSErrorConfig cfg;

        switch (grade) {
            case Grade::NAVIGATION:
                // High-end navigation grade (ring laser gyro)
                cfg.gyro_bias_stability_rad_s = 0.003f * M_PI / 180.0f / 3600.0f;  // 0.003 deg/hr
                cfg.gyro_random_walk_rad_sqrt_s = 0.001f * M_PI / 180.0f / 60.0f;
                cfg.gyro_noise_rad_s = 1e-5f * M_PI / 180.0f;
                cfg.accel_bias_stability_m_s2 = 0.01f * 9.81f / 1000.0f;  // 0.01 mg
                cfg.accel_random_walk_m_s_sqrt_s = 0.01f / 60.0f;
                cfg.accel_noise_m_s2 = 1e-4f * 9.81f;
                cfg.gyro_correlation_time_s = 3600.0f * 10.0f;
                cfg.accel_correlation_time_s = 3600.0f * 10.0f;
                break;

            case Grade::TACTICAL:
                // Tactical grade (MEMS or FOG)
                cfg.gyro_bias_stability_rad_s = 1.0f * M_PI / 180.0f / 3600.0f;  // 1 deg/hr
                cfg.gyro_random_walk_rad_sqrt_s = 0.1f * M_PI / 180.0f / 60.0f;
                cfg.gyro_noise_rad_s = 0.01f * M_PI / 180.0f;
                cfg.accel_bias_stability_m_s2 = 0.5f * 9.81f / 1000.0f;  // 0.5 mg
                cfg.accel_random_walk_m_s_sqrt_s = 0.1f / 60.0f;
                cfg.accel_noise_m_s2 = 0.01f * 9.81f;
                cfg.gyro_correlation_time_s = 3600.0f;
                cfg.accel_correlation_time_s = 3600.0f;
                break;

            case Grade::CONSUMER:
                // Consumer MEMS
                cfg.gyro_bias_stability_rad_s = 10.0f * M_PI / 180.0f / 3600.0f;  // 10 deg/hr
                cfg.gyro_random_walk_rad_sqrt_s = 1.0f * M_PI / 180.0f / 60.0f;
                cfg.gyro_noise_rad_s = 0.1f * M_PI / 180.0f;
                cfg.accel_bias_stability_m_s2 = 5.0f * 9.81f / 1000.0f;  // 5 mg
                cfg.accel_random_walk_m_s_sqrt_s = 1.0f / 60.0f;
                cfg.accel_noise_m_s2 = 0.1f * 9.81f;
                cfg.gyro_correlation_time_s = 300.0f;
                cfg.accel_correlation_time_s = 300.0f;
                break;
        }

        return cfg;
    }
};

/**
 * @brief INS error model for gyro and accelerometer bias simulation
 *
 * Models bias drift as first-order Gauss-Markov process:
 *   db/dt = -b/τ + w
 * where τ is correlation time and w is white noise.
 *
 * The discrete-time approximation is:
 *   b[k+1] = exp(-dt/τ) * b[k] + σ * sqrt(1 - exp(-2*dt/τ)) * w[k]
 */
class INSErrorModel {
public:
    using Vec3 = Eigen::Vector3f;

    explicit INSErrorModel(const INSErrorConfig& config, uint64_t seed = 0)
        : config_(config)
        , rng_(seed ? seed : std::random_device{}())
        , white_noise_(0.0f, 1.0f)
        , gyro_bias_(config.initial_gyro_bias)
        , accel_bias_(config.initial_accel_bias)
    {}

    /**
     * @brief Update bias states (call once per timestep)
     * @param dt Time step [s]
     */
    void update(float dt) {
        // Gyroscope bias update (Gauss-Markov)
        float alpha_g = std::exp(-dt / config_.gyro_correlation_time_s);
        float sigma_g = config_.gyro_bias_stability_rad_s *
                        std::sqrt(1.0f - alpha_g * alpha_g);

        for (int i = 0; i < 3; ++i) {
            gyro_bias_(i) = alpha_g * gyro_bias_(i) + sigma_g * white_noise_(rng_);
        }

        // Accelerometer bias update (Gauss-Markov)
        float alpha_a = std::exp(-dt / config_.accel_correlation_time_s);
        float sigma_a = config_.accel_bias_stability_m_s2 *
                        std::sqrt(1.0f - alpha_a * alpha_a);

        for (int i = 0; i < 3; ++i) {
            accel_bias_(i) = alpha_a * accel_bias_(i) + sigma_a * white_noise_(rng_);
        }
    }

    /**
     * @brief Corrupt true gyroscope measurement with errors
     * @param omega_true True angular rate [rad/s]
     * @return Corrupted measurement [rad/s]
     */
    Vec3 corruptGyro(const Vec3& omega_true) const {
        Vec3 noise;
        for (int i = 0; i < 3; ++i) {
            noise(i) = config_.gyro_noise_rad_s * const_cast<INSErrorModel*>(this)->white_noise_(
                const_cast<INSErrorModel*>(this)->rng_);
        }
        return omega_true + gyro_bias_ + noise;
    }

    /**
     * @brief Corrupt true accelerometer measurement with errors
     * @param a_true True specific force [m/s^2]
     * @return Corrupted measurement [m/s^2]
     */
    Vec3 corruptAccel(const Vec3& a_true) const {
        Vec3 noise;
        for (int i = 0; i < 3; ++i) {
            noise(i) = config_.accel_noise_m_s2 * const_cast<INSErrorModel*>(this)->white_noise_(
                const_cast<INSErrorModel*>(this)->rng_);
        }
        return a_true + accel_bias_ + noise;
    }

    /**
     * @brief Get process noise covariance for bias states
     * @param dt Time step [s]
     * @return 6x6 process noise covariance [gyro_bias; accel_bias]
     */
    Eigen::Matrix<float, 6, 6> getBiasProcessNoise(float dt) const {
        Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Zero();

        // Gyro bias noise (random walk component)
        float q_gyro = config_.gyro_random_walk_rad_sqrt_s *
                       config_.gyro_random_walk_rad_sqrt_s * dt;
        Q.block<3, 3>(0, 0) = q_gyro * Eigen::Matrix3f::Identity();

        // Accel bias noise (random walk component)
        float q_accel = config_.accel_random_walk_m_s_sqrt_s *
                        config_.accel_random_walk_m_s_sqrt_s * dt;
        Q.block<3, 3>(3, 3) = q_accel * Eigen::Matrix3f::Identity();

        return Q;
    }

    /**
     * @brief Get expected position error growth rate (INS coasting)
     *
     * Position error growth during GPS denial:
     * - Position error ~ (1/2) * accel_bias * t^2 + initial_vel_error * t
     * - Heading error from gyro drift causes velocity cross-track error
     *
     * @param t Time since last update [s]
     * @return Approximate position error [m]
     */
    float getExpectedPositionError(float t) const {
        // Simple model: error grows quadratically due to accel bias
        float accel_error = config_.accel_bias_stability_m_s2;
        float gyro_error = config_.gyro_bias_stability_rad_s;

        // Position error from accel bias: (1/2)*a*t^2
        float pos_error_accel = 0.5f * accel_error * t * t;

        // Position error from heading drift (assuming V ~ 100 m/s):
        // heading_error = gyro_bias * t
        // cross_track = V * heading_error * t = V * gyro_bias * t^2
        float V = 100.0f;  // Approximate velocity
        float pos_error_gyro = V * gyro_error * t * t;

        return std::sqrt(pos_error_accel * pos_error_accel +
                         pos_error_gyro * pos_error_gyro);
    }

    // Accessors
    const Vec3& getGyroBias() const { return gyro_bias_; }
    const Vec3& getAccelBias() const { return accel_bias_; }

    void setGyroBias(const Vec3& bias) { gyro_bias_ = bias; }
    void setAccelBias(const Vec3& bias) { accel_bias_ = bias; }

    void reset() {
        gyro_bias_ = config_.initial_gyro_bias;
        accel_bias_ = config_.initial_accel_bias;
    }

    const INSErrorConfig& config() const { return config_; }

private:
    INSErrorConfig config_;
    std::mt19937_64 rng_;
    mutable std::normal_distribution<float> white_noise_;

    Vec3 gyro_bias_;
    Vec3 accel_bias_;
};

} // namespace AircraftNav
