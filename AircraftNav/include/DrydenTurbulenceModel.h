/**
 * @file DrydenTurbulenceModel.h
 * @brief MIL-F-8785C Dryden Wind Turbulence Model
 *
 * Implements continuous Dryden turbulence model with spectral filters
 * for realistic wind gust simulation. Parameters based on MIL-F-8785C
 * for moderate turbulence at medium altitude (10,000 ft).
 *
 * Transfer functions:
 *   H_u(s) = σ_u * sqrt(2*L_u / (π*V)) / (1 + L_u*s/V)
 *   H_v(s) = σ_v * sqrt(L_v / (π*V)) * (1 + sqrt(3)*L_v*s/V) / (1 + L_v*s/V)^2
 *   H_w(s) = σ_w * sqrt(L_w / (π*V)) * (1 + sqrt(3)*L_w*s/V) / (1 + L_w*s/V)^2
 */

#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

namespace AircraftNav {

/**
 * @brief Dryden turbulence model configuration
 */
struct DrydenConfig {
    // Turbulence scale lengths (ft converted to m)
    float L_u = 1750.0f * 0.3048f;  // Longitudinal [m]
    float L_v = 1750.0f * 0.3048f;  // Lateral [m]
    float L_w = 1750.0f * 0.3048f;  // Vertical [m]

    // Turbulence intensities [m/s] (ft/s converted)
    float sigma_u = 6.0f * 0.3048f;  // Longitudinal (moderate: 5-10 ft/s)
    float sigma_v = 6.0f * 0.3048f;  // Lateral
    float sigma_w = 4.0f * 0.3048f;  // Vertical (usually smaller)

    // Aircraft true airspeed [m/s]
    float V = 200.0f * 0.5144f;  // 200 knots in m/s

    // Altitude for parameter scaling
    float altitude_m = 3048.0f;  // 10,000 ft in m

    // Turbulence severity level
    enum class Severity { LIGHT, MODERATE, SEVERE };
    Severity severity = Severity::MODERATE;

    /**
     * @brief Create configuration for given altitude and severity
     */
    static DrydenConfig create(float altitude_m, float airspeed_mps, Severity severity) {
        DrydenConfig cfg;
        cfg.altitude_m = altitude_m;
        cfg.V = airspeed_mps;

        // Scale lengths depend on altitude (MIL-F-8785C)
        float h_ft = altitude_m / 0.3048f;

        if (h_ft < 1000.0f) {
            // Low altitude model
            cfg.L_u = h_ft / std::pow(0.177f + 0.000823f * h_ft, 1.2f);
            cfg.L_v = cfg.L_u;
            cfg.L_w = h_ft;
        } else {
            // Medium/high altitude model
            cfg.L_u = 1750.0f;
            cfg.L_v = 1750.0f;
            cfg.L_w = 1750.0f;
        }

        // Convert to meters
        cfg.L_u *= 0.3048f;
        cfg.L_v *= 0.3048f;
        cfg.L_w *= 0.3048f;

        // Intensities based on severity
        float W20;  // Wind at 20ft [ft/s]
        switch (severity) {
            case Severity::LIGHT:
                W20 = 15.0f;
                break;
            case Severity::MODERATE:
                W20 = 30.0f;
                break;
            case Severity::SEVERE:
                W20 = 45.0f;
                break;
        }

        // Turbulence intensity scaling
        if (h_ft < 1000.0f) {
            cfg.sigma_u = W20 / std::pow(0.177f + 0.000823f * h_ft, 0.4f);
            cfg.sigma_v = cfg.sigma_u;
            cfg.sigma_w = 0.1f * W20;
        } else {
            float sigma_ref = 0.1f * W20;  // Reference at altitude
            cfg.sigma_u = sigma_ref;
            cfg.sigma_v = sigma_ref;
            cfg.sigma_w = sigma_ref;
        }

        // Convert to m/s
        cfg.sigma_u *= 0.3048f;
        cfg.sigma_v *= 0.3048f;
        cfg.sigma_w *= 0.3048f;

        cfg.severity = severity;
        return cfg;
    }
};

/**
 * @brief Dryden turbulence generator with state-space filter implementation
 *
 * Uses discrete-time approximation of continuous Dryden transfer functions.
 * Filter states are maintained internally for correlated noise generation.
 */
class DrydenTurbulenceModel {
public:
    using Vec3 = Eigen::Vector3f;

    explicit DrydenTurbulenceModel(const DrydenConfig& config, uint64_t seed = 0)
        : config_(config)
        , rng_(seed ? seed : std::random_device{}())
        , white_noise_(0.0f, 1.0f)
    {
        // Initialize filter states to zero
        x_u_ = 0.0f;
        x_v_.setZero();
        x_w_.setZero();

        // Precompute filter coefficients
        updateCoefficients();
    }

    /**
     * @brief Generate turbulence velocity components at current timestep
     * @param dt Time step [s]
     * @return Wind velocity [u_g, v_g, w_g] in body axes [m/s]
     */
    Vec3 update(float dt) {
        // Generate white noise inputs
        float nu = white_noise_(rng_);
        float nv = white_noise_(rng_);
        float nw = white_noise_(rng_);

        // Update filter states using discrete-time approximation
        // First-order filter for u: H_u(s) = K_u / (1 + T_u*s)
        float a_u = std::exp(-dt * config_.V / config_.L_u);
        x_u_ = a_u * x_u_ + (1.0f - a_u) * K_u_ * nu;
        float u_g = x_u_;

        // Second-order filter for v (von Karman approximation)
        // State-space: [x1, x2] with two poles at -V/L_v
        float a_v = std::exp(-dt * config_.V / config_.L_v);
        float a_v2 = a_v * a_v;

        // Update state vector
        Eigen::Vector2f x_v_new;
        x_v_new(0) = a_v * x_v_(0) + dt * x_v_(1) + K_v1_ * (1.0f - a_v) * nv;
        x_v_new(1) = a_v2 * x_v_(1) + K_v2_ * (1.0f - a_v2) * nv;
        x_v_ = x_v_new;
        float v_g = x_v_(0);

        // Second-order filter for w (similar to v)
        Eigen::Vector2f x_w_new;
        float a_w = std::exp(-dt * config_.V / config_.L_w);
        float a_w2 = a_w * a_w;
        x_w_new(0) = a_w * x_w_(0) + dt * x_w_(1) + K_w1_ * (1.0f - a_w) * nw;
        x_w_new(1) = a_w2 * x_w_(1) + K_w2_ * (1.0f - a_w2) * nw;
        x_w_ = x_w_new;
        float w_g = x_w_(0);

        return Vec3(u_g, v_g, w_g);
    }

    /**
     * @brief Get current turbulence velocity without advancing state
     */
    Vec3 getCurrentTurbulence() const {
        return Vec3(x_u_, x_v_(0), x_w_(0));
    }

    /**
     * @brief Reset filter states to zero
     */
    void reset() {
        x_u_ = 0.0f;
        x_v_.setZero();
        x_w_.setZero();
    }

    /**
     * @brief Update configuration (e.g., for changing airspeed)
     */
    void setConfig(const DrydenConfig& config) {
        config_ = config;
        updateCoefficients();
    }

    const DrydenConfig& config() const { return config_; }

private:
    DrydenConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> white_noise_;

    // Filter states
    float x_u_;              // Longitudinal (first-order)
    Eigen::Vector2f x_v_;    // Lateral (second-order)
    Eigen::Vector2f x_w_;    // Vertical (second-order)

    // Filter gains
    float K_u_, K_v1_, K_v2_, K_w1_, K_w2_;

    void updateCoefficients() {
        // First-order gain: sqrt(2*L/pi/V) * sigma
        K_u_ = config_.sigma_u * std::sqrt(2.0f * config_.L_u / (M_PI * config_.V));

        // Second-order gains (simplified from von Karman spectrum)
        K_v1_ = config_.sigma_v * std::sqrt(config_.L_v / (M_PI * config_.V));
        K_v2_ = K_v1_ * std::sqrt(3.0f) * config_.V / config_.L_v;

        K_w1_ = config_.sigma_w * std::sqrt(config_.L_w / (M_PI * config_.V));
        K_w2_ = K_w1_ * std::sqrt(3.0f) * config_.V / config_.L_w;
    }
};

} // namespace AircraftNav
