/**
 * @file AircraftAntennaModel.h
 * @brief Aircraft Antenna Model for Iridium Reception
 *
 * Models the dual-patch antenna system for AOA+Doppler measurements:
 * - Body-to-ENU coordinate transformation
 * - Link budget computation
 * - SNR estimation based on elevation angle
 * - Phase ambiguity handling for baseline > λ/2
 *
 * Antenna Configuration:
 * - Dual patch: 10cm baseline, 3.1 dBi gain
 * - LNA: 20 dB gain, 1.2 dB noise figure
 * - Expected SNR at 30° elevation: 15-18 dB
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace AircraftNav {

/**
 * @brief Antenna system configuration
 */
struct AntennaConfig {
    // Physical parameters
    float baseline_m = 0.10f;          // Antenna separation [m]
    float antenna_gain_dBi = 3.1f;     // Patch antenna gain [dBi]
    float lna_gain_dB = 20.0f;         // LNA gain [dB]
    float lna_nf_dB = 1.2f;            // LNA noise figure [dB]

    // System parameters
    float system_temp_K = 290.0f;      // System noise temperature [K]
    float bandwidth_Hz = 25000.0f;     // Receiver bandwidth [Hz]

    // Iridium parameters
    float satellite_eirp_dBW = 28.0f;  // Satellite EIRP [dBW]
    float frequency_Hz = 1626e6f;      // L-band frequency [Hz]
    float wavelength_m = 0.1844f;      // Wavelength [m]

    // Antenna position relative to aircraft CG (body frame) [m]
    Eigen::Vector3f position = Eigen::Vector3f(0.0f, 0.0f, -1.0f);  // Top of fuselage

    // Antenna boresight in body frame (unit vector)
    Eigen::Vector3f boresight = Eigen::Vector3f(0.0f, 0.0f, -1.0f);  // Pointing up

    // Half-power beamwidth [rad]
    float hpbw_rad = 70.0f * M_PI / 180.0f;  // 70° for patch antenna

    static AntennaConfig default_aircraft() {
        return AntennaConfig();
    }
};

/**
 * @brief Result of satellite visibility check
 */
struct SatelliteVisibility {
    bool visible;
    float azimuth_rad;       // Azimuth in body frame
    float elevation_rad;     // Elevation in body frame
    float snr_dB;            // Estimated SNR
    float range_m;           // Slant range to satellite
    float doppler_Hz;        // Doppler shift

    // Antenna pattern attenuation
    float pattern_loss_dB;
    bool in_main_beam;
};

/**
 * @brief Aircraft antenna model for Iridium reception
 *
 * Computes satellite visibility and signal quality accounting for:
 * - Aircraft attitude (body-to-NED transformation)
 * - Antenna pattern and pointing
 * - Link budget (path loss, atmospheric effects)
 * - Phase ambiguity for AOA estimation
 */
class AircraftAntennaModel {
public:
    using Vec3 = Eigen::Vector3f;
    using Mat3 = Eigen::Matrix3f;

    explicit AircraftAntennaModel(const AntennaConfig& config, uint64_t seed = 0)
        : config_(config)
        , rng_(seed ? seed : std::random_device{}())
        , noise_dist_(0.0f, 1.0f)
    {}

    /**
     * @brief Check satellite visibility from aircraft
     *
     * @param aircraft_lat Aircraft latitude [rad]
     * @param aircraft_lon Aircraft longitude [rad]
     * @param aircraft_alt Aircraft altitude [m]
     * @param roll Aircraft roll [rad]
     * @param pitch Aircraft pitch [rad]
     * @param yaw Aircraft yaw [rad]
     * @param sat_lat Satellite latitude [rad]
     * @param sat_lon Satellite longitude [rad]
     * @param sat_alt Satellite altitude [m]
     * @return SatelliteVisibility structure
     */
    SatelliteVisibility checkVisibility(
        float aircraft_lat, float aircraft_lon, float aircraft_alt,
        float roll, float pitch, float yaw,
        float sat_lat, float sat_lon, float sat_alt) const
    {
        SatelliteVisibility result;
        result.visible = false;

        // Convert positions to ECEF
        Vec3 ac_ecef = geodeticToECEF(aircraft_lat, aircraft_lon, aircraft_alt);
        Vec3 sat_ecef = geodeticToECEF(sat_lat, sat_lon, sat_alt);

        // Line of sight vector
        Vec3 los_ecef = sat_ecef - ac_ecef;
        result.range_m = los_ecef.norm();
        Vec3 los_hat_ecef = los_ecef / result.range_m;

        // Transform LOS to NED frame
        Mat3 R_ecef_ned = getECEFtoNEDRotation(aircraft_lat, aircraft_lon);
        Vec3 los_ned = R_ecef_ned * los_hat_ecef;

        // Transform to body frame
        Mat3 R_ned_body = getNEDtoBodyRotation(roll, pitch, yaw);
        Vec3 los_body = R_ned_body * los_ned;

        // Compute azimuth/elevation in body frame
        // Azimuth from nose (x-axis), clockwise positive when viewed from above
        result.azimuth_rad = std::atan2(los_body(1), los_body(0));

        // Elevation from horizontal plane
        result.elevation_rad = std::asin(std::clamp(-los_body(2), -1.0f, 1.0f));

        // Check if above horizon (minimum 5° elevation in NED)
        float elevation_ned = std::asin(std::clamp(-los_ned(2), -1.0f, 1.0f));
        if (elevation_ned < 5.0f * M_PI / 180.0f) {
            return result;
        }

        // Compute antenna pattern loss
        // Angle from antenna boresight
        float cos_theta = -los_body.dot(config_.boresight);
        float theta = std::acos(std::clamp(cos_theta, -1.0f, 1.0f));

        // Simple cosine pattern model
        result.in_main_beam = (theta < config_.hpbw_rad);
        if (theta < M_PI / 2.0f) {
            // Pattern: G(θ) = G_max * cos^2(θ)
            float gain_linear = std::pow(std::cos(theta), 2.0f);
            result.pattern_loss_dB = -10.0f * std::log10(std::max(gain_linear, 0.01f));
        } else {
            result.pattern_loss_dB = 20.0f;  // Back lobe, significant loss
        }

        // Link budget
        result.snr_dB = computeLinkBudget(result.range_m, elevation_ned,
                                          result.pattern_loss_dB);

        // Check SNR threshold (minimum 5 dB for reliable detection)
        if (result.snr_dB < 5.0f) {
            return result;
        }

        result.visible = true;
        return result;
    }

    /**
     * @brief Compute AOA measurement with realistic errors
     *
     * @param visibility Satellite visibility information
     * @return Measured azimuth and elevation [rad] with noise
     */
    std::pair<float, float> measureAOA(const SatelliteVisibility& vis) {
        if (!vis.visible) {
            return {0.0f, 0.0f};
        }

        // SNR-dependent noise
        float snr_linear = std::pow(10.0f, vis.snr_dB / 10.0f);
        float phase_noise_std = 1.0f / std::sqrt(2.0f * snr_linear);

        // Convert phase noise to angle noise
        // For interferometer: Δφ = 2π * d * sin(θ) / λ
        // δθ ≈ λ * δφ / (2π * d * cos(θ))
        float baseline_wavelengths = config_.baseline_m / config_.wavelength_m;

        // Azimuth noise (depends on elevation - worse at low elevation)
        float az_scale = 1.0f / std::max(std::cos(vis.elevation_rad), 0.1f);
        float sigma_az = phase_noise_std / (2.0f * M_PI * baseline_wavelengths) * az_scale;
        sigma_az = std::clamp(sigma_az, 0.02f, 0.1f);  // 1-6 degrees

        // Elevation noise (typically worse than azimuth for patch antenna)
        float sigma_el = sigma_az * 1.5f;
        sigma_el = std::clamp(sigma_el, 0.03f, 0.15f);  // 2-8 degrees

        float az_meas = vis.azimuth_rad + sigma_az * noise_dist_(rng_);
        float el_meas = vis.elevation_rad + sigma_el * noise_dist_(rng_);

        // Wrap azimuth
        while (az_meas > M_PI) az_meas -= 2.0f * M_PI;
        while (az_meas < -M_PI) az_meas += 2.0f * M_PI;

        // Clamp elevation
        el_meas = std::clamp(el_meas, 0.0f, static_cast<float>(M_PI / 2.0f));

        return {az_meas, el_meas};
    }

    /**
     * @brief Get measurement noise covariance for AOA
     *
     * @param snr_dB Signal-to-noise ratio [dB]
     * @return 2x2 covariance matrix [azimuth, elevation] [rad^2]
     */
    Eigen::Matrix2f getAOANoiseCovariance(float snr_dB) const {
        float snr_linear = std::pow(10.0f, snr_dB / 10.0f);
        float phase_noise_std = 1.0f / std::sqrt(2.0f * snr_linear);
        float baseline_wavelengths = config_.baseline_m / config_.wavelength_m;

        float sigma_az = phase_noise_std / (2.0f * M_PI * baseline_wavelengths);
        sigma_az = std::clamp(sigma_az, 0.02f, 0.1f);

        float sigma_el = sigma_az * 1.5f;
        sigma_el = std::clamp(sigma_el, 0.03f, 0.15f);

        Eigen::Matrix2f R;
        R << sigma_az * sigma_az, 0.0f,
             0.0f, sigma_el * sigma_el;
        return R;
    }

    /**
     * @brief Check for phase ambiguity
     *
     * For baseline > λ/2, phase measurements are ambiguous.
     * Returns number of ambiguous lobes.
     */
    int getPhaseAmbiguityCount() const {
        return static_cast<int>(std::floor(config_.baseline_m / config_.wavelength_m + 0.5f));
    }

    /**
     * @brief Resolve phase ambiguity using multiple frequencies or prior info
     *
     * @param measured_az Ambiguous azimuth measurement [rad]
     * @param prior_az Prior azimuth estimate [rad]
     * @param prior_std Prior uncertainty [rad]
     * @return Resolved azimuth [rad]
     */
    float resolveAmbiguity(float measured_az, float prior_az, float prior_std) const {
        int n_amb = getPhaseAmbiguityCount();
        if (n_amb <= 1) {
            return measured_az;  // No ambiguity
        }

        // Find closest ambiguous solution to prior
        float best_az = measured_az;
        float best_dist = std::abs(measured_az - prior_az);

        float amb_spacing = 2.0f * M_PI / n_amb;  // Approximate spacing
        for (int k = -n_amb; k <= n_amb; ++k) {
            float candidate = measured_az + k * amb_spacing;
            float dist = std::abs(candidate - prior_az);
            if (dist < best_dist) {
                best_dist = dist;
                best_az = candidate;
            }
        }

        return best_az;
    }

    const AntennaConfig& config() const { return config_; }

private:
    AntennaConfig config_;
    mutable std::mt19937_64 rng_;
    mutable std::normal_distribution<float> noise_dist_;

    float computeLinkBudget(float range_m, float elevation_rad, float pattern_loss_dB) const {
        // Satellite EIRP [dBW]
        float eirp_dBW = config_.satellite_eirp_dBW;

        // Free space path loss [dB]
        // FSPL = 20*log10(4*π*d/λ)
        float fspl_dB = 20.0f * std::log10(4.0f * M_PI * range_m / config_.wavelength_m);

        // Atmospheric losses (simplified model)
        // Higher at low elevation due to longer path through atmosphere
        float atm_loss_dB = 0.5f / std::max(std::sin(elevation_rad), 0.1f);

        // Antenna gain [dBi]
        float antenna_gain_dB = config_.antenna_gain_dBi - pattern_loss_dB;

        // LNA gain [dB]
        float lna_gain_dB = config_.lna_gain_dB;

        // System noise [dBW]
        // N = k * T * B
        const float k_B = 1.38e-23f;  // Boltzmann constant
        float noise_power_W = k_B * config_.system_temp_K * config_.bandwidth_Hz;
        float noise_power_dBW = 10.0f * std::log10(noise_power_W);

        // Noise figure contribution
        float nf_contribution_dB = config_.lna_nf_dB;

        // Total received power [dBW]
        float rx_power_dBW = eirp_dBW - fspl_dB - atm_loss_dB + antenna_gain_dB + lna_gain_dB;

        // SNR [dB]
        float snr_dB = rx_power_dBW - noise_power_dBW - nf_contribution_dB;

        return snr_dB;
    }

    static Vec3 geodeticToECEF(float lat, float lon, float alt) {
        constexpr float A = 6378137.0f;
        constexpr float E2 = 0.00669437999f;

        float sin_lat = std::sin(lat);
        float cos_lat = std::cos(lat);
        float sin_lon = std::sin(lon);
        float cos_lon = std::cos(lon);

        float N = A / std::sqrt(1.0f - E2 * sin_lat * sin_lat);

        Vec3 ecef;
        ecef(0) = (N + alt) * cos_lat * cos_lon;
        ecef(1) = (N + alt) * cos_lat * sin_lon;
        ecef(2) = (N * (1.0f - E2) + alt) * sin_lat;
        return ecef;
    }

    static Mat3 getECEFtoNEDRotation(float lat, float lon) {
        float sin_lat = std::sin(lat);
        float cos_lat = std::cos(lat);
        float sin_lon = std::sin(lon);
        float cos_lon = std::cos(lon);

        Mat3 R;
        R << -sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat,
             -sin_lon,            cos_lon,             0.0f,
             -cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat;
        return R;
    }

    static Mat3 getNEDtoBodyRotation(float roll, float pitch, float yaw) {
        float cr = std::cos(roll);
        float sr = std::sin(roll);
        float cp = std::cos(pitch);
        float sp = std::sin(pitch);
        float cy = std::cos(yaw);
        float sy = std::sin(yaw);

        Mat3 R;
        R << cp*cy,                cp*sy,                -sp,
             sr*sp*cy - cr*sy,     sr*sp*sy + cr*cy,     sr*cp,
             cr*sp*cy + sr*sy,     cr*sp*sy - sr*cy,     cr*cp;
        return R;
    }
};

} // namespace AircraftNav
