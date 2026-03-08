/**
 * @file AircraftNavSRUKF.h
 * @brief Mode-Switching SRUKF for Aircraft Navigation
 *
 * Wraps the SRUKF for seamless transition between:
 * - GPS/INS integrated mode (6D measurement)
 * - Iridium AOA+Doppler mode (3D measurement per satellite)
 *
 * Handles:
 * - Automatic mode switching based on GPS availability
 * - Covariance management during transitions
 * - Multi-satellite fusion for Iridium mode
 * - Divergence detection and recovery
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <cmath>
#include "SRUKF.h"
#include "AircraftNavStateSpaceModel.h"
#include "AircraftAntennaModel.h"

namespace AircraftNav {

/**
 * @brief Navigation mode enumeration
 */
enum class NavMode {
    GPS_INS,        // Normal GPS/INS integration
    INS_COASTING,   // GPS denied, INS-only propagation
    IRIDIUM_NAV,    // Iridium AOA+Doppler measurements
    DEGRADED        // Filter diverging, limited confidence
};

/**
 * @brief Navigation filter configuration
 */
struct NavFilterConfig {
    // Mode switching thresholds
    float gps_timeout_s = 1.0f;           // GPS timeout threshold
    float iridium_timeout_s = 5.0f;       // Iridium timeout threshold
    float divergence_threshold = 1000.0f; // Position uncertainty limit [m]

    // Filter parameters
    float dt_nominal = 0.01f;             // Nominal timestep [s]
    float gps_update_rate = 10.0f;        // GPS update rate [Hz]
    float iridium_update_rate = 1.0f;     // Iridium update rate [Hz]

    // Iridium parameters
    float min_satellites = 1;             // Minimum satellites for update
    float snr_threshold_dB = 8.0f;        // Minimum SNR for measurement

    // Covariance inflation during GPS outage
    float covariance_growth_rate = 1.1f;  // Multiplicative growth per second

    static NavFilterConfig default_config() {
        return NavFilterConfig();
    }
};

/**
 * @brief Iridium satellite measurement
 */
struct IridiumMeasurement {
    int satellite_id;
    float azimuth_rad;
    float elevation_rad;
    float doppler_Hz;
    float snr_dB;
    bool valid;

    // Satellite ephemeris for measurement model
    IridiumMeasurementModel::SatellitePosition satellite_pos;
};

/**
 * @brief Navigation filter status
 */
struct NavFilterStatus {
    NavMode mode;
    float position_uncertainty_m;
    float velocity_uncertainty_mps;
    float heading_uncertainty_deg;

    float time_since_gps_s;
    float time_since_iridium_s;

    int num_visible_satellites;
    int num_tracked_satellites;

    float nees;  // Normalized estimation error squared
    bool is_converged;
    bool is_diverging;
};

/**
 * @brief Mode-switching SRUKF for aircraft navigation
 *
 * This wrapper manages two internal SRUKF instances:
 * - GPS mode: 15-state, 6-measurement
 * - Iridium mode: 15-state, 3-measurement (per satellite)
 *
 * The filter seamlessly switches between modes while maintaining
 * covariance consistency and preventing filter divergence.
 */
class AircraftNavSRUKF {
public:
    using State = Eigen::Matrix<float, 15, 1>;
    using StateMat = Eigen::Matrix<float, 15, 15>;
    using GPSObs = Eigen::Matrix<float, 6, 1>;
    using IridiumObs = Eigen::Matrix<float, 3, 1>;

    explicit AircraftNavSRUKF(const NavFilterConfig& config = NavFilterConfig::default_config())
        : config_(config)
        , model_gps_()
        , model_iridium_()
        , srukf_gps_(model_gps_)
        , srukf_iridium_(model_iridium_)
        , mode_(NavMode::GPS_INS)
        , time_since_gps_(0.0f)
        , time_since_iridium_(0.0f)
    {}

    /**
     * @brief Initialize filter with initial state and covariance
     */
    void initialize(const State& x0, const StateMat& P0) {
        srukf_gps_.initialize(x0, P0);
        srukf_iridium_.initialize(x0, P0);

        state_ = x0;
        S_ = computeCholesky(P0);

        // Reset all timing and mode state for fresh trial
        mode_ = NavMode::GPS_INS;
        time_since_gps_ = 0.0f;
        time_since_iridium_ = 0.0f;
        current_time_ = 0.0f;
        initialized_ = true;
    }

    /**
     * @brief Initialize with default covariance
     */
    void initialize(const State& x0) {
        initialize(x0, AircraftNavStateSpaceModel::getInitialCovariance());
    }

    /**
     * @brief Initialize for recovery phase after GPS outage
     *
     * Uses large initial covariance to account for ~3km drift during
     * 30s outage at 100 m/s airspeed.
     */
    void initializeForRecovery(const State& x0) {
        // Large initial covariance for recovery
        // Position uncertainty: ~5km (aircraft could be anywhere in this radius)
        // Velocity uncertainty: ~50 m/s (heading may have changed)
        StateMat P0 = StateMat::Zero();

        float R_M = 6371000.0f;
        float pos_std_m = 5000.0f;   // 5 km position uncertainty
        float vel_std = 50.0f;       // 50 m/s velocity uncertainty
        float att_std = 0.5f;        // 30 deg attitude uncertainty
        float gyro_std = 0.01f;      // Gyro bias uncertainty
        float accel_std = 0.5f;      // Accel bias uncertainty

        // Position (in radians)
        P0(LAT, LAT) = std::pow(pos_std_m / R_M, 2);
        P0(LON, LON) = std::pow(pos_std_m / R_M, 2);
        P0(ALT, ALT) = std::pow(pos_std_m, 2);

        // Velocity
        P0(VN, VN) = vel_std * vel_std;
        P0(VE, VE) = vel_std * vel_std;
        P0(VD, VD) = vel_std * vel_std;

        // Attitude
        P0(ROLL, ROLL) = att_std * att_std;
        P0(PITCH, PITCH) = att_std * att_std;
        P0(YAW, YAW) = att_std * att_std;

        // Biases
        for (int i = 9; i < 12; ++i) P0(i, i) = gyro_std * gyro_std;
        for (int i = 12; i < 15; ++i) P0(i, i) = accel_std * accel_std;

        initialize(x0, P0);

        // Set mode to Iridium navigation
        mode_ = NavMode::IRIDIUM_NAV;
    }

    /**
     * @brief Check if state is numerically valid
     * @return true if state contains no NaN/Inf and covariance is bounded
     */
    bool isStateValid() const {
        // Check for NaN/Inf in state
        for (int i = 0; i < 15; ++i) {
            if (!std::isfinite(state_(i))) return false;
        }

        // Check for NaN/Inf in covariance
        for (int i = 0; i < 15; ++i) {
            for (int j = 0; j < 15; ++j) {
                if (!std::isfinite(S_(i, j))) return false;
            }
        }

        // Check for excessive position uncertainty (> 1000 km in radians)
        float R_M = 6371000.0f;
        float max_pos_std = 1000000.0f / R_M;  // 1000 km in radians
        if (S_(LAT, LAT) > max_pos_std || S_(LON, LON) > max_pos_std) {
            return false;
        }

        return true;
    }

    /**
     * @brief Reinitialize filter if state is invalid
     * @param x0 State to reinitialize with
     */
    void recoverIfInvalid(const State& x0) {
        if (!isStateValid()) {
            initialize(x0);
        }
    }

    /**
     * @brief Propagate state using IMU measurements
     * @param gyro Gyroscope measurement [rad/s]
     * @param accel Accelerometer measurement [m/s^2]
     * @param dt Time step [s]
     */
    void predict(const Eigen::Vector3f& gyro, const Eigen::Vector3f& accel, float dt) {
        if (!initialized_) return;

        // Set IMU data in BOTH models to keep them in sync
        IMUMeasurement imu;
        imu.gyro = gyro;
        imu.accel = accel;
        model_gps_.setIMU(imu);
        model_gps_.setTimestep(dt);
        model_iridium_.setIMU(imu);
        model_iridium_.setTimestep(dt);

        // Predict BOTH filters to keep them in sync
        // This ensures Iridium filter is always ready for updates
        State u = State::Zero();  // No control input

        srukf_gps_.predict(current_time_, u);
        srukf_iridium_.predict(current_time_, u);

        // Use state from the active mode
        if (mode_ == NavMode::GPS_INS || mode_ == NavMode::INS_COASTING) {
            state_ = srukf_gps_.getState();
            S_ = srukf_gps_.getSqrtCovariance();
        } else {
            state_ = srukf_iridium_.getState();
            S_ = srukf_iridium_.getSqrtCovariance();
        }

        // Update timers
        time_since_gps_ += dt;
        time_since_iridium_ += dt;
        current_time_ += dt;

        // Check for mode transitions
        updateMode();

        // Inflate covariance during GPS outage
        if (mode_ == NavMode::INS_COASTING || mode_ == NavMode::IRIDIUM_NAV) {
            inflateCovariance(dt);
        }
    }

    /**
     * @brief Update with GPS measurement
     *
     * GPS updates ONLY affect the GPS filter. The Iridium filter maintains
     * its own independent state using IMU + Iridium measurements.
     * When GPS goes out, we switch to using the Iridium filter's state.
     */
    void updateGPS(const GPSObs& gps_meas) {
        if (!initialized_) return;

        // Switch to GPS mode if not already
        if (mode_ != NavMode::GPS_INS) {
            switchToGPSMode();
        }

        // Perform update (GPS filter only)
        srukf_gps_.update(current_time_, gps_meas);

        // Use GPS state as the nav output
        state_ = srukf_gps_.getState();
        S_ = srukf_gps_.getSqrtCovariance();

        // Reset GPS timer
        time_since_gps_ = 0.0f;

        // NOTE: Iridium filter is NOT synced - it maintains independent state
        // using IMU + Iridium updates only
    }

    /**
     * @brief Update with Iridium measurements from multiple satellites
     *
     * Iridium filter is INDEPENDENT of GPS. It maintains its own state
     * using IMU propagation + Iridium measurements. The nav output only
     * switches to Iridium state when GPS is unavailable.
     */
    void updateIridium(const std::vector<IridiumMeasurement>& measurements) {
        if (!initialized_) return;

        // Filter valid measurements
        std::vector<IridiumMeasurement> valid_meas;
        for (const auto& m : measurements) {
            if (m.valid && m.snr_dB >= config_.snr_threshold_dB) {
                valid_meas.push_back(m);
            }
        }

        if (valid_meas.size() < static_cast<size_t>(config_.min_satellites)) {
            return;
        }

        // Sequential update for each satellite (always update Iridium filter)
        for (const auto& meas : valid_meas) {
            // Set satellite position in measurement model
            model_iridium_.setSatellitePosition(meas.satellite_pos);
            model_iridium_.updateNoiseFromSNR(meas.snr_dB);

            // Create measurement vector
            IridiumObs y;
            y(0) = meas.azimuth_rad;
            y(1) = meas.elevation_rad;
            y(2) = meas.doppler_Hz;

            // Update Iridium filter
            srukf_iridium_.update(current_time_, y);
        }

        // Reset Iridium timer
        time_since_iridium_ = 0.0f;

        // Switch to Iridium mode if GPS unavailable
        if (mode_ == NavMode::INS_COASTING) {
            switchToIridiumMode();
        }

        // Only use Iridium state for nav output when NOT in GPS mode
        if (mode_ != NavMode::GPS_INS) {
            state_ = srukf_iridium_.getState();
            S_ = srukf_iridium_.getSqrtCovariance();
        }
        // When in GPS mode, GPS state remains the nav output
        // Iridium filter continues updating independently
    }

    /**
     * @brief Notify GPS outage (explicit call)
     *
     * When GPS jamming is detected, copy the last good GPS state
     * to the Iridium filter so it starts from an accurate position.
     */
    void notifyGPSOutage() {
        if (mode_ == NavMode::GPS_INS) {
            // Copy GPS state to Iridium filter before switching modes
            // This ensures Iridium starts from the last known good state
            srukf_iridium_.setState(state_);
            srukf_iridium_.setSqrtCovariance(S_);

            mode_ = NavMode::INS_COASTING;
        }
    }

    /**
     * @brief Get current navigation state
     */
    const State& getState() const { return state_; }

    /**
     * @brief Get current covariance matrix
     */
    StateMat getCovariance() const { return S_ * S_.transpose(); }

    /**
     * @brief Get square root covariance
     */
    const StateMat& getSqrtCovariance() const { return S_; }

    /**
     * @brief Get filter status
     */
    NavFilterStatus getStatus() const {
        NavFilterStatus status;
        status.mode = mode_;

        // Compute position uncertainty (3D)
        float R_M = 6371000.0f;
        StateMat P = getCovariance();
        float sigma_lat_m = std::sqrt(P(LAT, LAT)) * R_M;
        float sigma_lon_m = std::sqrt(P(LON, LON)) * R_M * std::cos(state_(LAT));
        float sigma_alt_m = std::sqrt(P(ALT, ALT));
        status.position_uncertainty_m = std::sqrt(
            sigma_lat_m * sigma_lat_m + sigma_lon_m * sigma_lon_m + sigma_alt_m * sigma_alt_m);

        // Velocity uncertainty
        float sigma_vn = std::sqrt(P(VN, VN));
        float sigma_ve = std::sqrt(P(VE, VE));
        float sigma_vd = std::sqrt(P(VD, VD));
        status.velocity_uncertainty_mps = std::sqrt(
            sigma_vn * sigma_vn + sigma_ve * sigma_ve + sigma_vd * sigma_vd);

        // Heading uncertainty
        status.heading_uncertainty_deg = std::sqrt(P(YAW, YAW)) * 180.0f / M_PI;

        status.time_since_gps_s = time_since_gps_;
        status.time_since_iridium_s = time_since_iridium_;

        status.num_visible_satellites = 0;  // Updated externally
        status.num_tracked_satellites = 0;

        // Convergence status
        status.is_converged = (status.position_uncertainty_m < 100.0f);
        status.is_diverging = (status.position_uncertainty_m > config_.divergence_threshold);

        return status;
    }

    /**
     * @brief Get current mode
     */
    NavMode getMode() const { return mode_; }

    /**
     * @brief Force mode change (for testing)
     */
    void setMode(NavMode mode) { mode_ = mode; }

    /**
     * @brief Reset filter
     */
    void reset() {
        mode_ = NavMode::GPS_INS;
        time_since_gps_ = 0.0f;
        time_since_iridium_ = 0.0f;
        current_time_ = 0.0f;
        initialized_ = false;
    }

    /**
     * @brief Full reset for Monte Carlo trials
     *
     * Resets all internal state to ensure complete isolation between trials.
     * This prevents any numerical artifacts from affecting subsequent runs.
     */
    void resetForNewTrial() {
        // Reset timing and mode
        mode_ = NavMode::GPS_INS;
        time_since_gps_ = 0.0f;
        time_since_iridium_ = 0.0f;
        current_time_ = 0.0f;
        initialized_ = false;

        // Reset state and covariance to defaults
        state_.setZero();
        S_.setIdentity();

        // Reset both SRUKF instances with default state
        State default_state = State::Zero();
        StateMat default_P = AircraftNavStateSpaceModel::getInitialCovariance();
        srukf_gps_.initialize(default_state, default_P);
        srukf_iridium_.initialize(default_state, default_P);
    }

private:
    NavFilterConfig config_;

    // State space models
    AircraftNavStateSpaceModel model_gps_;
    IridiumMeasurementModel model_iridium_;

    // SRUKF instances
    UKFCore::SRUKF<15, 6> srukf_gps_;
    UKFCore::SRUKF<15, 3> srukf_iridium_;

    // Current state (shared between modes)
    State state_;
    StateMat S_;

    // Mode tracking
    NavMode mode_;
    float time_since_gps_;
    float time_since_iridium_;
    float current_time_ = 0.0f;
    bool initialized_ = false;

    void updateMode() {
        switch (mode_) {
            case NavMode::GPS_INS:
                if (time_since_gps_ > config_.gps_timeout_s) {
                    mode_ = NavMode::INS_COASTING;
                }
                break;

            case NavMode::INS_COASTING:
                if (time_since_gps_ < config_.gps_timeout_s) {
                    mode_ = NavMode::GPS_INS;
                }
                // Transition to Iridium mode handled in updateIridium()
                break;

            case NavMode::IRIDIUM_NAV:
                if (time_since_gps_ < config_.gps_timeout_s) {
                    mode_ = NavMode::GPS_INS;
                } else if (time_since_iridium_ > config_.iridium_timeout_s) {
                    mode_ = NavMode::INS_COASTING;
                }
                break;

            case NavMode::DEGRADED:
                // Recover if uncertainty decreases
                if (getStatus().position_uncertainty_m < config_.divergence_threshold * 0.5f) {
                    if (time_since_gps_ < config_.gps_timeout_s) {
                        mode_ = NavMode::GPS_INS;
                    } else if (time_since_iridium_ < config_.iridium_timeout_s) {
                        mode_ = NavMode::IRIDIUM_NAV;
                    } else {
                        mode_ = NavMode::INS_COASTING;
                    }
                }
                break;
        }

        // Check for divergence
        if (getStatus().position_uncertainty_m > config_.divergence_threshold) {
            mode_ = NavMode::DEGRADED;
        }
    }

    void switchToGPSMode() {
        // Transfer state from Iridium to GPS filter
        srukf_gps_.setState(state_);
        srukf_gps_.setSqrtCovariance(S_);
        mode_ = NavMode::GPS_INS;
    }

    void switchToIridiumMode() {
        // Transfer state from GPS to Iridium filter
        srukf_iridium_.setState(state_);
        srukf_iridium_.setSqrtCovariance(S_);
        mode_ = NavMode::IRIDIUM_NAV;
    }

    void syncFilters() {
        // Synchronize both filters with current best estimate
        srukf_gps_.setState(state_);
        srukf_gps_.setSqrtCovariance(S_);
        srukf_iridium_.setState(state_);
        srukf_iridium_.setSqrtCovariance(S_);
    }

    void inflateCovariance(float dt) {
        // Inflate covariance during GPS outage to account for
        // unmodeled error growth

        // Growth rate per second
        float growth = std::pow(config_.covariance_growth_rate, dt);

        // Apply to position and velocity states
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 15; ++j) {
                S_(i, j) *= growth;
            }
        }

        // Ensure positive definiteness
        for (int i = 0; i < 15; ++i) {
            if (S_(i, i) < 1e-10f) {
                S_(i, i) = 1e-10f;
            }
        }

        // Update both filters
        srukf_gps_.setSqrtCovariance(S_);
        srukf_iridium_.setSqrtCovariance(S_);
    }

    static StateMat computeCholesky(const StateMat& P) {
        Eigen::LLT<StateMat> llt(P);
        if (llt.info() == Eigen::Success) {
            return llt.matrixL();
        }

        // Fallback: use diagonal
        StateMat S = StateMat::Zero();
        for (int i = 0; i < 15; ++i) {
            S(i, i) = std::sqrt(std::max(P(i, i), 1e-10f));
        }
        return S;
    }
};

} // namespace AircraftNav
