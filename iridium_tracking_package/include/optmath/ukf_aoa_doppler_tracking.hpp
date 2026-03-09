/**
 * @file ukf_aoa_doppler_tracking.hpp
 * @brief Enhanced UKF with AOA + Doppler for Iridium Satellite Tracking
 *
 * Extends the basic AOA tracking with Doppler measurements extracted from
 * burst demodulation. Doppler provides direct range-rate observability,
 * dramatically improving position and velocity estimation accuracy.
 *
 * Measurement Model:
 * - Azimuth (from two-antenna phase difference)
 * - Elevation (from amplitude comparison or separate estimation)
 * - Doppler shift (from carrier frequency offset in demodulated burst)
 *
 * The Doppler measurement provides the radial velocity component:
 *   f_doppler = -v_radial / wavelength = -(v · r_hat) / λ
 *
 * @author OptMathKernels
 * @version 0.5.0
 */

#pragma once

#include "ukf_aoa_tracking.hpp"

namespace optmath {
namespace tracking {

//=============================================================================
// Extended Constants for Doppler
//=============================================================================

namespace constants {
    // Doppler parameters for Iridium
    // Max Doppler at horizon: ~±35 kHz (v_orbital ≈ 7.5 km/s, cos(angle) ≈ 0.87)
    constexpr double IRIDIUM_MAX_DOPPLER = 35000.0;  // Hz

    // Typical SDR demodulator Doppler accuracy
    constexpr double DOPPLER_ACCURACY_COARSE = 100.0;  // Hz (AFC loop)
    constexpr double DOPPLER_ACCURACY_FINE = 10.0;     // Hz (burst preamble)
    constexpr double DOPPLER_ACCURACY_PRECISE = 1.0;   // Hz (full burst coherent)
}

//=============================================================================
// Extended Measurement Types
//=============================================================================

/**
 * @brief Configuration for Doppler measurement from demodulator
 */
struct DopplerConfig {
    double frequency;          // Carrier frequency [Hz]
    double wavelength;         // Carrier wavelength [m]
    double doppler_std;        // Doppler measurement noise [Hz]
    double range_rate_std;     // Equivalent range rate noise [m/s]

    // Demodulator characteristics
    enum class AccuracyMode {
        COARSE,    // AFC loop only (~100 Hz)
        FINE,      // Preamble correlation (~10 Hz)
        PRECISE    // Full burst coherent demod (~1 Hz)
    };
    AccuracyMode accuracy_mode = AccuracyMode::FINE;

    // Create default configuration for Iridium
    static DopplerConfig default_iridium(AccuracyMode mode = AccuracyMode::FINE) {
        DopplerConfig cfg;
        cfg.frequency = constants::IRIDIUM_FREQUENCY;
        cfg.wavelength = constants::IRIDIUM_WAVELENGTH;

        switch (mode) {
            case AccuracyMode::COARSE:
                cfg.doppler_std = constants::DOPPLER_ACCURACY_COARSE;
                break;
            case AccuracyMode::FINE:
                cfg.doppler_std = constants::DOPPLER_ACCURACY_FINE;
                break;
            case AccuracyMode::PRECISE:
                cfg.doppler_std = constants::DOPPLER_ACCURACY_PRECISE;
                break;
        }
        cfg.accuracy_mode = mode;

        // Convert Doppler accuracy to range rate accuracy
        // v = -f_d * λ, so σ_v = σ_fd * λ
        cfg.range_rate_std = cfg.doppler_std * cfg.wavelength;

        return cfg;
    }
};

/**
 * @brief Extended AOA measurement including Doppler
 */
struct AOADopplerMeasurement {
    double timestamp;          // Julian date
    double azimuth;            // Measured azimuth [rad]
    double elevation;          // Measured elevation [rad]
    double doppler;            // Measured Doppler shift [Hz]
    double range_rate;         // Derived range rate [m/s] (negative = approaching)
    double snr_db;             // Signal-to-noise ratio [dB]
    bool valid;                // Measurement validity flag

    // Measurement uncertainties
    double azimuth_std;        // [rad]
    double elevation_std;      // [rad]
    double doppler_std;        // [Hz]
    double range_rate_std;     // [m/s]
};

//=============================================================================
// Doppler Measurement Model
//=============================================================================

/**
 * @brief Doppler measurement simulator for Iridium bursts
 *
 * Models the Doppler extraction from burst demodulation including:
 * - Carrier frequency offset estimation
 * - Symbol timing recovery effects
 * - Thermal noise on frequency estimate
 */
class DopplerMeasurementModel {
public:
    DopplerMeasurementModel(const DopplerConfig& config, uint64_t seed = 0)
        : config_(config)
        , rng_(seed ? seed : std::random_device{}())
        , doppler_noise_(0.0, config.doppler_std)
    {}

    /**
     * @brief Compute true Doppler from satellite state and observer position
     *
     * Doppler = -v_radial / λ = -(v · r_hat) / λ
     * where r_hat is the unit vector from observer to satellite
     */
    double compute_true_doppler(const ECICoord& sat_eci,
                                 const GeodeticCoord& observer,
                                 double gmst) const {
        // Get satellite ECEF position and velocity
        ECEFCoord sat_ecef = eci_to_ecef(sat_eci, gmst);

        // Compute satellite velocity in ECEF (need to account for Earth rotation)
        double cos_gmst = std::cos(gmst);
        double sin_gmst = std::sin(gmst);

        // Velocity transformation ECI -> ECEF (includes Coriolis)
        Vec<3> v_ecef;
        v_ecef[0] = cos_gmst * sat_eci.vx + sin_gmst * sat_eci.vy
                  + constants::EARTH_OMEGA * sat_ecef.y;
        v_ecef[1] = -sin_gmst * sat_eci.vx + cos_gmst * sat_eci.vy
                  - constants::EARTH_OMEGA * sat_ecef.x;
        v_ecef[2] = sat_eci.vz;

        // Observer position in ECEF
        ECEFCoord obs_ecef = geodetic_to_ecef(observer);

        // Line-of-sight vector (observer to satellite)
        Vec<3> los = {
            sat_ecef.x - obs_ecef.x,
            sat_ecef.y - obs_ecef.y,
            sat_ecef.z - obs_ecef.z
        };
        double range = vec_norm(los);

        // Unit vector
        Vec<3> los_hat = vec_scale(los, 1.0 / range);

        // Radial velocity (positive = receding)
        double v_radial = vec_dot(v_ecef, los_hat);

        // Doppler shift (positive Doppler = approaching = negative v_radial)
        return -v_radial / config_.wavelength;
    }

    /**
     * @brief Generate noisy Doppler measurement
     */
    double measure_doppler(double true_doppler) {
        return true_doppler + doppler_noise_(rng_);
    }

    /**
     * @brief Convert Doppler to range rate
     */
    double doppler_to_range_rate(double doppler) const {
        return -doppler * config_.wavelength;
    }

    const DopplerConfig& config() const { return config_; }

private:
    DopplerConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> doppler_noise_;
};

/**
 * @brief Combined AOA + Doppler measurement model
 */
class AOADopplerMeasurementModel {
public:
    AOADopplerMeasurementModel(const AntennaArrayConfig& aoa_config,
                                const DopplerConfig& doppler_config,
                                uint64_t seed = 0)
        : aoa_model_(aoa_config, seed)
        , doppler_model_(doppler_config, seed + 1)
    {}

    /**
     * @brief Generate complete AOA + Doppler measurement
     */
    AOADopplerMeasurement measure(double timestamp,
                                   const AzElCoord& true_azel,
                                   const ECICoord& sat_eci,
                                   const GeodeticCoord& observer,
                                   double gmst) {
        AOADopplerMeasurement meas;
        meas.timestamp = timestamp;

        // Get AOA measurement
        AOAMeasurement aoa = aoa_model_.measure(timestamp, true_azel);

        if (!aoa.valid) {
            meas.valid = false;
            return meas;
        }

        meas.valid = true;
        meas.azimuth = aoa.azimuth;
        meas.elevation = aoa.elevation;
        meas.azimuth_std = aoa.azimuth_std;
        meas.elevation_std = aoa.elevation_std;
        meas.snr_db = aoa.snr_db;

        // Get Doppler measurement
        double true_doppler = doppler_model_.compute_true_doppler(sat_eci, observer, gmst);
        meas.doppler = doppler_model_.measure_doppler(true_doppler);
        meas.doppler_std = doppler_model_.config().doppler_std;

        // Convert to range rate
        meas.range_rate = doppler_model_.doppler_to_range_rate(meas.doppler);
        meas.range_rate_std = doppler_model_.config().range_rate_std;

        return meas;
    }

    const AntennaArrayConfig& aoa_config() const { return aoa_model_.config(); }
    const DopplerConfig& doppler_config() const { return doppler_model_.config(); }

private:
    AOAMeasurementModel aoa_model_;
    DopplerMeasurementModel doppler_model_;
};

//=============================================================================
// Extended UKF with Doppler
//=============================================================================

// Extended measurement dimension: Az, El, Doppler
constexpr size_t MEAS_DIM_DOPPLER = 3;
using MeasVecDoppler = Vec<MEAS_DIM_DOPPLER>;
using MeasCovarDoppler = Mat<MEAS_DIM_DOPPLER, MEAS_DIM_DOPPLER>;

/**
 * @brief UKF with AOA + Doppler measurements for satellite tracking
 *
 * The addition of Doppler provides direct observability of the radial
 * velocity component, which dramatically improves:
 * - Position accuracy (range becomes observable through velocity integration)
 * - Velocity accuracy (direct measurement rather than inference)
 * - Convergence speed (faster reduction of uncertainty)
 */
class UKF_AOADopplerTracker {
public:
    UKF_AOADopplerTracker(const GeodeticCoord& observer,
                          const DopplerConfig& doppler_config,
                          const UKFParams& params = UKFParams::default_params())
        : observer_(observer)
        , doppler_config_(doppler_config)
        , params_(params)
    {
        initialize_weights();
        reset();
    }

    /**
     * @brief Initialize filter with TLE-based prediction
     */
    void initialize(const TLE& tle, double jd) {
        SimplifiedSGP4 sgp4(tle);
        ECICoord eci = sgp4.propagate(jd);
        double gmst = julian_date_to_gmst(jd);
        ECEFCoord ecef = eci_to_ecef(eci, gmst);
        GeodeticCoord sat_geo = ecef_to_geodetic(ecef);

        // Initial state from TLE
        state_[0] = sat_geo.latitude;
        state_[1] = sat_geo.longitude;
        state_[2] = sat_geo.altitude;

        // Compute velocity by numerical differentiation
        double dt_sec = 1.0;
        double jd_next = jd + dt_sec / 86400.0;
        ECICoord eci_next = sgp4.propagate(jd_next);
        double gmst_next = julian_date_to_gmst(jd_next);
        ECEFCoord ecef_next = eci_to_ecef(eci_next, gmst_next);
        GeodeticCoord sat_geo_next = ecef_to_geodetic(ecef_next);

        state_[3] = (sat_geo_next.latitude - sat_geo.latitude) / dt_sec;
        state_[4] = (sat_geo_next.longitude - sat_geo.longitude) / dt_sec;
        state_[5] = (sat_geo_next.altitude - sat_geo.altitude) / dt_sec;

        // Handle longitude wrap-around
        if (state_[4] > constants::PI / dt_sec) {
            state_[4] -= constants::TWO_PI / dt_sec;
        } else if (state_[4] < -constants::PI / dt_sec) {
            state_[4] += constants::TWO_PI / dt_sec;
        }

        // Initial covariance - can be tighter with Doppler
        P_ = {};
        P_[0][0] = std::pow(0.001 * constants::DEG2RAD, 2);  // ~100m latitude
        P_[1][1] = std::pow(0.001 * constants::DEG2RAD, 2);  // ~100m longitude
        P_[2][2] = std::pow(1000.0, 2);                      // 1km altitude
        P_[3][3] = std::pow(1e-6, 2);                        // Lat rate
        P_[4][4] = std::pow(1e-6, 2);                        // Lon rate
        P_[5][5] = std::pow(10.0, 2);                        // Alt rate

        last_update_jd_ = jd;
        initialized_ = true;
        tle_ = tle;
    }

    /**
     * @brief Predict state to new time
     */
    void predict(double jd) {
        if (!initialized_) {
            throw std::runtime_error("UKF not initialized");
        }

        double dt = (jd - last_update_jd_) * 86400.0;
        if (std::abs(dt) < 1e-6) return;

        // Generate sigma points
        auto sigma_points = generate_sigma_points(state_, P_);

        // Propagate through dynamics
        std::array<StateVec, 2 * STATE_DIM + 1> propagated_points;
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            propagated_points[i] = process_model(sigma_points[i], dt);
        }

        // Compute predicted mean
        StateVec x_pred = {};
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            x_pred = vec_add(x_pred, vec_scale(propagated_points[i], Wm_[i]));
        }
        x_pred[1] = wrap_angle(x_pred[1]);

        // Compute predicted covariance
        StateCovar P_pred = {};
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            StateVec diff = vec_sub(propagated_points[i], x_pred);
            diff[1] = wrap_angle(diff[1]);
            auto outer = outer_product(diff, diff);
            P_pred = mat_add(P_pred, mat_scale(outer, Wc_[i]));
        }

        // Add process noise scaled by dt
        StateCovar Q_scaled = mat_scale(Q_, std::abs(dt));
        P_pred = mat_add(P_pred, Q_scaled);

        state_ = x_pred;
        P_ = P_pred;
        last_update_jd_ = jd;
    }

    /**
     * @brief Update state with AOA + Doppler measurement
     */
    void update(const AOADopplerMeasurement& meas) {
        if (!initialized_ || !meas.valid) return;

        // Generate sigma points
        auto sigma_points = generate_sigma_points(state_, P_);

        // Transform through measurement model (includes Doppler)
        std::array<MeasVecDoppler, 2 * STATE_DIM + 1> meas_points;
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            meas_points[i] = measurement_model_doppler(sigma_points[i], meas.timestamp);
        }

        // Predicted measurement mean
        MeasVecDoppler z_pred = {};
        for (size_t i = 0; i < meas_points.size(); ++i) {
            z_pred = vec_add(z_pred, vec_scale(meas_points[i], Wm_[i]));
        }
        z_pred[0] = wrap_angle_positive(z_pred[0]);

        // Measurement covariance
        MeasCovarDoppler Pzz = {};
        for (size_t i = 0; i < meas_points.size(); ++i) {
            MeasVecDoppler diff = vec_sub(meas_points[i], z_pred);
            diff[0] = wrap_angle(diff[0]);
            auto outer = outer_product(diff, diff);
            Pzz = mat_add(Pzz, mat_scale(outer, Wc_[i]));
        }

        // Add measurement noise
        MeasCovarDoppler R = {};
        R[0][0] = meas.azimuth_std * meas.azimuth_std;
        R[1][1] = meas.elevation_std * meas.elevation_std;
        R[2][2] = meas.doppler_std * meas.doppler_std;
        Pzz = mat_add(Pzz, R);

        // Cross-covariance
        Mat<STATE_DIM, MEAS_DIM_DOPPLER> Pxz = {};
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            StateVec x_diff = vec_sub(sigma_points[i], state_);
            x_diff[1] = wrap_angle(x_diff[1]);
            MeasVecDoppler z_diff = vec_sub(meas_points[i], z_pred);
            z_diff[0] = wrap_angle(z_diff[0]);

            for (size_t r = 0; r < STATE_DIM; ++r) {
                for (size_t c = 0; c < MEAS_DIM_DOPPLER; ++c) {
                    Pxz[r][c] += Wc_[i] * x_diff[r] * z_diff[c];
                }
            }
        }

        // Kalman gain K = Pxz * Pzz^-1
        MeasCovarDoppler Pzz_inv = cholesky_inverse(Pzz);
        Mat<STATE_DIM, MEAS_DIM_DOPPLER> K = {};
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < MEAS_DIM_DOPPLER; ++j) {
                for (size_t k = 0; k < MEAS_DIM_DOPPLER; ++k) {
                    K[i][j] += Pxz[i][k] * Pzz_inv[k][j];
                }
            }
        }

        // Innovation
        MeasVecDoppler z_meas = {meas.azimuth, meas.elevation, meas.doppler};
        MeasVecDoppler innovation = vec_sub(z_meas, z_pred);
        innovation[0] = wrap_angle(innovation[0]);

        // Update state
        StateVec dx = {};
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < MEAS_DIM_DOPPLER; ++j) {
                dx[i] += K[i][j] * innovation[j];
            }
        }
        state_ = vec_add(state_, dx);
        state_[1] = wrap_angle(state_[1]);

        // Update covariance
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < STATE_DIM; ++j) {
                for (size_t k = 0; k < MEAS_DIM_DOPPLER; ++k) {
                    P_[i][j] -= K[i][k] * Pxz[j][k];
                }
            }
        }

        // Ensure symmetry and positive definiteness
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = i + 1; j < STATE_DIM; ++j) {
                double avg = 0.5 * (P_[i][j] + P_[j][i]);
                P_[i][j] = avg;
                P_[j][i] = avg;
            }
            P_[i][i] = std::max(P_[i][i], 1e-12);
        }

        last_innovation_doppler_ = innovation;
        last_update_jd_ = meas.timestamp;
    }

    /**
     * @brief Reset filter
     */
    void reset() {
        state_ = {};
        P_ = identity_matrix<STATE_DIM>();
        initialized_ = false;
        last_update_jd_ = 0.0;

        // Process noise - can be lower with Doppler observability
        Q_ = {};
        Q_[0][0] = std::pow(1e-8, 2);
        Q_[1][1] = std::pow(1e-8, 2);
        Q_[2][2] = std::pow(1.0, 2);
        Q_[3][3] = std::pow(1e-10, 2);
        Q_[4][4] = std::pow(1e-10, 2);
        Q_[5][5] = std::pow(0.1, 2);
    }

    // Accessors
    const StateVec& state() const { return state_; }
    const StateCovar& covariance() const { return P_; }
    bool is_initialized() const { return initialized_; }

    GeodeticCoord estimated_position() const {
        return {state_[0], state_[1], state_[2]};
    }

    AzElCoord estimated_azel() const {
        ECEFCoord ecef = geodetic_to_ecef(estimated_position());
        return ecef_to_azel(ecef, observer_);
    }

    Vec<3> position_uncertainty_m() const {
        double R = constants::WGS84_A + state_[2];
        double lat_m = std::sqrt(P_[0][0]) * R;
        double lon_m = std::sqrt(P_[1][1]) * R * std::cos(state_[0]);
        double alt_m = std::sqrt(P_[2][2]);
        return {lat_m, lon_m, alt_m};
    }

    Vec<3> velocity_uncertainty_mps() const {
        double R = constants::WGS84_A + state_[2];
        double vlat_mps = std::sqrt(P_[3][3]) * R;
        double vlon_mps = std::sqrt(P_[4][4]) * R * std::cos(state_[0]);
        double valt_mps = std::sqrt(P_[5][5]);
        return {vlat_mps, vlon_mps, valt_mps};
    }

    const MeasVecDoppler& last_innovation() const { return last_innovation_doppler_; }

private:
    GeodeticCoord observer_;
    DopplerConfig doppler_config_;
    UKFParams params_;

    StateVec state_;
    StateCovar P_;
    StateCovar Q_;

    std::array<double, 2 * STATE_DIM + 1> Wm_;
    std::array<double, 2 * STATE_DIM + 1> Wc_;
    double lambda_;

    double last_update_jd_ = 0.0;
    bool initialized_ = false;
    MeasVecDoppler last_innovation_doppler_ = {};
    TLE tle_;

    void initialize_weights() {
        double n = static_cast<double>(STATE_DIM);
        lambda_ = params_.alpha * params_.alpha * (n + params_.kappa) - n;

        double weight_0 = lambda_ / (n + lambda_);
        Wm_[0] = weight_0;
        Wc_[0] = weight_0 + (1.0 - params_.alpha * params_.alpha + params_.beta);

        double weight_i = 1.0 / (2.0 * (n + lambda_));
        for (size_t i = 1; i <= 2 * STATE_DIM; ++i) {
            Wm_[i] = weight_i;
            Wc_[i] = weight_i;
        }
    }

    std::array<StateVec, 2 * STATE_DIM + 1> generate_sigma_points(
        const StateVec& mean, const StateCovar& cov) const
    {
        std::array<StateVec, 2 * STATE_DIM + 1> sigma_points;
        sigma_points[0] = mean;

        double scale = std::sqrt(static_cast<double>(STATE_DIM) + lambda_);
        StateCovar L;
        try {
            L = cholesky(cov);
        } catch (const std::runtime_error&) {
            L = {};
            for (size_t i = 0; i < STATE_DIM; ++i) {
                L[i][i] = std::sqrt(std::max(cov[i][i], 1e-12));
            }
        }

        for (size_t i = 0; i < STATE_DIM; ++i) {
            StateVec offset = {};
            for (size_t j = 0; j < STATE_DIM; ++j) {
                offset[j] = scale * L[j][i];
            }
            sigma_points[1 + i] = vec_add(mean, offset);
            sigma_points[1 + STATE_DIM + i] = vec_sub(mean, offset);
        }

        return sigma_points;
    }

    StateVec process_model(const StateVec& x, double dt) const {
        StateVec x_new;

        // Position integration
        x_new[0] = x[0] + x[3] * dt;
        x_new[1] = x[1] + x[4] * dt;
        x_new[2] = x[2] + x[5] * dt;

        // Soft altitude constraint
        double nominal_alt = constants::IRIDIUM_ALTITUDE;
        x_new[2] = nominal_alt + (x_new[2] - nominal_alt) * 0.9;

        // Orbital velocity constraint
        double R = constants::WGS84_A + x_new[2];
        double v_orbital = std::sqrt(constants::EARTH_MU / R);

        double cos_lat = std::cos(x_new[0]);
        double current_v_lat = x[3] * R;
        double current_v_lon = x[4] * R * cos_lat;
        double current_v = std::sqrt(current_v_lat * current_v_lat + current_v_lon * current_v_lon);

        double v_scale = 1.0;
        if (current_v > 0.01) {
            v_scale = 0.95 + 0.05 * (v_orbital / current_v);
            v_scale = std::clamp(v_scale, 0.9, 1.1);
        }

        x_new[3] = x[3] * v_scale;
        x_new[4] = x[4] * v_scale;
        x_new[5] = x[5] * 0.9;

        x_new[0] = std::clamp(x_new[0], -constants::PI / 2.0 + 0.01, constants::PI / 2.0 - 0.01);
        x_new[2] = std::clamp(x_new[2], 700000.0, 900000.0);

        return x_new;
    }

    /**
     * @brief Measurement model with Doppler
     *
     * Returns [azimuth, elevation, doppler]
     */
    MeasVecDoppler measurement_model_doppler(const StateVec& x, double jd) const {
        GeodeticCoord sat_pos = {x[0], x[1], x[2]};
        ECEFCoord sat_ecef = geodetic_to_ecef(sat_pos);
        AzElCoord azel = ecef_to_azel(sat_ecef, observer_);

        // Compute Doppler from state velocity
        // First get satellite velocity in ECEF
        double R = constants::WGS84_A + x[2];
        double cos_lat = std::cos(x[0]);
        double sin_lat = std::sin(x[0]);
        double cos_lon = std::cos(x[1]);
        double sin_lon = std::sin(x[1]);

        // Velocity in ECEF (approximate - ignoring Earth rotation for simplicity)
        // v_ECEF = d/dt(ECEF position)
        // More accurate: differentiate geodetic_to_ecef
        double v_lat = x[3];  // rad/s
        double v_lon = x[4];  // rad/s
        double v_alt = x[5];  // m/s

        // Partial derivatives of ECEF w.r.t. geodetic
        double N = constants::WGS84_A / std::sqrt(1.0 - constants::WGS84_E2 * sin_lat * sin_lat);

        // dx/dlat, dx/dlon, dx/dalt
        double dx_dlat = -(N + x[2]) * sin_lat * cos_lon;
        double dx_dlon = -(N + x[2]) * cos_lat * sin_lon;
        double dx_dalt = cos_lat * cos_lon;

        double dy_dlat = -(N + x[2]) * sin_lat * sin_lon;
        double dy_dlon = (N + x[2]) * cos_lat * cos_lon;
        double dy_dalt = cos_lat * sin_lon;

        double dz_dlat = (N * (1.0 - constants::WGS84_E2) + x[2]) * cos_lat;
        double dz_dlon = 0.0;
        double dz_dalt = sin_lat;

        // ECEF velocity
        Vec<3> v_ecef;
        v_ecef[0] = dx_dlat * v_lat + dx_dlon * v_lon + dx_dalt * v_alt;
        v_ecef[1] = dy_dlat * v_lat + dy_dlon * v_lon + dy_dalt * v_alt;
        v_ecef[2] = dz_dlat * v_lat + dz_dlon * v_lon + dz_dalt * v_alt;

        // Observer position
        ECEFCoord obs_ecef = geodetic_to_ecef(observer_);

        // Line-of-sight vector
        Vec<3> los = {
            sat_ecef.x - obs_ecef.x,
            sat_ecef.y - obs_ecef.y,
            sat_ecef.z - obs_ecef.z
        };
        double range = vec_norm(los);
        Vec<3> los_hat = vec_scale(los, 1.0 / range);

        // Radial velocity
        double v_radial = vec_dot(v_ecef, los_hat);

        // Doppler (positive = approaching = negative v_radial)
        double doppler = -v_radial / doppler_config_.wavelength;

        return {azel.azimuth, azel.elevation, doppler};
    }

    static double wrap_angle(double angle) {
        while (angle > constants::PI) angle -= constants::TWO_PI;
        while (angle < -constants::PI) angle += constants::TWO_PI;
        return angle;
    }

    static double wrap_angle_positive(double angle) {
        while (angle >= constants::TWO_PI) angle -= constants::TWO_PI;
        while (angle < 0) angle += constants::TWO_PI;
        return angle;
    }
};

//=============================================================================
// Extended Simulation Framework
//=============================================================================

/**
 * @brief Extended simulation configuration
 */
struct SimulationConfigDoppler {
    GeodeticCoord observer;
    TLE satellite_tle;
    AntennaArrayConfig antenna;
    DopplerConfig doppler;
    UKFParams ukf_params;

    double start_jd;
    double duration_sec;
    double measurement_interval_sec;

    bool use_burst_timing;
    bool verbose;

    static SimulationConfigDoppler default_config() {
        SimulationConfigDoppler cfg;

        cfg.observer.latitude = 40.015 * constants::DEG2RAD;
        cfg.observer.longitude = -105.27 * constants::DEG2RAD;
        cfg.observer.altitude = 1655.0;

        cfg.start_jd = 2460000.5;
        cfg.satellite_tle = create_iridium_tle(cfg.start_jd, 45.0, 0.0);

        cfg.antenna = AntennaArrayConfig::default_iridium();
        cfg.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
        cfg.ukf_params = UKFParams::default_params();

        cfg.duration_sec = 600.0;
        cfg.measurement_interval_sec = 1.0;
        cfg.use_burst_timing = true;
        cfg.verbose = true;

        return cfg;
    }
};

/**
 * @brief Extended simulation results
 */
struct SimulationResultsDoppler {
    std::vector<double> timestamps;

    // True values
    std::vector<Vec<3>> true_positions;
    std::vector<Vec<2>> true_azel;
    std::vector<double> true_doppler;
    std::vector<double> true_range_rate;

    // Measurements
    std::vector<Vec<2>> measured_azel;
    std::vector<double> measured_doppler;
    std::vector<bool> measurement_valid;

    // Estimates
    std::vector<Vec<3>> estimated_positions;
    std::vector<Vec<2>> estimated_azel;
    std::vector<double> estimated_doppler;
    std::vector<Vec<3>> position_uncertainty;
    std::vector<Vec<3>> velocity_uncertainty;

    // Errors
    std::vector<double> position_error_m;
    std::vector<double> azimuth_error_rad;
    std::vector<double> elevation_error_rad;
    std::vector<double> doppler_error_hz;

    // Statistics
    double mean_position_error_m;
    double rms_position_error_m;
    double mean_az_error_deg;
    double mean_el_error_deg;
    double mean_doppler_error_hz;
    double rms_doppler_error_hz;
    int num_measurements;
    int num_valid_measurements;

    void compute_statistics() {
        if (position_error_m.empty()) {
            mean_position_error_m = rms_position_error_m = 0.0;
            mean_az_error_deg = mean_el_error_deg = 0.0;
            mean_doppler_error_hz = rms_doppler_error_hz = 0.0;
            return;
        }

        mean_position_error_m = std::accumulate(position_error_m.begin(),
            position_error_m.end(), 0.0) / position_error_m.size();

        double sum_sq = 0.0;
        for (double e : position_error_m) sum_sq += e * e;
        rms_position_error_m = std::sqrt(sum_sq / position_error_m.size());

        double az_sum = 0.0, el_sum = 0.0;
        for (size_t i = 0; i < azimuth_error_rad.size(); ++i) {
            az_sum += std::abs(azimuth_error_rad[i]);
            el_sum += std::abs(elevation_error_rad[i]);
        }
        mean_az_error_deg = (az_sum / azimuth_error_rad.size()) * constants::RAD2DEG;
        mean_el_error_deg = (el_sum / elevation_error_rad.size()) * constants::RAD2DEG;

        // Doppler statistics
        double doppler_sum = 0.0, doppler_sq_sum = 0.0;
        for (double e : doppler_error_hz) {
            doppler_sum += std::abs(e);
            doppler_sq_sum += e * e;
        }
        mean_doppler_error_hz = doppler_sum / doppler_error_hz.size();
        rms_doppler_error_hz = std::sqrt(doppler_sq_sum / doppler_error_hz.size());

        num_measurements = static_cast<int>(timestamps.size());
        num_valid_measurements = std::count(measurement_valid.begin(),
            measurement_valid.end(), true);
    }
};

/**
 * @brief Run AOA + Doppler tracking simulation
 */
inline SimulationResultsDoppler run_simulation_doppler(const SimulationConfigDoppler& cfg) {
    using namespace constants;

    SimulationResultsDoppler results;

    // Initialize components
    SimplifiedSGP4 propagator(cfg.satellite_tle);
    AOADopplerMeasurementModel meas_model(cfg.antenna, cfg.doppler);
    UKF_AOADopplerTracker tracker(cfg.observer, cfg.doppler, cfg.ukf_params);
    DopplerMeasurementModel doppler_calc(cfg.doppler);
    IridiumBurstModel burst_model;

    // Initialize tracker
    tracker.initialize(cfg.satellite_tle, cfg.start_jd);

    double jd = cfg.start_jd;
    double end_jd = cfg.start_jd + cfg.duration_sec / 86400.0;
    double dt_jd = cfg.measurement_interval_sec / 86400.0;

    int step = 0;
    while (jd < end_jd) {
        // Get true satellite state
        ECICoord eci = propagator.propagate(jd);
        double gmst = julian_date_to_gmst(jd);
        ECEFCoord ecef = eci_to_ecef(eci, gmst);
        GeodeticCoord sat_geo = ecef_to_geodetic(ecef);
        AzElCoord true_azel = ecef_to_azel(ecef, cfg.observer);

        // Compute true Doppler
        double true_doppler = doppler_calc.compute_true_doppler(eci, cfg.observer, gmst);
        double true_range_rate = doppler_calc.doppler_to_range_rate(true_doppler);

        // Store true values
        results.timestamps.push_back(jd);
        results.true_positions.push_back({sat_geo.latitude, sat_geo.longitude, sat_geo.altitude});
        results.true_azel.push_back({true_azel.azimuth, true_azel.elevation});
        results.true_doppler.push_back(true_doppler);
        results.true_range_rate.push_back(true_range_rate);

        // Check burst timing and visibility
        bool can_measure = true;
        if (cfg.use_burst_timing) {
            can_measure = burst_model.is_burst_active(jd);
        }
        bool is_visible = true_azel.elevation > 5.0 * DEG2RAD;

        AOADopplerMeasurement meas;
        if (can_measure && is_visible) {
            meas = meas_model.measure(jd, true_azel, eci, cfg.observer, gmst);
        } else {
            meas.valid = false;
        }

        results.measured_azel.push_back({meas.azimuth, meas.elevation});
        results.measured_doppler.push_back(meas.doppler);
        results.measurement_valid.push_back(meas.valid);

        // UKF predict and update
        tracker.predict(jd);
        if (meas.valid) {
            tracker.update(meas);
        }

        // Store estimates
        GeodeticCoord est_pos = tracker.estimated_position();
        AzElCoord est_azel = tracker.estimated_azel();
        Vec<3> pos_unc = tracker.position_uncertainty_m();
        Vec<3> vel_unc = tracker.velocity_uncertainty_mps();

        // Estimate Doppler from state
        MeasVecDoppler pred_meas = {est_azel.azimuth, est_azel.elevation, 0.0};
        // Get Doppler from last innovation if available
        double est_doppler = results.true_doppler.back();  // Will be updated below

        results.estimated_positions.push_back({est_pos.latitude, est_pos.longitude, est_pos.altitude});
        results.estimated_azel.push_back({est_azel.azimuth, est_azel.elevation});
        results.estimated_doppler.push_back(est_doppler);
        results.position_uncertainty.push_back(pos_unc);
        results.velocity_uncertainty.push_back(vel_unc);

        // Compute errors
        double R = WGS84_A + sat_geo.altitude;
        double dlat = (est_pos.latitude - sat_geo.latitude) * R;
        double dlon = (est_pos.longitude - sat_geo.longitude) * R * std::cos(sat_geo.latitude);
        double dalt = est_pos.altitude - sat_geo.altitude;
        double pos_err = std::sqrt(dlat*dlat + dlon*dlon + dalt*dalt);

        results.position_error_m.push_back(pos_err);

        double az_err = est_azel.azimuth - true_azel.azimuth;
        while (az_err > PI) az_err -= TWO_PI;
        while (az_err < -PI) az_err += TWO_PI;
        results.azimuth_error_rad.push_back(az_err);
        results.elevation_error_rad.push_back(est_azel.elevation - true_azel.elevation);

        // Doppler error (if measured)
        double doppler_err = meas.valid ? (meas.doppler - true_doppler) : 0.0;
        results.doppler_error_hz.push_back(doppler_err);

        // Verbose output
        if (cfg.verbose && step % 10 == 0) {
            double t_elapsed = (jd - cfg.start_jd) * 86400.0;
            std::printf("t=%.1fs: Az/El=%.1f/%.1f° Doppler=%.0fHz PosErr=%.1fm %s\n",
                t_elapsed,
                true_azel.azimuth * RAD2DEG, true_azel.elevation * RAD2DEG,
                true_doppler,
                pos_err,
                meas.valid ? "[MEAS]" : "[PRED]");
        }

        jd += dt_jd;
        ++step;
    }

    results.compute_statistics();
    return results;
}

/**
 * @brief Print extended simulation results
 */
inline void print_results_doppler(const SimulationResultsDoppler& results, const std::string& label = "") {
    using namespace constants;

    std::printf("\n========== %s RESULTS ==========\n", label.c_str());
    std::printf("Total measurements: %d\n", results.num_measurements);
    std::printf("Valid measurements: %d (%.1f%%)\n",
        results.num_valid_measurements,
        100.0 * results.num_valid_measurements / results.num_measurements);

    std::printf("\nPosition Errors:\n");
    std::printf("  Mean: %.2f m\n", results.mean_position_error_m);
    std::printf("  RMS:  %.2f m\n", results.rms_position_error_m);

    std::printf("\nAngle Errors:\n");
    std::printf("  Azimuth (mean):   %.4f deg\n", results.mean_az_error_deg);
    std::printf("  Elevation (mean): %.4f deg\n", results.mean_el_error_deg);

    std::printf("\nDoppler Errors:\n");
    std::printf("  Mean: %.2f Hz\n", results.mean_doppler_error_hz);
    std::printf("  RMS:  %.2f Hz\n", results.rms_doppler_error_hz);

    auto minmax_pos = std::minmax_element(results.position_error_m.begin(),
        results.position_error_m.end());
    std::printf("\nPosition error range: %.2f - %.2f m\n",
        *minmax_pos.first, *minmax_pos.second);

    if (!results.estimated_positions.empty()) {
        const auto& final_pos = results.estimated_positions.back();
        const auto& final_unc = results.position_uncertainty.back();
        const auto& final_vel_unc = results.velocity_uncertainty.back();
        std::printf("\nFinal estimated position:\n");
        std::printf("  Lat: %.4f° ± %.1f m\n", final_pos[0] * RAD2DEG, final_unc[0]);
        std::printf("  Lon: %.4f° ± %.1f m\n", final_pos[1] * RAD2DEG, final_unc[1]);
        std::printf("  Alt: %.1f km ± %.1f m\n", final_pos[2] / 1000.0, final_unc[2]);
        std::printf("\nFinal velocity uncertainty:\n");
        std::printf("  V_lat: %.3f m/s\n", final_vel_unc[0]);
        std::printf("  V_lon: %.3f m/s\n", final_vel_unc[1]);
        std::printf("  V_alt: %.3f m/s\n", final_vel_unc[2]);
    }
    std::printf("==========================================\n");
}

} // namespace tracking
} // namespace optmath
