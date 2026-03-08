/**
 * @file AircraftNavSimulation.h
 * @brief Main Aircraft Navigation Simulation Orchestrator
 *
 * Orchestrates the complete simulation scenario:
 * 1. GPS/INS Integration (0-60s): Normal operation
 * 2. GPS Outage (60-90s): INS coasting with position drift
 * 3. Iridium Recovery (90-300s): AOA+Doppler updates
 *
 * Generates truth trajectories, simulates measurements, and
 * runs the navigation filter for performance analysis.
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>

#include "AircraftDynamicsModel.h"
#include "AircraftNavSRUKF.h"
#include "AircraftAntennaModel.h"
#include "MonteCarloRunner.h"

namespace AircraftNav {

/**
 * @brief Simulation scenario configuration
 */
struct SimulationConfig {
    // Timing
    double duration_s = 300.0;
    double dt_s = 0.01;  // Integration timestep
    double gps_outage_start_s = 60.0;
    double gps_outage_duration_s = 30.0;

    // Update rates
    double gps_rate_hz = 10.0;
    double iridium_rate_hz = 1.0;
    double imu_rate_hz = 100.0;

    // Aircraft configuration
    AircraftDynamicsConfig aircraft;

    // Filter configuration
    NavFilterConfig filter;

    // Antenna configuration
    AntennaConfig antenna;

    // Iridium constellation
    int num_satellites = 66;
    double satellite_altitude_m = 780000.0;

    // Scenario flags
    bool enable_gps = true;
    bool enable_iridium = true;
    bool enable_turbulence = true;

    // GPS validation thresholds (anti-spoofing/jamming)
    // GPS is only accepted if it agrees with Iridium navigation solution
    float gps_position_threshold_m = 100.0f;   // Max position offset [m]
    float gps_velocity_threshold_mps = 10.3f;  // Max velocity offset [m/s] (~20 knots)

    // Output
    bool save_trajectory = true;
    bool verbose = true;

    static SimulationConfig default_config() {
        SimulationConfig cfg;
        cfg.aircraft.latitude = 40.0 * M_PI / 180.0;
        cfg.aircraft.longitude = -105.0 * M_PI / 180.0;
        cfg.aircraft.altitude = 3048.0;
        cfg.aircraft.airspeed = 103.0;  // 200 knots
        cfg.aircraft.heading = 0.0;     // North
        return cfg;
    }
};

/**
 * @brief Simulation time series data
 */
struct SimulationTrajectory {
    std::vector<double> time;

    // True state
    std::vector<double> true_lat;
    std::vector<double> true_lon;
    std::vector<double> true_alt;
    std::vector<double> true_vn;
    std::vector<double> true_ve;
    std::vector<double> true_vd;
    std::vector<double> true_roll;
    std::vector<double> true_pitch;
    std::vector<double> true_yaw;

    // Estimated state
    std::vector<double> est_lat;
    std::vector<double> est_lon;
    std::vector<double> est_alt;
    std::vector<double> est_vn;
    std::vector<double> est_ve;
    std::vector<double> est_vd;
    std::vector<double> est_roll;
    std::vector<double> est_pitch;
    std::vector<double> est_yaw;

    // Errors
    std::vector<double> pos_error_m;
    std::vector<double> vel_error_mps;
    std::vector<double> heading_error_deg;

    // Filter status
    std::vector<int> nav_mode;
    std::vector<double> pos_uncertainty_m;
    std::vector<int> num_satellites;
    std::vector<bool> gps_available;
    std::vector<bool> iridium_available;
};

/**
 * @brief Simplified Iridium satellite model
 */
struct SimpleSatellite {
    int id;
    double raan_rad;       // Right ascension of ascending node
    double mean_anomaly_rad;
    double inclination_rad = 86.4 * M_PI / 180.0;

    /**
     * @brief Get satellite geodetic position at given time
     */
    void getPosition(double t, double& lat, double& lon, double& alt) const {
        const double ORBITAL_PERIOD = 100.4 * 60.0;  // 100.4 minutes
        const double ORBITAL_RATE = 2.0 * M_PI / ORBITAL_PERIOD;

        double M = mean_anomaly_rad + ORBITAL_RATE * t;
        double u = M;  // True anomaly ≈ mean anomaly for circular orbit

        // Satellite position in orbital plane
        alt = 780000.0;  // Constant altitude

        // Latitude from orbital geometry
        lat = std::asin(std::sin(inclination_rad) * std::sin(u));

        // Longitude (including Earth rotation)
        const double OMEGA_E = 7.292115e-5;
        double lambda_asc = raan_rad - OMEGA_E * t;
        lon = lambda_asc + std::atan2(std::cos(inclination_rad) * std::sin(u),
                                       std::cos(u));

        // Normalize longitude
        while (lon > M_PI) lon -= 2.0 * M_PI;
        while (lon < -M_PI) lon += 2.0 * M_PI;
    }

    /**
     * @brief Get satellite velocity (approximate)
     */
    void getVelocity(double t, double& v_lat, double& v_lon, double& v_alt) const {
        // Numerical differentiation
        double dt = 1.0;
        double lat1, lon1, alt1, lat2, lon2, alt2;
        getPosition(t - dt/2, lat1, lon1, alt1);
        getPosition(t + dt/2, lat2, lon2, alt2);

        v_lat = (lat2 - lat1) / dt;
        v_lon = (lon2 - lon1) / dt;
        v_alt = (alt2 - alt1) / dt;
    }
};

/**
 * @brief Aircraft Navigation Simulation
 *
 * Main class for running complete navigation scenarios.
 */
class AircraftNavSimulation {
public:
    explicit AircraftNavSimulation(const SimulationConfig& config, uint64_t seed = 0)
        : config_(config)
        , dynamics_(config.aircraft, seed)
        , filter_(config.filter)
        , antenna_(config.antenna, seed + 1)
        , rng_(seed + 2)
        , gps_noise_(0.0f, 1.0f)
    {
        initializeConstellation();
    }

    /**
     * @brief Run complete simulation
     * @return Simulation trajectory data
     *
     * Navigation Architecture:
     * - GPS provides initial state vector to nav computer (when valid)
     * - Iridium filter ALWAYS runs, continuously updating state
     * - During GPS jamming: flywheel with IMU (dead reckoning)
     * - After jamming: Iridium maintains accuracy
     */
    SimulationTrajectory run() {
        SimulationTrajectory traj;

        // Initialize dynamics
        dynamics_.reset();

        // Initialize filter with GPS-derived initial state
        auto state = dynamics_.state();
        Eigen::Vector3f gyro_bias = dynamics_.insErrors().getGyroBias();
        Eigen::Vector3f accel_bias = dynamics_.insErrors().getAccelBias();
        auto x0 = state.toNavState(gyro_bias, accel_bias);
        filter_.initialize(x0);

        // Timing
        double t = 0.0;
        double gps_timer = 0.0;
        double iridium_timer = 0.0;
        double output_timer = 0.0;

        double gps_interval = 1.0 / config_.gps_rate_hz;
        double iridium_interval = 1.0 / config_.iridium_rate_hz;
        double output_interval = 0.1;  // 10 Hz output

        // State tracking
        bool gps_jammed = !config_.enable_gps;
        int gps_update_count = 0;
        int iridium_update_count = 0;
        int iridium_meas_count = 0;

        // Jamming period
        double outage_end_s = config_.gps_outage_start_s + config_.gps_outage_duration_s;

        while (t < config_.duration_s) {
            // Propagate dynamics (truth trajectory)
            dynamics_.propagate(config_.dt_s);

            // Get IMU measurements
            Eigen::Vector3f gyro, accel;
            dynamics_.getCorruptedIMU(gyro, accel);

            // IMU propagation (flywheel) - always runs
            filter_.predict(gyro, accel, static_cast<float>(config_.dt_s));

            // Determine if GPS is jammed
            bool in_jamming = (t >= config_.gps_outage_start_s && t < outage_end_s);
            bool was_jammed = gps_jammed;
            gps_jammed = !config_.enable_gps || in_jamming;

            // Notify filter of GPS outage (copies GPS state to Iridium filter)
            if (gps_jammed && !was_jammed) {
                filter_.notifyGPSOutage();
            }

            // GPS update (when not jammed) - provides state to nav computer
            if (config_.enable_gps && !in_jamming && gps_timer >= gps_interval) {
                auto gps_meas = generateGPSMeasurement();
                filter_.updateGPS(gps_meas);
                gps_update_count++;
                gps_timer = 0.0;
            }
            if (config_.enable_gps) {
                gps_timer += config_.dt_s;
            }

            // Iridium update - only during GPS phase or recovery (NOT during outage)
            // During outage: IMU dead reckoning only
            // After outage: Iridium updates for recovery
            bool iridium_active = !in_jamming && config_.enable_iridium;
            if (iridium_active && iridium_timer >= iridium_interval) {
                auto iridium_meas = generateIridiumMeasurements();
                iridium_meas_count += iridium_meas.size();
                if (!iridium_meas.empty()) {
                    filter_.updateIridium(iridium_meas);
                    iridium_update_count++;
                }
                iridium_timer = 0.0;
            }
            iridium_timer += config_.dt_s;

            // Record output
            if (output_timer >= output_interval) {
                recordState(traj, t, !gps_jammed, true);
                output_timer = 0.0;
            }
            output_timer += config_.dt_s;

            t += config_.dt_s;
        }

        if (config_.verbose) {
            std::cout << "Simulation stats:\n"
                      << "  GPS updates: " << gps_update_count << "\n"
                      << "  Iridium updates: " << iridium_update_count << "\n"
                      << "  Iridium measurements: " << iridium_meas_count << "\n"
                      << "  (During jamming: IMU flywheel only)\n";
        }

        return traj;
    }

    /**
     * @brief Run single trial for Monte Carlo
     */
    TrialResult runTrial(int trial_id, uint64_t seed) {
        // Reinitialize with new seed
        dynamics_ = AircraftDynamicsModel(config_.aircraft, seed);
        antenna_ = AircraftAntennaModel(config_.antenna, seed + 1);
        rng_.seed(seed + 2);

        // Run simulation
        SimulationTrajectory traj = run();

        // Compute trial results
        return computeTrialResult(trial_id, seed, traj);
    }

    /**
     * @brief Save trajectory to CSV
     */
    void saveTrajectoryCSV(const std::string& filename, const SimulationTrajectory& traj) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Header
        file << "time,true_lat,true_lon,true_alt,true_vn,true_ve,true_vd,"
             << "est_lat,est_lon,est_alt,est_vn,est_ve,est_vd,"
             << "pos_error_m,vel_error_mps,heading_error_deg,"
             << "nav_mode,pos_uncertainty_m,num_sats,gps,iridium\n";

        // Data
        for (size_t i = 0; i < traj.time.size(); ++i) {
            file << std::fixed << std::setprecision(6)
                 << traj.time[i] << ","
                 << traj.true_lat[i] << "," << traj.true_lon[i] << "," << traj.true_alt[i] << ","
                 << traj.true_vn[i] << "," << traj.true_ve[i] << "," << traj.true_vd[i] << ","
                 << traj.est_lat[i] << "," << traj.est_lon[i] << "," << traj.est_alt[i] << ","
                 << traj.est_vn[i] << "," << traj.est_ve[i] << "," << traj.est_vd[i] << ","
                 << std::setprecision(2)
                 << traj.pos_error_m[i] << "," << traj.vel_error_mps[i] << ","
                 << traj.heading_error_deg[i] << ","
                 << traj.nav_mode[i] << "," << traj.pos_uncertainty_m[i] << ","
                 << traj.num_satellites[i] << ","
                 << (traj.gps_available[i] ? 1 : 0) << ","
                 << (traj.iridium_available[i] ? 1 : 0) << "\n";
        }

        file.close();
        std::cout << "Saved trajectory to " << filename << std::endl;
    }

private:
    SimulationConfig config_;
    AircraftDynamicsModel dynamics_;
    AircraftNavSRUKF filter_;
    AircraftAntennaModel antenna_;
    std::vector<SimpleSatellite> constellation_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> gps_noise_;

    void initializeConstellation() {
        constellation_.clear();

        const int NUM_PLANES = 6;
        const int SATS_PER_PLANE = 11;

        int id = 0;
        for (int plane = 0; plane < NUM_PLANES; ++plane) {
            double raan = plane * (2.0 * M_PI / NUM_PLANES);
            double phase_offset = plane * (M_PI / NUM_PLANES / SATS_PER_PLANE);

            for (int slot = 0; slot < SATS_PER_PLANE; ++slot) {
                SimpleSatellite sat;
                sat.id = id++;
                sat.raan_rad = raan;
                sat.mean_anomaly_rad = slot * (2.0 * M_PI / SATS_PER_PLANE) + phase_offset;
                constellation_.push_back(sat);
            }
        }
    }

    Eigen::Matrix<float, 6, 1> generateGPSMeasurement() {
        const auto& state = dynamics_.state();

        Eigen::Matrix<float, 6, 1> meas;

        // True position with noise
        // Altitude: GPS + topographic map gives accurate aerial height (AGL)
        float R_M = 6371000.0f;
        meas(0) = static_cast<float>(state.lat) + (4.0f / R_M) * gps_noise_(rng_);
        meas(1) = static_cast<float>(state.lon) + (4.0f / R_M) * gps_noise_(rng_);
        meas(2) = static_cast<float>(state.alt) + 1.0f * gps_noise_(rng_);  // 1m with topo map

        // True velocity with noise
        meas(3) = static_cast<float>(state.v_n) + 0.1f * gps_noise_(rng_);
        meas(4) = static_cast<float>(state.v_e) + 0.1f * gps_noise_(rng_);
        meas(5) = static_cast<float>(state.v_d) + 0.15f * gps_noise_(rng_);

        return meas;
    }

    /**
     * @brief Validate GPS measurement against Iridium navigation solution
     *
     * GPS is only accepted if it agrees with the filter (Iridium) estimate:
     * - Position offset < 100m
     * - Velocity offset < 20 knots (~10.3 m/s)
     *
     * @param gps_meas GPS measurement [lat, lon, alt, vn, ve, vd]
     * @return true if GPS is valid (agrees with Iridium), false if spoofed/jammed
     */
    bool validateGPSMeasurement(const Eigen::Matrix<float, 6, 1>& gps_meas) {
        auto filter_state = filter_.getState();
        const float R_M = 6371000.0f;

        // Position error (meters)
        float dlat_m = (gps_meas(0) - filter_state(LAT)) * R_M;
        float dlon_m = (gps_meas(1) - filter_state(LON)) * R_M *
                       std::cos(filter_state(LAT));
        float dalt_m = gps_meas(2) - filter_state(ALT);
        float pos_error = std::sqrt(dlat_m*dlat_m + dlon_m*dlon_m + dalt_m*dalt_m);

        // Velocity error (m/s)
        float dvn = gps_meas(3) - filter_state(VN);
        float dve = gps_meas(4) - filter_state(VE);
        float dvd = gps_meas(5) - filter_state(VD);
        float vel_error = std::sqrt(dvn*dvn + dve*dve + dvd*dvd);

        // Validate against thresholds
        bool pos_valid = (pos_error < config_.gps_position_threshold_m);
        bool vel_valid = (vel_error < config_.gps_velocity_threshold_mps);

        return pos_valid && vel_valid;
    }

    std::vector<IridiumMeasurement> generateIridiumMeasurements() {
        std::vector<IridiumMeasurement> measurements;

        const auto& state = dynamics_.state();
        double t = dynamics_.time();

        for (const auto& sat : constellation_) {
            double sat_lat, sat_lon, sat_alt;
            sat.getPosition(t, sat_lat, sat_lon, sat_alt);

            // Check visibility
            auto vis = antenna_.checkVisibility(
                static_cast<float>(state.lat), static_cast<float>(state.lon),
                static_cast<float>(state.alt),
                static_cast<float>(state.roll), static_cast<float>(state.pitch),
                static_cast<float>(state.yaw),
                static_cast<float>(sat_lat), static_cast<float>(sat_lon),
                static_cast<float>(sat_alt));

            if (vis.visible) {
                IridiumMeasurement meas;
                meas.satellite_id = sat.id;
                meas.valid = true;
                meas.snr_dB = vis.snr_dB;

                // Get noisy AOA measurement
                auto [az, el] = antenna_.measureAOA(vis);
                meas.azimuth_rad = az;
                meas.elevation_rad = el;

                // Compute Doppler
                double v_lat, v_lon, v_alt;
                sat.getVelocity(t, v_lat, v_lon, v_alt);

                // Simplified Doppler (based on relative radial velocity)
                float doppler = computeDoppler(state, sat_lat, sat_lon, sat_alt,
                                                v_lat, v_lon, v_alt);
                meas.doppler_Hz = doppler + 10.0f * gps_noise_(rng_);  // Add noise

                // Set satellite position for measurement model
                meas.satellite_pos.lat = static_cast<float>(sat_lat);
                meas.satellite_pos.lon = static_cast<float>(sat_lon);
                meas.satellite_pos.alt = static_cast<float>(sat_alt);
                meas.satellite_pos.v_lat = static_cast<float>(v_lat);
                meas.satellite_pos.v_lon = static_cast<float>(v_lon);
                meas.satellite_pos.v_alt = static_cast<float>(v_alt);

                measurements.push_back(meas);
            }
        }

        return measurements;
    }

    float computeDoppler(const AircraftState& ac,
                         double sat_lat, double sat_lon, double sat_alt,
                         double sat_v_lat, double sat_v_lon, double sat_v_alt) const {
        const float WAVELENGTH = 0.1844f;
        const float R_M = 6371000.0f;

        // Simplified relative velocity in radial direction
        double dlat = sat_lat - ac.lat;
        double dlon = sat_lon - ac.lon;
        double dalt = sat_alt - ac.alt;

        double range = std::sqrt(
            std::pow(dlat * R_M, 2) +
            std::pow(dlon * R_M * std::cos(ac.lat), 2) +
            std::pow(dalt, 2));

        // Unit vector
        double u_lat = dlat * R_M / range;
        double u_lon = dlon * R_M * std::cos(ac.lat) / range;
        double u_alt = dalt / range;

        // Relative velocity
        double dv_lat = sat_v_lat * R_M - ac.v_n;
        double dv_lon = sat_v_lon * R_M * std::cos(sat_lat) - ac.v_e;
        double dv_alt = -sat_v_alt - ac.v_d;

        // Radial velocity
        double v_radial = u_lat * dv_lat + u_lon * dv_lon + u_alt * dv_alt;

        // Doppler (positive = approaching)
        return static_cast<float>(-v_radial / WAVELENGTH);
    }

    /**
     * @brief Initialize filter for recovery phase
     *
     * Called at start of recovery (after 30s outage). Initializes filter
     * with last known GPS position but large uncertainty to account for
     * drift during outage.
     */
    void initializeFilterForRecovery(const Eigen::Matrix<float, 6, 1>& last_gps) {
        // Initialize state with last known GPS position
        // But with large uncertainty (aircraft has drifted ~3km in 30s at 100 m/s)
        Eigen::Matrix<float, 15, 1> x0;
        x0 << last_gps(0), last_gps(1), last_gps(2),  // Position from last GPS
              last_gps(3), last_gps(4), last_gps(5),  // Velocity from last GPS
              0.0f, 0.0f, 0.0f,                        // Attitude (assume level)
              0.0f, 0.0f, 0.0f,                        // Gyro bias
              0.0f, 0.0f, 0.0f;                        // Accel bias

        filter_.initializeForRecovery(x0);
    }

    /**
     * @brief Record state for trajectory output
     *
     * @param traj Output trajectory
     * @param t Current time
     * @param gps_valid True if GPS is accepted (not jammed/spoofed)
     * @param filter_running Always true in new architecture
     */
    void recordState(SimulationTrajectory& traj, double t,
                     bool gps_valid, bool filter_running) {
        const auto& true_state = dynamics_.state();

        traj.time.push_back(t);

        // True state (always available)
        traj.true_lat.push_back(true_state.lat);
        traj.true_lon.push_back(true_state.lon);
        traj.true_alt.push_back(true_state.alt);
        traj.true_vn.push_back(true_state.v_n);
        traj.true_ve.push_back(true_state.v_e);
        traj.true_vd.push_back(true_state.v_d);
        traj.true_roll.push_back(true_state.roll);
        traj.true_pitch.push_back(true_state.pitch);
        traj.true_yaw.push_back(true_state.yaw);

        float R_M = 6371000.0f;

        // Filter always runs (Iridium is primary navigation)
        // GPS valid flag indicates whether GPS is accepted or rejected (jammed)
        auto est_state = filter_.getState();
        auto status = filter_.getStatus();

        traj.est_lat.push_back(est_state(LAT));
        traj.est_lon.push_back(est_state(LON));
        traj.est_alt.push_back(est_state(ALT));
        traj.est_vn.push_back(est_state(VN));
        traj.est_ve.push_back(est_state(VE));
        traj.est_vd.push_back(est_state(VD));
        traj.est_roll.push_back(est_state(ROLL));
        traj.est_pitch.push_back(est_state(PITCH));
        traj.est_yaw.push_back(est_state(YAW));

        // Compute errors
        float dlat_m = static_cast<float>(est_state(LAT) - true_state.lat) * R_M;
        float dlon_m = static_cast<float>(est_state(LON) - true_state.lon) * R_M *
                       std::cos(static_cast<float>(true_state.lat));
        float dalt_m = static_cast<float>(est_state(ALT) - true_state.alt);
        float pos_err = std::sqrt(dlat_m*dlat_m + dlon_m*dlon_m + dalt_m*dalt_m);
        traj.pos_error_m.push_back(pos_err);

        float dvn = static_cast<float>(est_state(VN) - true_state.v_n);
        float dve = static_cast<float>(est_state(VE) - true_state.v_e);
        float dvd = static_cast<float>(est_state(VD) - true_state.v_d);
        traj.vel_error_mps.push_back(std::sqrt(dvn*dvn + dve*dve + dvd*dvd));

        float dyaw = static_cast<float>(est_state(YAW) - true_state.yaw);
        while (dyaw > M_PI) dyaw -= 2.0f * M_PI;
        while (dyaw < -M_PI) dyaw += 2.0f * M_PI;
        traj.heading_error_deg.push_back(std::abs(dyaw) * 180.0f / M_PI);

        // Nav mode: 0 = GPS+Iridium, 1 = Iridium only (GPS jammed)
        traj.nav_mode.push_back(gps_valid ? 0 : 1);
        traj.pos_uncertainty_m.push_back(status.position_uncertainty_m);
        traj.num_satellites.push_back(status.num_visible_satellites);

        // GPS valid indicates whether GPS was accepted (not jammed/spoofed)
        traj.gps_available.push_back(gps_valid);
        traj.iridium_available.push_back(true);  // Iridium always active
    }

    TrialResult computeTrialResult(int trial_id, uint64_t seed,
                                    const SimulationTrajectory& traj) const {
        TrialResult result;
        result.trial_id = trial_id;
        result.seed = seed;

        // Find key time indices
        size_t outage_start_idx = 0;
        size_t outage_end_idx = 0;

        for (size_t i = 0; i < traj.time.size(); ++i) {
            if (traj.time[i] >= config_.gps_outage_start_s && outage_start_idx == 0) {
                outage_start_idx = i;
            }
            double outage_end_time = config_.gps_outage_start_s + config_.gps_outage_duration_s;
            if (traj.time[i] >= outage_end_time && outage_end_idx == 0) {
                outage_end_idx = i;
            }
        }

        result.error_at_outage_start_m = traj.pos_error_m[outage_start_idx];
        result.error_at_outage_end_m = traj.pos_error_m[outage_end_idx];
        result.final_error_m = traj.pos_error_m.back();

        // Max error
        result.max_error_m = *std::max_element(traj.pos_error_m.begin(), traj.pos_error_m.end());

        // Convergence time (time after outage to reach threshold)
        result.converged = false;
        result.convergence_time_s = config_.duration_s;
        double threshold = 500.0;  // meters

        for (size_t i = outage_end_idx; i < traj.time.size(); ++i) {
            if (traj.pos_error_m[i] < threshold) {
                result.converged = true;
                result.convergence_time_s = traj.time[i] -
                    (config_.gps_outage_start_s + config_.gps_outage_duration_s);
                break;
            }
        }

        // Check for divergence
        result.diverged = result.max_error_m > 5000.0;

        // RMSE by phase
        result.rmse_gps_phase_m = computeRMSE(traj.pos_error_m, 0, outage_start_idx);
        result.rmse_outage_phase_m = computeRMSE(traj.pos_error_m, outage_start_idx, outage_end_idx);
        result.rmse_recovery_phase_m = computeRMSE(traj.pos_error_m, outage_end_idx, traj.pos_error_m.size());

        // Velocity errors
        result.max_velocity_error_mps = *std::max_element(
            traj.vel_error_mps.begin(), traj.vel_error_mps.end());
        result.final_velocity_error_mps = traj.vel_error_mps.back();

        // Count Iridium measurements (approximation)
        result.num_iridium_measurements = 0;
        result.num_visible_satellites_avg = 0;
        int count = 0;
        for (size_t i = 0; i < traj.iridium_available.size(); ++i) {
            if (traj.iridium_available[i]) {
                result.num_visible_satellites_avg += traj.num_satellites[i];
                count++;
            }
        }
        if (count > 0) {
            result.num_visible_satellites_avg /= count;
            result.num_iridium_measurements = count;
        }

        return result;
    }

    static double computeRMSE(const std::vector<double>& errors,
                               size_t start, size_t end) {
        if (end <= start) return 0.0;

        double sum_sq = 0.0;
        for (size_t i = start; i < end; ++i) {
            sum_sq += errors[i] * errors[i];
        }
        return std::sqrt(sum_sq / (end - start));
    }
};

} // namespace AircraftNav
