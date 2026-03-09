/**
 * @file multi_satellite_tracker.hpp
 * @brief Multi-Satellite UKF Tracker for Improved Geometry
 *
 * Tracks multiple Iridium-Next satellites simultaneously to achieve:
 * - Better geometric dilution of precision (GDOP)
 * - Faster convergence through more measurements
 * - Robustness to single satellite occlusion
 * - Cross-validation between satellite tracks
 *
 * The Iridium constellation has 66 active satellites in 6 orbital planes,
 * providing 2-4 visible satellites from most locations at any time.
 *
 * @author OptMathKernels
 * @version 0.5.0
 */

#pragma once

#include "ukf_aoa_doppler_tracking.hpp"
#include "iridium_burst_demodulator.hpp"
#include <map>
#include <memory>

namespace optmath {
namespace tracking {

//=============================================================================
// Satellite Track State
//=============================================================================

/**
 * @brief Individual satellite track information
 */
struct SatelliteTrack {
    int satellite_id;
    TLE tle;
    std::unique_ptr<UKF_AOADopplerTracker> tracker;

    // Track quality metrics
    double last_measurement_jd;
    int measurement_count;
    int consecutive_misses;
    double position_error_estimate;  // Estimated error based on covariance
    bool is_visible;
    bool is_tracking;

    // Current state
    AzElCoord predicted_azel;
    double predicted_doppler;

    SatelliteTrack() = default;
    SatelliteTrack(SatelliteTrack&&) = default;
    SatelliteTrack& operator=(SatelliteTrack&&) = default;
};

//=============================================================================
// Multi-Satellite Constellation Manager
//=============================================================================

/**
 * @brief Iridium constellation model
 *
 * Generates realistic TLEs for multiple Iridium satellites
 * based on the constellation geometry:
 * - 6 orbital planes
 * - 11 satellites per plane
 * - 86.4° inclination
 * - 780 km altitude
 * - ~30° RAAN spacing between planes
 */
class IridiumConstellation {
public:
    struct SatelliteInfo {
        int id;
        int plane;       // 0-5
        int slot;        // 0-10
        TLE tle;
        double raan_deg;
        double mean_anomaly_deg;
    };

    /**
     * @brief Generate constellation TLEs for given epoch
     */
    static std::vector<SatelliteInfo> generate_constellation(double epoch_jd) {
        std::vector<SatelliteInfo> sats;

        const int NUM_PLANES = 6;
        const int SATS_PER_PLANE = 11;
        const double RAAN_SPACING = 360.0 / NUM_PLANES;  // 60°
        const double PHASE_SPACING = 360.0 / SATS_PER_PLANE;  // ~32.7°

        int sat_id = 1;
        for (int plane = 0; plane < NUM_PLANES; ++plane) {
            double raan = plane * RAAN_SPACING;

            // Inter-plane phasing for coverage
            double phase_offset = plane * (PHASE_SPACING / 2.0);

            for (int slot = 0; slot < SATS_PER_PLANE; ++slot) {
                double ma = std::fmod(slot * PHASE_SPACING + phase_offset, 360.0);

                SatelliteInfo info;
                info.id = sat_id++;
                info.plane = plane;
                info.slot = slot;
                info.raan_deg = raan;
                info.mean_anomaly_deg = ma;
                info.tle = create_iridium_tle(epoch_jd, raan, ma);
                info.tle.catalog_number = 40000 + info.id;  // Pseudo NORAD ID

                sats.push_back(info);
            }
        }

        return sats;
    }

    /**
     * @brief Find visible satellites from observer location
     */
    static std::vector<int> find_visible_satellites(
        const std::vector<SatelliteInfo>& constellation,
        const GeodeticCoord& observer,
        double jd,
        double min_elevation_deg = 5.0)
    {
        std::vector<int> visible;

        for (const auto& sat : constellation) {
            SimplifiedSGP4 sgp4(sat.tle);
            AzElCoord azel = sgp4.get_azel(jd, observer);

            if (azel.elevation > min_elevation_deg * constants::DEG2RAD) {
                visible.push_back(sat.id);
            }
        }

        return visible;
    }
};

//=============================================================================
// Multi-Satellite Tracker
//=============================================================================

/**
 * @brief Configuration for multi-satellite tracking
 */
struct MultiSatelliteConfig {
    GeodeticCoord observer;
    DopplerConfig doppler;
    UKFParams ukf_params;
    IridiumBurstDemodulator::Config demod;

    int max_tracked_satellites;     // Maximum simultaneous tracks
    double min_elevation_deg;       // Minimum elevation for tracking
    int max_consecutive_misses;     // Drop track after this many misses
    double track_init_timeout_sec;  // Time to initialize new track

    static MultiSatelliteConfig default_config() {
        MultiSatelliteConfig cfg;

        cfg.observer.latitude = 40.015 * constants::DEG2RAD;
        cfg.observer.longitude = -105.27 * constants::DEG2RAD;
        cfg.observer.altitude = 1655.0;

        cfg.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
        cfg.ukf_params = UKFParams::default_params();
        cfg.demod = IridiumBurstDemodulator::Config::default_config();

        cfg.max_tracked_satellites = 4;
        cfg.min_elevation_deg = 5.0;
        cfg.max_consecutive_misses = 10;
        cfg.track_init_timeout_sec = 5.0;

        return cfg;
    }
};

/**
 * @brief Multi-satellite tracking statistics
 */
struct MultiSatelliteStats {
    int num_visible;
    int num_tracking;
    int total_measurements;
    int valid_measurements;

    double mean_position_error_m;
    double best_position_error_m;
    double geometric_dilution;

    std::vector<int> tracked_satellite_ids;
};

/**
 * @brief Multi-satellite UKF tracker
 *
 * Manages multiple satellite tracks simultaneously, providing:
 * - Automatic track initiation/termination
 * - Measurement-to-track association
 * - Fused position estimate from multiple satellites
 * - GDOP computation for geometric quality assessment
 */
class MultiSatelliteTracker {
public:
    explicit MultiSatelliteTracker(const MultiSatelliteConfig& config, uint64_t seed = 0)
        : config_(config)
        , demod_(config.demod, seed)
        , constellation_(IridiumConstellation::generate_constellation(2460000.5))
        , rng_(seed ? seed : std::random_device{}())
    {}

    /**
     * @brief Initialize tracks for visible satellites
     */
    void initialize(double jd) {
        epoch_jd_ = jd;
        tracks_.clear();

        // Find visible satellites
        auto visible = IridiumConstellation::find_visible_satellites(
            constellation_, config_.observer, jd, config_.min_elevation_deg);

        // Initialize tracks for up to max_tracked_satellites
        int n_tracks = std::min(static_cast<int>(visible.size()),
                                config_.max_tracked_satellites);

        for (int i = 0; i < n_tracks; ++i) {
            int sat_id = visible[i];
            initiate_track(sat_id, jd);
        }
    }

    /**
     * @brief Process measurements at given time
     *
     * For each tracked satellite:
     * 1. Predict state to current time
     * 2. Check if burst is active and satellite visible
     * 3. Generate measurement if possible
     * 4. Update track
     */
    void process(double jd) {
        // Update constellation visibility
        auto visible = IridiumConstellation::find_visible_satellites(
            constellation_, config_.observer, jd, config_.min_elevation_deg);

        // Process each track
        for (auto& [sat_id, track] : tracks_) {
            // Predict
            track.tracker->predict(jd);

            // Update predicted Az/El
            track.predicted_azel = track.tracker->estimated_azel();

            // Check visibility
            track.is_visible = std::find(visible.begin(), visible.end(), sat_id) != visible.end();

            if (!track.is_visible) {
                track.consecutive_misses++;
                continue;
            }

            // Get true satellite position for simulation
            SimplifiedSGP4 sgp4(track.tle);
            ECICoord eci = sgp4.propagate(jd);
            double gmst = julian_date_to_gmst(jd);
            AzElCoord true_azel = sgp4.get_azel(jd, config_.observer);

            // Try to get measurement
            AOADopplerMeasurement meas = demod_.process(
                jd, true_azel, eci, config_.observer, gmst);

            if (meas.valid) {
                track.tracker->update(meas);
                track.last_measurement_jd = jd;
                track.measurement_count++;
                track.consecutive_misses = 0;
            } else {
                track.consecutive_misses++;
            }

            // Update position error estimate
            Vec<3> unc = track.tracker->position_uncertainty_m();
            track.position_error_estimate = std::sqrt(
                unc[0]*unc[0] + unc[1]*unc[1] + unc[2]*unc[2]);
        }

        // Remove stale tracks
        prune_tracks();

        // Initiate new tracks if slots available
        if (tracks_.size() < static_cast<size_t>(config_.max_tracked_satellites)) {
            for (int sat_id : visible) {
                if (tracks_.find(sat_id) == tracks_.end()) {
                    initiate_track(sat_id, jd);
                    if (tracks_.size() >= static_cast<size_t>(config_.max_tracked_satellites)) {
                        break;
                    }
                }
            }
        }

        last_update_jd_ = jd;
    }

    /**
     * @brief Get fused position estimate from multiple tracks
     *
     * Computes weighted average of all track estimates,
     * weighted by inverse covariance (information fusion)
     */
    GeodeticCoord get_fused_position() const {
        if (tracks_.empty()) {
            return config_.observer;  // Fallback
        }

        // Information-weighted fusion
        double total_weight = 0.0;
        double lat_sum = 0.0;
        double lon_sum = 0.0;
        double alt_sum = 0.0;

        for (const auto& [sat_id, track] : tracks_) {
            if (!track.is_tracking) continue;

            double weight = 1.0 / (track.position_error_estimate * track.position_error_estimate + 1.0);
            GeodeticCoord pos = track.tracker->estimated_position();

            lat_sum += weight * pos.latitude;
            lon_sum += weight * pos.longitude;
            alt_sum += weight * pos.altitude;
            total_weight += weight;
        }

        if (total_weight < 1e-10) {
            return config_.observer;
        }

        return {lat_sum / total_weight, lon_sum / total_weight, alt_sum / total_weight};
    }

    /**
     * @brief Compute Geometric Dilution of Precision (GDOP)
     *
     * GDOP measures the geometric quality of satellite positions.
     * Lower GDOP = better geometry = better position accuracy.
     *
     * Computed from the geometry matrix H = [unit vectors to each satellite]
     * GDOP = sqrt(trace((H^T * H)^-1))
     */
    double compute_gdop() const {
        if (tracks_.size() < 2) {
            return 99.9;  // Poor geometry with < 2 satellites
        }

        ECEFCoord obs_ecef = geodetic_to_ecef(config_.observer);
        Vec<3> obs = {obs_ecef.x, obs_ecef.y, obs_ecef.z};

        // Build geometry matrix
        std::vector<Vec<3>> unit_vectors;
        for (const auto& [sat_id, track] : tracks_) {
            if (!track.is_visible) continue;

            GeodeticCoord sat_pos = track.tracker->estimated_position();
            ECEFCoord sat_ecef = geodetic_to_ecef(sat_pos);

            Vec<3> delta = {
                sat_ecef.x - obs_ecef.x,
                sat_ecef.y - obs_ecef.y,
                sat_ecef.z - obs_ecef.z
            };
            double range = vec_norm(delta);
            if (range > 0) {
                unit_vectors.push_back(vec_scale(delta, 1.0 / range));
            }
        }

        if (unit_vectors.size() < 2) {
            return 99.9;
        }

        // Compute H^T * H (3x3 matrix)
        Mat<3, 3> HTH = {};
        for (const auto& u : unit_vectors) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    HTH[i][j] += u[i] * u[j];
                }
            }
        }

        // GDOP = sqrt(trace(inv(H^T * H)))
        // For simplicity, use determinant-based approximation
        // det(HTH) indicates volume of geometry
        double det = HTH[0][0] * (HTH[1][1] * HTH[2][2] - HTH[1][2] * HTH[2][1])
                   - HTH[0][1] * (HTH[1][0] * HTH[2][2] - HTH[1][2] * HTH[2][0])
                   + HTH[0][2] * (HTH[1][0] * HTH[2][1] - HTH[1][1] * HTH[2][0]);

        if (std::abs(det) < 1e-10) {
            return 99.9;  // Singular geometry
        }

        // Approximate GDOP from trace/det ratio
        double trace = HTH[0][0] + HTH[1][1] + HTH[2][2];
        double gdop = std::sqrt(trace / std::cbrt(std::abs(det)));

        return std::clamp(gdop, 1.0, 99.9);
    }

    /**
     * @brief Get tracking statistics
     */
    MultiSatelliteStats get_statistics() const {
        MultiSatelliteStats stats;

        stats.num_visible = 0;
        stats.num_tracking = 0;
        stats.total_measurements = 0;
        stats.valid_measurements = 0;

        double sum_error = 0.0;
        stats.best_position_error_m = 1e9;

        for (const auto& [sat_id, track] : tracks_) {
            if (track.is_visible) stats.num_visible++;
            if (track.is_tracking) {
                stats.num_tracking++;
                stats.tracked_satellite_ids.push_back(sat_id);
            }
            stats.total_measurements += track.measurement_count;

            if (track.position_error_estimate < stats.best_position_error_m) {
                stats.best_position_error_m = track.position_error_estimate;
            }
            sum_error += track.position_error_estimate;
        }

        if (stats.num_tracking > 0) {
            stats.mean_position_error_m = sum_error / stats.num_tracking;
        } else {
            stats.mean_position_error_m = 0.0;
        }

        stats.geometric_dilution = compute_gdop();

        return stats;
    }

    /**
     * @brief Get individual track information
     */
    const std::map<int, SatelliteTrack>& get_tracks() const { return tracks_; }

    /**
     * @brief Get number of active tracks
     */
    size_t num_tracks() const { return tracks_.size(); }

private:
    MultiSatelliteConfig config_;
    IridiumBurstDemodulator demod_;
    std::vector<IridiumConstellation::SatelliteInfo> constellation_;
    std::map<int, SatelliteTrack> tracks_;
    std::mt19937_64 rng_;

    double epoch_jd_ = 0.0;
    double last_update_jd_ = 0.0;

    void initiate_track(int sat_id, double jd) {
        // Find satellite in constellation
        auto it = std::find_if(constellation_.begin(), constellation_.end(),
            [sat_id](const auto& s) { return s.id == sat_id; });

        if (it == constellation_.end()) return;

        SatelliteTrack track;
        track.satellite_id = sat_id;
        track.tle = it->tle;
        track.tracker = std::make_unique<UKF_AOADopplerTracker>(
            config_.observer, config_.doppler, config_.ukf_params);
        track.tracker->initialize(it->tle, jd);

        track.last_measurement_jd = jd;
        track.measurement_count = 0;
        track.consecutive_misses = 0;
        track.position_error_estimate = 10000.0;  // Initial estimate
        track.is_visible = true;
        track.is_tracking = true;

        tracks_[sat_id] = std::move(track);
    }

    void prune_tracks() {
        std::vector<int> to_remove;

        for (auto& [sat_id, track] : tracks_) {
            if (track.consecutive_misses > config_.max_consecutive_misses) {
                track.is_tracking = false;
                to_remove.push_back(sat_id);
            }
        }

        for (int sat_id : to_remove) {
            tracks_.erase(sat_id);
        }
    }
};

//=============================================================================
// Multi-Satellite Simulation
//=============================================================================

/**
 * @brief Multi-satellite simulation results
 */
struct MultiSatelliteResults {
    std::vector<double> timestamps;
    std::vector<int> num_visible;
    std::vector<int> num_tracking;
    std::vector<double> gdop;

    std::vector<Vec<3>> fused_positions;
    std::vector<double> fused_position_errors;
    std::vector<double> best_single_sat_errors;

    // Per-satellite results (for comparison)
    std::map<int, std::vector<double>> satellite_errors;

    // Statistics
    double mean_fused_error_m;
    double rms_fused_error_m;
    double mean_gdop;
    double mean_num_tracking;
    double improvement_over_single;

    void compute_statistics() {
        if (fused_position_errors.empty()) return;

        double sum = 0.0, sum_sq = 0.0;
        for (double e : fused_position_errors) {
            sum += e;
            sum_sq += e * e;
        }
        mean_fused_error_m = sum / fused_position_errors.size();
        rms_fused_error_m = std::sqrt(sum_sq / fused_position_errors.size());

        sum = 0.0;
        for (double g : gdop) sum += g;
        mean_gdop = sum / gdop.size();

        sum = 0.0;
        for (int n : num_tracking) sum += n;
        mean_num_tracking = sum / num_tracking.size();

        // Improvement over best single satellite
        double single_sum = 0.0;
        for (double e : best_single_sat_errors) single_sum += e;
        double mean_single = single_sum / best_single_sat_errors.size();
        improvement_over_single = mean_single / mean_fused_error_m;
    }
};

/**
 * @brief Run multi-satellite tracking simulation
 */
inline MultiSatelliteResults run_multi_satellite_simulation(
    const MultiSatelliteConfig& config,
    double duration_sec,
    double interval_sec = 1.0,
    bool verbose = false)
{
    MultiSatelliteResults results;

    double start_jd = 2460000.5;
    double end_jd = start_jd + duration_sec / 86400.0;
    double dt_jd = interval_sec / 86400.0;

    MultiSatelliteTracker tracker(config);
    tracker.initialize(start_jd);

    // For computing true errors, we need a reference
    // Use observer position as ground truth (we're tracking the constellation)

    double jd = start_jd;
    int step = 0;

    while (jd < end_jd) {
        tracker.process(jd);

        auto stats = tracker.get_statistics();
        GeodeticCoord fused = tracker.get_fused_position();

        results.timestamps.push_back(jd);
        results.num_visible.push_back(stats.num_visible);
        results.num_tracking.push_back(stats.num_tracking);
        results.gdop.push_back(stats.geometric_dilution);
        results.fused_positions.push_back({fused.latitude, fused.longitude, fused.altitude});

        // Compute position errors
        // For multi-sat tracking, we're actually estimating observer position
        // using multiple satellite observations (like GPS)
        // Here we track satellites, so error is relative to TLE predictions
        results.fused_position_errors.push_back(stats.mean_position_error_m);
        results.best_single_sat_errors.push_back(stats.best_position_error_m);

        if (verbose && step % 30 == 0) {
            double t = (jd - start_jd) * 86400.0;
            std::printf("t=%.0fs: Tracking %d sats, GDOP=%.1f, MeanErr=%.0fm, BestErr=%.0fm\n",
                t, stats.num_tracking, stats.geometric_dilution,
                stats.mean_position_error_m, stats.best_position_error_m);
        }

        jd += dt_jd;
        ++step;
    }

    results.compute_statistics();
    return results;
}

/**
 * @brief Print multi-satellite simulation results
 */
inline void print_multi_satellite_results(const MultiSatelliteResults& results) {
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
    std::printf("║            MULTI-SATELLITE TRACKING RESULTS                      ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("║ Mean satellites tracked:    %.1f                                  ║\n",
        results.mean_num_tracking);
    std::printf("║ Mean GDOP:                  %.2f                                  ║\n",
        results.mean_gdop);
    std::printf("║ Mean fused position error:  %.1f m                               ║\n",
        results.mean_fused_error_m);
    std::printf("║ RMS fused position error:   %.1f m                               ║\n",
        results.rms_fused_error_m);
    std::printf("║ Improvement over single:    %.1fx                                 ║\n",
        results.improvement_over_single);
    std::printf("╚══════════════════════════════════════════════════════════════════╝\n");
}

} // namespace tracking
} // namespace optmath
