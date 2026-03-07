/**
 * @file compare_aoa_doppler.cpp
 * @brief Compare AOA-only vs AOA+Doppler tracking performance
 *
 * Runs both tracking approaches on identical scenarios and compares:
 * - Position accuracy
 * - Velocity accuracy
 * - Convergence rate
 * - Robustness to measurement gaps
 */

#include <optmath/ukf_aoa_tracking.hpp>
#include <optmath/ukf_aoa_doppler_tracking.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace optmath::tracking;
using namespace optmath::tracking::constants;

/**
 * @brief Run both simulations and compare results
 */
void run_comparison(double raan_deg, double ma_deg,
                    double duration_sec, bool use_burst,
                    DopplerConfig::AccuracyMode doppler_mode) {

    // Common configuration
    GeodeticCoord observer;
    observer.latitude = 40.015 * DEG2RAD;
    observer.longitude = -105.27 * DEG2RAD;
    observer.altitude = 1655.0;

    double start_jd = 2460000.5;
    TLE tle = create_iridium_tle(start_jd, raan_deg, ma_deg);

    AntennaArrayConfig antenna = AntennaArrayConfig::default_iridium();
    DopplerConfig doppler = DopplerConfig::default_iridium(doppler_mode);

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
    std::printf("║        AOA-ONLY vs AOA+DOPPLER TRACKING COMPARISON               ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("║ Observer: %.3f°N, %.3f°E, %.0fm                           ║\n",
        observer.latitude * RAD2DEG, observer.longitude * RAD2DEG, observer.altitude);
    std::printf("║ Satellite: Iridium-Next (RAAN=%.0f°, MA=%.0f°)                       ║\n",
        raan_deg, ma_deg);
    std::printf("║ Duration: %.0f seconds                                             ║\n",
        duration_sec);
    std::printf("║ Burst timing: %s                                              ║\n",
        use_burst ? "enabled " : "disabled");
    std::printf("║ Doppler accuracy: %s                                          ║\n",
        doppler_mode == DopplerConfig::AccuracyMode::COARSE ? "COARSE (~100 Hz)" :
        doppler_mode == DopplerConfig::AccuracyMode::FINE ? "FINE (~10 Hz)   " :
        "PRECISE (~1 Hz) ");
    std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // =========================================================================
    // Run AOA-only simulation
    // =========================================================================
    std::printf("Running AOA-only simulation...\n");

    SimulationConfig cfg_aoa = SimulationConfig::default_config();
    cfg_aoa.observer = observer;
    cfg_aoa.satellite_tle = tle;
    cfg_aoa.antenna = antenna;
    cfg_aoa.start_jd = start_jd;
    cfg_aoa.duration_sec = duration_sec;
    cfg_aoa.measurement_interval_sec = 1.0;
    cfg_aoa.use_burst_timing = use_burst;
    cfg_aoa.verbose = false;

    SimulationResults results_aoa = run_simulation(cfg_aoa);
    results_aoa.compute_statistics();

    // =========================================================================
    // Run AOA+Doppler simulation
    // =========================================================================
    std::printf("Running AOA+Doppler simulation...\n");

    SimulationConfigDoppler cfg_doppler = SimulationConfigDoppler::default_config();
    cfg_doppler.observer = observer;
    cfg_doppler.satellite_tle = tle;
    cfg_doppler.antenna = antenna;
    cfg_doppler.doppler = doppler;
    cfg_doppler.start_jd = start_jd;
    cfg_doppler.duration_sec = duration_sec;
    cfg_doppler.measurement_interval_sec = 1.0;
    cfg_doppler.use_burst_timing = use_burst;
    cfg_doppler.verbose = false;

    SimulationResultsDoppler results_doppler = run_simulation_doppler(cfg_doppler);
    results_doppler.compute_statistics();

    // =========================================================================
    // Print comparison
    // =========================================================================

    std::printf("\n");
    std::printf("┌────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                    COMPARISON RESULTS                             │\n");
    std::printf("├──────────────────────────┬──────────────────┬──────────────────────┤\n");
    std::printf("│        Metric            │    AOA-only      │    AOA+Doppler       │\n");
    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");

    std::printf("│ Valid measurements       │ %6d (%5.1f%%)  │ %6d (%5.1f%%)       │\n",
        results_aoa.num_valid_measurements,
        100.0 * results_aoa.num_valid_measurements / results_aoa.num_measurements,
        results_doppler.num_valid_measurements,
        100.0 * results_doppler.num_valid_measurements / results_doppler.num_measurements);

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");
    std::printf("│ Position Error (mean)    │ %10.1f m     │ %10.1f m          │\n",
        results_aoa.mean_position_error_m,
        results_doppler.mean_position_error_m);

    std::printf("│ Position Error (RMS)     │ %10.1f m     │ %10.1f m          │\n",
        results_aoa.rms_position_error_m,
        results_doppler.rms_position_error_m);

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");
    std::printf("│ Azimuth Error (mean)     │ %10.4f deg   │ %10.4f deg        │\n",
        results_aoa.mean_az_error_deg,
        results_doppler.mean_az_error_deg);

    std::printf("│ Elevation Error (mean)   │ %10.4f deg   │ %10.4f deg        │\n",
        results_aoa.mean_el_error_deg,
        results_doppler.mean_el_error_deg);

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");
    std::printf("│ Doppler Error (mean)     │       N/A        │ %10.2f Hz         │\n",
        results_doppler.mean_doppler_error_hz);

    std::printf("│ Doppler Error (RMS)      │       N/A        │ %10.2f Hz         │\n",
        results_doppler.rms_doppler_error_hz);

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");

    // Final position errors
    double final_err_aoa = results_aoa.position_error_m.back();
    double final_err_doppler = results_doppler.position_error_m.back();

    std::printf("│ Final Position Error     │ %10.1f m     │ %10.1f m          │\n",
        final_err_aoa, final_err_doppler);

    // Position uncertainty
    auto unc_aoa = results_aoa.position_uncertainty.back();
    auto unc_doppler = results_doppler.position_uncertainty.back();

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");
    std::printf("│ Final Lat Uncertainty    │ %10.1f m     │ %10.1f m          │\n",
        unc_aoa[0], unc_doppler[0]);
    std::printf("│ Final Lon Uncertainty    │ %10.1f m     │ %10.1f m          │\n",
        unc_aoa[1], unc_doppler[1]);
    std::printf("│ Final Alt Uncertainty    │ %10.1f m     │ %10.1f m          │\n",
        unc_aoa[2], unc_doppler[2]);

    std::printf("├──────────────────────────┼──────────────────┼──────────────────────┤\n");

    // Velocity uncertainty (only for Doppler)
    auto vel_unc = results_doppler.velocity_uncertainty.back();
    std::printf("│ Final V_lat Uncertainty  │       N/A        │ %10.3f m/s        │\n",
        vel_unc[0]);
    std::printf("│ Final V_lon Uncertainty  │       N/A        │ %10.3f m/s        │\n",
        vel_unc[1]);
    std::printf("│ Final V_alt Uncertainty  │       N/A        │ %10.3f m/s        │\n",
        vel_unc[2]);

    std::printf("└──────────────────────────┴──────────────────┴──────────────────────┘\n");

    // =========================================================================
    // Compute improvement factors
    // =========================================================================

    std::printf("\n");
    std::printf("┌────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                    IMPROVEMENT FACTORS                            │\n");
    std::printf("├────────────────────────────────────────────────────────────────────┤\n");

    double pos_improvement = results_aoa.mean_position_error_m / results_doppler.mean_position_error_m;
    double rms_improvement = results_aoa.rms_position_error_m / results_doppler.rms_position_error_m;
    double final_improvement = final_err_aoa / final_err_doppler;

    std::printf("│ Mean Position Error Improvement:  %6.1fx                          │\n",
        pos_improvement);
    std::printf("│ RMS Position Error Improvement:   %6.1fx                          │\n",
        rms_improvement);
    std::printf("│ Final Position Error Improvement: %6.1fx                          │\n",
        final_improvement);

    double lat_unc_improvement = unc_aoa[0] / unc_doppler[0];
    double lon_unc_improvement = unc_aoa[1] / unc_doppler[1];
    double alt_unc_improvement = unc_aoa[2] / unc_doppler[2];

    std::printf("│ Latitude Uncertainty Improvement: %6.1fx                          │\n",
        lat_unc_improvement);
    std::printf("│ Longitude Uncertainty Improvement:%6.1fx                          │\n",
        lon_unc_improvement);
    std::printf("│ Altitude Uncertainty Improvement: %6.1fx                          │\n",
        alt_unc_improvement);

    std::printf("└────────────────────────────────────────────────────────────────────┘\n");

    // =========================================================================
    // Error evolution comparison
    // =========================================================================

    std::printf("\n");
    std::printf("┌────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                    ERROR EVOLUTION                                │\n");
    std::printf("├────────────┬─────────────────────┬─────────────────────────────────┤\n");
    std::printf("│  Time (s)  │  AOA-only Error (m) │  AOA+Doppler Error (m)          │\n");
    std::printf("├────────────┼─────────────────────┼─────────────────────────────────┤\n");

    size_t n_samples = results_aoa.position_error_m.size();
    size_t step = std::max(size_t(1), n_samples / 10);

    for (size_t i = 0; i < n_samples; i += step) {
        double t = (results_aoa.timestamps[i] - start_jd) * 86400.0;
        std::printf("│ %10.1f │ %19.1f │ %31.1f │\n",
            t,
            results_aoa.position_error_m[i],
            results_doppler.position_error_m[i]);
    }

    // Always show final
    size_t last = n_samples - 1;
    double t_final = (results_aoa.timestamps[last] - start_jd) * 86400.0;
    std::printf("│ %10.1f │ %19.1f │ %31.1f │\n",
        t_final,
        results_aoa.position_error_m[last],
        results_doppler.position_error_m[last]);

    std::printf("└────────────┴─────────────────────┴─────────────────────────────────┘\n");
}

void print_usage(const char* prog) {
    std::cout << "AOA-only vs AOA+Doppler Tracking Comparison\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --duration <sec>     Simulation duration (default: 300)\n"
              << "  --raan <deg>         Satellite RAAN (default: 30)\n"
              << "  --ma <deg>           Initial mean anomaly (default: 30)\n"
              << "  --no-burst           Disable burst timing\n"
              << "  --doppler-coarse     Use coarse Doppler (~100 Hz accuracy)\n"
              << "  --doppler-fine       Use fine Doppler (~10 Hz accuracy) [default]\n"
              << "  --doppler-precise    Use precise Doppler (~1 Hz accuracy)\n"
              << "  --all-doppler        Compare all Doppler accuracy modes\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    double duration_sec = 300.0;
    double raan_deg = 30.0;
    double ma_deg = 30.0;
    bool use_burst = true;
    bool compare_all = false;
    DopplerConfig::AccuracyMode doppler_mode = DopplerConfig::AccuracyMode::FINE;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_sec = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--raan") == 0 && i + 1 < argc) {
            raan_deg = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--ma") == 0 && i + 1 < argc) {
            ma_deg = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-burst") == 0) {
            use_burst = false;
        } else if (std::strcmp(argv[i], "--doppler-coarse") == 0) {
            doppler_mode = DopplerConfig::AccuracyMode::COARSE;
        } else if (std::strcmp(argv[i], "--doppler-fine") == 0) {
            doppler_mode = DopplerConfig::AccuracyMode::FINE;
        } else if (std::strcmp(argv[i], "--doppler-precise") == 0) {
            doppler_mode = DopplerConfig::AccuracyMode::PRECISE;
        } else if (std::strcmp(argv[i], "--all-doppler") == 0) {
            compare_all = true;
        }
    }

    if (compare_all) {
        // Run comparison for all Doppler accuracy modes
        std::printf("\n═══════════════════════════════════════════════════════════════════════\n");
        std::printf("           COMPARING ALL DOPPLER ACCURACY MODES\n");
        std::printf("═══════════════════════════════════════════════════════════════════════\n");

        run_comparison(raan_deg, ma_deg, duration_sec, use_burst,
                      DopplerConfig::AccuracyMode::COARSE);

        run_comparison(raan_deg, ma_deg, duration_sec, use_burst,
                      DopplerConfig::AccuracyMode::FINE);

        run_comparison(raan_deg, ma_deg, duration_sec, use_burst,
                      DopplerConfig::AccuracyMode::PRECISE);
    } else {
        run_comparison(raan_deg, ma_deg, duration_sec, use_burst, doppler_mode);
    }

    return 0;
}
