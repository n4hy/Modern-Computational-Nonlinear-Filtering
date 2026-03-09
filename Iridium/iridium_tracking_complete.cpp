/**
 * @file iridium_tracking_complete.cpp
 * @brief Complete Iridium Tracking System Demonstration
 *
 * Demonstrates the complete tracking system with all recommendations:
 * 1. AOA+Doppler with FINE accuracy (~10 Hz from preamble correlation)
 * 2. Realistic burst demodulator model
 * 3. Multi-satellite tracking for improved geometry
 * 4. Comparison of all approaches
 */

#include <optmath/ukf_aoa_tracking.hpp>
#include <optmath/ukf_aoa_doppler_tracking.hpp>
#include <optmath/iridium_burst_demodulator.hpp>
#include <optmath/multi_satellite_tracker.hpp>
#include <iostream>
#include <iomanip>

using namespace optmath::tracking;
using namespace optmath::tracking::constants;

void print_header() {
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    std::printf("║                                                                          ║\n");
    std::printf("║     IRIDIUM-NEXT SATELLITE TRACKING SYSTEM                               ║\n");
    std::printf("║     Two-Antenna Coherent Receiver with UKF                               ║\n");
    std::printf("║                                                                          ║\n");
    std::printf("║     Features:                                                            ║\n");
    std::printf("║     • AOA from phase-difference (two-antenna array)                      ║\n");
    std::printf("║     • Doppler from burst preamble correlation (~10 Hz accuracy)          ║\n");
    std::printf("║     • Unscented Kalman Filter for nonlinear estimation                   ║\n");
    std::printf("║     • Multi-satellite tracking for improved geometry                     ║\n");
    std::printf("║                                                                          ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");
}

/**
 * @brief Run single satellite comparison
 */
void run_single_satellite_comparison(double duration_sec, bool verbose) {
    std::printf("═══════════════════════════════════════════════════════════════════════════\n");
    std::printf("                    SINGLE SATELLITE TRACKING COMPARISON\n");
    std::printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    // Find visible satellite configuration
    GeodeticCoord observer;
    observer.latitude = 40.015 * DEG2RAD;
    observer.longitude = -105.27 * DEG2RAD;
    observer.altitude = 1655.0;

    double start_jd = 2460000.5;

    // Search for visible satellite
    double best_el = -90.0;
    double best_raan = 0.0, best_ma = 0.0;

    for (int raan = 0; raan < 360; raan += 30) {
        for (int ma = 0; ma < 360; ma += 30) {
            TLE tle = create_iridium_tle(start_jd, raan, ma);
            SimplifiedSGP4 sgp4(tle);
            AzElCoord azel = sgp4.get_azel(start_jd, observer);

            if (azel.elevation > best_el) {
                best_el = azel.elevation;
                best_raan = raan;
                best_ma = ma;
            }
        }
    }

    std::printf("Selected satellite: RAAN=%.0f°, MA=%.0f° (initial elevation=%.1f°)\n\n",
        best_raan, best_ma, best_el * RAD2DEG);

    TLE tle = create_iridium_tle(start_jd, best_raan, best_ma);
    AntennaArrayConfig antenna = AntennaArrayConfig::default_iridium();

    // =========================================================================
    // Method 1: AOA-only
    // =========================================================================
    std::printf("Running AOA-only tracking...\n");

    SimulationConfig cfg_aoa;
    cfg_aoa.observer = observer;
    cfg_aoa.satellite_tle = tle;
    cfg_aoa.antenna = antenna;
    cfg_aoa.start_jd = start_jd;
    cfg_aoa.duration_sec = duration_sec;
    cfg_aoa.measurement_interval_sec = 1.0;
    cfg_aoa.use_burst_timing = false;  // Continuous for fair comparison
    cfg_aoa.verbose = false;

    SimulationResults results_aoa = run_simulation(cfg_aoa);
    results_aoa.compute_statistics();

    // =========================================================================
    // Method 2: AOA + Doppler (FINE - preamble correlation)
    // =========================================================================
    std::printf("Running AOA+Doppler (FINE) tracking...\n");

    SimulationConfigDoppler cfg_doppler;
    cfg_doppler.observer = observer;
    cfg_doppler.satellite_tle = tle;
    cfg_doppler.antenna = antenna;
    cfg_doppler.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
    cfg_doppler.start_jd = start_jd;
    cfg_doppler.duration_sec = duration_sec;
    cfg_doppler.measurement_interval_sec = 1.0;
    cfg_doppler.use_burst_timing = false;
    cfg_doppler.verbose = false;

    SimulationResultsDoppler results_doppler = run_simulation_doppler(cfg_doppler);
    results_doppler.compute_statistics();

    // =========================================================================
    // Results comparison
    // =========================================================================
    std::printf("\n");
    std::printf("┌─────────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                    SINGLE SATELLITE RESULTS                            │\n");
    std::printf("├───────────────────────────────┬───────────────────┬────────────────────┤\n");
    std::printf("│            Metric             │     AOA-only      │    AOA+Doppler     │\n");
    std::printf("├───────────────────────────────┼───────────────────┼────────────────────┤\n");
    std::printf("│ Mean Position Error           │ %12.1f m    │ %12.1f m     │\n",
        results_aoa.mean_position_error_m, results_doppler.mean_position_error_m);
    std::printf("│ RMS Position Error            │ %12.1f m    │ %12.1f m     │\n",
        results_aoa.rms_position_error_m, results_doppler.rms_position_error_m);
    std::printf("│ Final Position Error          │ %12.1f m    │ %12.1f m     │\n",
        results_aoa.position_error_m.back(), results_doppler.position_error_m.back());
    std::printf("├───────────────────────────────┼───────────────────┼────────────────────┤\n");
    std::printf("│ Azimuth Error (mean)          │ %12.4f deg  │ %12.4f deg   │\n",
        results_aoa.mean_az_error_deg, results_doppler.mean_az_error_deg);
    std::printf("│ Elevation Error (mean)        │ %12.4f deg  │ %12.4f deg   │\n",
        results_aoa.mean_el_error_deg, results_doppler.mean_el_error_deg);
    std::printf("├───────────────────────────────┼───────────────────┼────────────────────┤\n");
    std::printf("│ Doppler Error (mean)          │        N/A        │ %12.2f Hz    │\n",
        results_doppler.mean_doppler_error_hz);
    std::printf("└───────────────────────────────┴───────────────────┴────────────────────┘\n");

    double improvement = results_aoa.mean_position_error_m / results_doppler.mean_position_error_m;
    std::printf("\n→ Doppler provides %.1fx improvement in mean position error\n", improvement);
}

/**
 * @brief Run multi-satellite demonstration
 */
void run_multi_satellite_demo(double duration_sec, bool verbose) {
    std::printf("\n");
    std::printf("═══════════════════════════════════════════════════════════════════════════\n");
    std::printf("                    MULTI-SATELLITE TRACKING DEMONSTRATION\n");
    std::printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    MultiSatelliteConfig config = MultiSatelliteConfig::default_config();
    config.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
    config.max_tracked_satellites = 4;

    std::printf("Configuration:\n");
    std::printf("  Observer: %.3f°N, %.3f°E\n",
        config.observer.latitude * RAD2DEG,
        config.observer.longitude * RAD2DEG);
    std::printf("  Max simultaneous tracks: %d\n", config.max_tracked_satellites);
    std::printf("  Doppler accuracy: FINE (~10 Hz from preamble)\n");
    std::printf("  Duration: %.0f seconds\n\n", duration_sec);

    std::printf("Running simulation...\n\n");

    MultiSatelliteResults results = run_multi_satellite_simulation(
        config, duration_sec, 1.0, verbose);

    print_multi_satellite_results(results);
}

/**
 * @brief Compare all approaches
 */
void run_full_comparison(double duration_sec) {
    std::printf("\n");
    std::printf("═══════════════════════════════════════════════════════════════════════════\n");
    std::printf("                    FULL SYSTEM COMPARISON                                \n");
    std::printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    GeodeticCoord observer;
    observer.latitude = 40.015 * DEG2RAD;
    observer.longitude = -105.27 * DEG2RAD;
    observer.altitude = 1655.0;

    double start_jd = 2460000.5;

    // Find visible satellite
    double best_raan = 30.0, best_ma = 30.0;  // Known visible config
    TLE tle = create_iridium_tle(start_jd, best_raan, best_ma);
    AntennaArrayConfig antenna = AntennaArrayConfig::default_iridium();

    std::printf("Running all tracking methods...\n\n");

    // AOA-only
    SimulationConfig cfg_aoa;
    cfg_aoa.observer = observer;
    cfg_aoa.satellite_tle = tle;
    cfg_aoa.antenna = antenna;
    cfg_aoa.start_jd = start_jd;
    cfg_aoa.duration_sec = duration_sec;
    cfg_aoa.measurement_interval_sec = 1.0;
    cfg_aoa.use_burst_timing = false;
    cfg_aoa.verbose = false;
    SimulationResults r_aoa = run_simulation(cfg_aoa);
    r_aoa.compute_statistics();

    // AOA + Doppler COARSE
    SimulationConfigDoppler cfg_coarse;
    cfg_coarse.observer = observer;
    cfg_coarse.satellite_tle = tle;
    cfg_coarse.antenna = antenna;
    cfg_coarse.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::COARSE);
    cfg_coarse.start_jd = start_jd;
    cfg_coarse.duration_sec = duration_sec;
    cfg_coarse.measurement_interval_sec = 1.0;
    cfg_coarse.use_burst_timing = false;
    cfg_coarse.verbose = false;
    SimulationResultsDoppler r_coarse = run_simulation_doppler(cfg_coarse);
    r_coarse.compute_statistics();

    // AOA + Doppler FINE (recommended)
    SimulationConfigDoppler cfg_fine;
    cfg_fine.observer = observer;
    cfg_fine.satellite_tle = tle;
    cfg_fine.antenna = antenna;
    cfg_fine.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
    cfg_fine.start_jd = start_jd;
    cfg_fine.duration_sec = duration_sec;
    cfg_fine.measurement_interval_sec = 1.0;
    cfg_fine.use_burst_timing = false;
    cfg_fine.verbose = false;
    SimulationResultsDoppler r_fine = run_simulation_doppler(cfg_fine);
    r_fine.compute_statistics();

    // AOA + Doppler PRECISE
    SimulationConfigDoppler cfg_precise;
    cfg_precise.observer = observer;
    cfg_precise.satellite_tle = tle;
    cfg_precise.antenna = antenna;
    cfg_precise.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::PRECISE);
    cfg_precise.start_jd = start_jd;
    cfg_precise.duration_sec = duration_sec;
    cfg_precise.measurement_interval_sec = 1.0;
    cfg_precise.use_burst_timing = false;
    cfg_precise.verbose = false;
    SimulationResultsDoppler r_precise = run_simulation_doppler(cfg_precise);
    r_precise.compute_statistics();

    // Multi-satellite
    MultiSatelliteConfig cfg_multi = MultiSatelliteConfig::default_config();
    cfg_multi.doppler = DopplerConfig::default_iridium(DopplerConfig::AccuracyMode::FINE);
    MultiSatelliteResults r_multi = run_multi_satellite_simulation(cfg_multi, duration_sec, 1.0, false);

    // Print comparison
    std::printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                         FULL COMPARISON TABLE                              │\n");
    std::printf("├──────────────────────────┬──────────────┬──────────────┬───────────────────┤\n");
    std::printf("│         Method           │  Mean Error  │ Final Error  │   Improvement     │\n");
    std::printf("├──────────────────────────┼──────────────┼──────────────┼───────────────────┤\n");
    std::printf("│ AOA-only (baseline)      │ %9.0f m  │ %9.0f m  │       1.0x        │\n",
        r_aoa.mean_position_error_m, r_aoa.position_error_m.back());
    std::printf("│ AOA+Doppler COARSE       │ %9.0f m  │ %9.0f m  │       %.1fx        │\n",
        r_coarse.mean_position_error_m, r_coarse.position_error_m.back(),
        r_aoa.mean_position_error_m / r_coarse.mean_position_error_m);
    std::printf("│ AOA+Doppler FINE ★       │ %9.0f m  │ %9.0f m  │       %.1fx        │\n",
        r_fine.mean_position_error_m, r_fine.position_error_m.back(),
        r_aoa.mean_position_error_m / r_fine.mean_position_error_m);
    std::printf("│ AOA+Doppler PRECISE      │ %9.0f m  │ %9.0f m  │       %.1fx        │\n",
        r_precise.mean_position_error_m, r_precise.position_error_m.back(),
        r_aoa.mean_position_error_m / r_precise.mean_position_error_m);
    std::printf("│ Multi-Satellite (4) ★    │ %9.0f m  │ %9.0f m  │       %.1fx        │\n",
        r_multi.mean_fused_error_m, r_multi.fused_position_errors.back(),
        r_aoa.mean_position_error_m / r_multi.mean_fused_error_m);
    std::printf("└──────────────────────────┴──────────────┴──────────────┴───────────────────┘\n");
    std::printf("\n★ = Recommended configuration\n");

    std::printf("\n");
    std::printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    std::printf("│                            RECOMMENDATIONS                                 │\n");
    std::printf("├─────────────────────────────────────────────────────────────────────────────┤\n");
    std::printf("│ 1. USE AOA+DOPPLER WITH FINE ACCURACY                                      │\n");
    std::printf("│    • Extract Doppler from burst preamble correlation (~10 Hz)             │\n");
    std::printf("│    • Provides 2-3x improvement over AOA-only                              │\n");
    std::printf("├─────────────────────────────────────────────────────────────────────────────┤\n");
    std::printf("│ 2. PREAMBLE CORRELATION IS OPTIMAL                                         │\n");
    std::printf("│    • 64-symbol preamble provides sufficient integration time              │\n");
    std::printf("│    • ~10 Hz accuracy achievable with modest SNR (>10 dB)                  │\n");
    std::printf("│    • Going to 1 Hz (full burst coherent) provides minimal extra benefit   │\n");
    std::printf("├─────────────────────────────────────────────────────────────────────────────┤\n");
    std::printf("│ 3. MULTI-SATELLITE TRACKING RECOMMENDED                                    │\n");
    std::printf("│    • 2-4 satellites visible from most locations                           │\n");
    std::printf("│    • Improved geometry (lower GDOP) reduces position uncertainty          │\n");
    std::printf("│    • Information fusion provides robustness to individual track loss      │\n");
    std::printf("└─────────────────────────────────────────────────────────────────────────────┘\n");
}

void print_usage(const char* prog) {
    std::cout << "Iridium Tracking System Demonstration\n\n"
              << "Usage: " << prog << " [mode] [options]\n\n"
              << "Modes:\n"
              << "  single       Run single satellite comparison\n"
              << "  multi        Run multi-satellite demonstration\n"
              << "  full         Run full system comparison (default)\n"
              << "\nOptions:\n"
              << "  --duration <sec>   Simulation duration (default: 300)\n"
              << "  --verbose          Enable verbose output\n"
              << "  --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string mode = "full";
    double duration = 300.0;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::atof(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "single" || arg == "multi" || arg == "full") {
            mode = arg;
        }
    }

    print_header();

    if (mode == "single") {
        run_single_satellite_comparison(duration, verbose);
    } else if (mode == "multi") {
        run_multi_satellite_demo(duration, verbose);
    } else {
        run_single_satellite_comparison(duration, verbose);
        run_multi_satellite_demo(duration, verbose);
        run_full_comparison(duration);
    }

    return 0;
}
