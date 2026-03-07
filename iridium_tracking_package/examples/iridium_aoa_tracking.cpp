/**
 * @file iridium_aoa_tracking.cpp
 * @brief Iridium-Next AOA Tracking Simulation using UKF
 *
 * Demonstrates tracking an Iridium-Next satellite using a two-antenna
 * coherent receiver array with Unscented Kalman Filter for nonlinear
 * estimation of position from noisy angle-of-arrival measurements.
 *
 * Usage: ./iridium_aoa_tracking [options]
 *   --duration <sec>     Simulation duration (default: 600)
 *   --baseline <m>       Antenna baseline (default: 0.1)
 *   --noise <rad>        Phase noise std (default: 0.1)
 *   --lat <deg>          Observer latitude (default: 40.015)
 *   --lon <deg>          Observer longitude (default: -105.27)
 *   --alt <m>            Observer altitude (default: 1655)
 *   --quiet              Suppress verbose output
 *   --help               Show this help
 */

#include <optmath/ukf_aoa_tracking.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>

using namespace optmath::tracking;
using namespace optmath::tracking::constants;

void print_usage(const char* prog) {
    std::cout << "Iridium-Next AOA Tracking Simulation using UKF\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --duration <sec>     Simulation duration (default: 600)\n"
              << "  --interval <sec>     Measurement interval (default: 1.0)\n"
              << "  --baseline <m>       Antenna baseline (default: 0.1)\n"
              << "  --noise <rad>        Phase noise std (default: 0.1)\n"
              << "  --lat <deg>          Observer latitude (default: 40.015)\n"
              << "  --lon <deg>          Observer longitude (default: -105.27)\n"
              << "  --alt <m>            Observer altitude (default: 1655)\n"
              << "  --raan <deg>         Satellite RAAN (default: 45)\n"
              << "  --ma <deg>           Initial mean anomaly (default: 0)\n"
              << "  --no-burst           Disable burst timing simulation\n"
              << "  --output <file>      Write results to CSV file\n"
              << "  --quiet              Suppress verbose output\n"
              << "  --help               Show this help\n"
              << "\nExample:\n"
              << "  " << prog << " --duration 300 --baseline 0.15 --output results.csv\n";
}

int main(int argc, char* argv[]) {
    SimulationConfig cfg = SimulationConfig::default_config();
    std::string output_file;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            cfg.duration_sec = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            cfg.measurement_interval_sec = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) {
            cfg.antenna.baseline = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--noise") == 0 && i + 1 < argc) {
            cfg.antenna.phase_noise_std = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--lat") == 0 && i + 1 < argc) {
            cfg.observer.latitude = std::atof(argv[++i]) * DEG2RAD;
        } else if (std::strcmp(argv[i], "--lon") == 0 && i + 1 < argc) {
            cfg.observer.longitude = std::atof(argv[++i]) * DEG2RAD;
        } else if (std::strcmp(argv[i], "--alt") == 0 && i + 1 < argc) {
            cfg.observer.altitude = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--raan") == 0 && i + 1 < argc) {
            double raan = std::atof(argv[++i]);
            cfg.satellite_tle = create_iridium_tle(cfg.start_jd, raan, 0.0);
        } else if (std::strcmp(argv[i], "--ma") == 0 && i + 1 < argc) {
            double ma = std::atof(argv[++i]);
            cfg.satellite_tle.mean_anomaly = ma * DEG2RAD;
        } else if (std::strcmp(argv[i], "--no-burst") == 0) {
            cfg.use_burst_timing = false;
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            cfg.verbose = false;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Print configuration
    std::cout << "========================================\n"
              << "Iridium-Next AOA Tracking Simulation\n"
              << "========================================\n\n"
              << "Configuration:\n"
              << "  Observer: " << std::fixed << std::setprecision(4)
              << cfg.observer.latitude * RAD2DEG << "° N, "
              << cfg.observer.longitude * RAD2DEG << "° E, "
              << cfg.observer.altitude << " m\n"
              << "  Antenna baseline: " << cfg.antenna.baseline * 100.0 << " cm\n"
              << "  Phase noise: " << cfg.antenna.phase_noise_std * RAD2DEG << "° RMS\n"
              << "  Duration: " << cfg.duration_sec << " s\n"
              << "  Measurement interval: " << cfg.measurement_interval_sec << " s\n"
              << "  Burst timing: " << (cfg.use_burst_timing ? "enabled" : "disabled") << "\n\n"
              << "Satellite:\n"
              << "  Type: Iridium-Next (LEO)\n"
              << "  Altitude: " << IRIDIUM_ALTITUDE / 1000.0 << " km\n"
              << "  Inclination: " << IRIDIUM_INCLINATION * RAD2DEG << "°\n"
              << "  Period: " << IRIDIUM_PERIOD / 60.0 << " min\n"
              << "  Frequency: " << IRIDIUM_FREQUENCY / 1e6 << " MHz\n"
              << "  Wavelength: " << IRIDIUM_WAVELENGTH * 100.0 << " cm\n\n";

    // Run simulation
    std::cout << "Running simulation...\n\n";
    SimulationResults results = run_simulation(cfg);

    // Print summary
    print_results(results);

    // Write CSV output if requested
    if (!output_file.empty()) {
        std::ofstream ofs(output_file);
        if (!ofs) {
            std::cerr << "Error: Could not open output file: " << output_file << "\n";
            return 1;
        }

        ofs << "time_sec,true_lat_deg,true_lon_deg,true_alt_m,"
            << "true_az_deg,true_el_deg,"
            << "meas_az_deg,meas_el_deg,meas_valid,"
            << "est_lat_deg,est_lon_deg,est_alt_m,"
            << "est_az_deg,est_el_deg,"
            << "pos_err_m,az_err_deg,el_err_deg,"
            << "unc_lat_m,unc_lon_m,unc_alt_m\n";

        for (size_t i = 0; i < results.timestamps.size(); ++i) {
            double t_sec = (results.timestamps[i] - cfg.start_jd) * 86400.0;

            ofs << std::fixed << std::setprecision(6)
                << t_sec << ","
                << results.true_positions[i][0] * RAD2DEG << ","
                << results.true_positions[i][1] * RAD2DEG << ","
                << results.true_positions[i][2] << ","
                << results.true_azel[i][0] * RAD2DEG << ","
                << results.true_azel[i][1] * RAD2DEG << ","
                << results.measured_azel[i][0] * RAD2DEG << ","
                << results.measured_azel[i][1] * RAD2DEG << ","
                << (results.measurement_valid[i] ? 1 : 0) << ","
                << results.estimated_positions[i][0] * RAD2DEG << ","
                << results.estimated_positions[i][1] * RAD2DEG << ","
                << results.estimated_positions[i][2] << ","
                << results.estimated_azel[i][0] * RAD2DEG << ","
                << results.estimated_azel[i][1] * RAD2DEG << ","
                << results.position_error_m[i] << ","
                << results.azimuth_error_rad[i] * RAD2DEG << ","
                << results.elevation_error_rad[i] * RAD2DEG << ","
                << results.position_uncertainty[i][0] << ","
                << results.position_uncertainty[i][1] << ","
                << results.position_uncertainty[i][2] << "\n";
        }

        std::cout << "\nResults written to: " << output_file << "\n";
    }

    return 0;
}
