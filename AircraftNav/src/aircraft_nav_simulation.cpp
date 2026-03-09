/**
 * @file aircraft_nav_simulation.cpp
 * @brief Main executable for aircraft navigation simulation
 *
 * Runs a single simulation scenario:
 * 1. GPS/INS Integration (0-60s)
 * 2. GPS Outage (60-90s)
 * 3. Iridium Recovery (90-300s)
 *
 * Outputs trajectory CSV and performance summary.
 */

#include <iostream>
#include <string>
#include <cstdlib>

#include "AircraftNavSimulation.h"

using namespace AircraftNav;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  --duration <seconds>   Simulation duration (default: 300)\n"
              << "  --outage-start <sec>   GPS outage start time (default: 60)\n"
              << "  --outage-duration <s>  GPS outage duration (default: 30)\n"
              << "  --turbulence <level>   Turbulence: light/moderate/severe (default: moderate)\n"
              << "  --no-gps               Disable GPS entirely\n"
              << "  --no-iridium           Disable Iridium\n"
              << "  --output <file>        Output CSV filename (default: aircraft_nav_results.csv)\n"
              << "  --seed <value>         Random seed (default: random)\n"
              << "  --quiet                Suppress verbose output\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char* argv[]) {
    SimulationConfig config = SimulationConfig::default_config();
    std::string output_file = "aircraft_nav_results.csv";
    uint64_t seed = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration_s = std::atof(argv[++i]);
        } else if (arg == "--outage-start" && i + 1 < argc) {
            config.gps_outage_start_s = std::atof(argv[++i]);
        } else if (arg == "--outage-duration" && i + 1 < argc) {
            config.gps_outage_duration_s = std::atof(argv[++i]);
        } else if (arg == "--turbulence" && i + 1 < argc) {
            std::string level = argv[++i];
            if (level == "light") {
                config.aircraft.turbulence_severity = DrydenConfig::Severity::LIGHT;
            } else if (level == "moderate") {
                config.aircraft.turbulence_severity = DrydenConfig::Severity::MODERATE;
            } else if (level == "severe") {
                config.aircraft.turbulence_severity = DrydenConfig::Severity::SEVERE;
            }
        } else if (arg == "--no-gps") {
            config.enable_gps = false;
        } else if (arg == "--no-iridium") {
            config.enable_iridium = false;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--quiet") {
            config.verbose = false;
        }
    }

    // Print configuration
    std::cout << "Aircraft Navigation Simulation\n";
    std::cout << "==============================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Duration:         " << config.duration_s << " s\n";
    std::cout << "  GPS outage:       " << config.gps_outage_start_s << " - "
              << (config.gps_outage_start_s + config.gps_outage_duration_s) << " s\n";
    std::cout << "  GPS enabled:      " << (config.enable_gps ? "yes" : "no") << "\n";
    std::cout << "  Iridium enabled:  " << (config.enable_iridium ? "yes" : "no") << "\n";
    std::cout << "  Turbulence:       ";
    switch (config.aircraft.turbulence_severity) {
        case DrydenConfig::Severity::LIGHT: std::cout << "light\n"; break;
        case DrydenConfig::Severity::MODERATE: std::cout << "moderate\n"; break;
        case DrydenConfig::Severity::SEVERE: std::cout << "severe\n"; break;
    }
    std::cout << "  Output file:      " << output_file << "\n";
    std::cout << "  Seed:             " << (seed == 0 ? "random" : std::to_string(seed)) << "\n";
    std::cout << "\n";

    // Create and run simulation
    AircraftNavSimulation sim(config, seed);

    std::cout << "Running simulation...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    SimulationTrajectory traj = sim.run();

    auto end_time = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Simulation completed in " << std::fixed << std::setprecision(2)
              << runtime << " seconds\n\n";

    // Compute and print summary statistics
    std::cout << "Results Summary\n";
    std::cout << "===============\n\n";

    // Find phase boundaries
    size_t outage_start_idx = 0;
    size_t outage_end_idx = 0;
    for (size_t i = 0; i < traj.time.size(); ++i) {
        if (traj.time[i] >= config.gps_outage_start_s && outage_start_idx == 0) {
            outage_start_idx = i;
        }
        if (traj.time[i] >= config.gps_outage_start_s + config.gps_outage_duration_s &&
            outage_end_idx == 0) {
            outage_end_idx = i;
        }
    }

    // GPS phase statistics
    double rmse_gps = 0.0;
    for (size_t i = 0; i < outage_start_idx; ++i) {
        rmse_gps += traj.pos_error_m[i] * traj.pos_error_m[i];
    }
    rmse_gps = std::sqrt(rmse_gps / outage_start_idx);

    // Outage phase statistics
    double max_outage_error = 0.0;
    for (size_t i = outage_start_idx; i < outage_end_idx; ++i) {
        max_outage_error = std::max(max_outage_error, traj.pos_error_m[i]);
    }

    // Recovery phase statistics
    double final_error = traj.pos_error_m.back();
    double convergence_time = -1.0;
    for (size_t i = outage_end_idx; i < traj.time.size(); ++i) {
        if (traj.pos_error_m[i] < 500.0) {  // 500m threshold
            convergence_time = traj.time[i] -
                (config.gps_outage_start_s + config.gps_outage_duration_s);
            break;
        }
    }

    std::cout << "GPS/INS Phase (0-" << config.gps_outage_start_s << "s):\n";
    std::cout << "  RMSE:           " << std::setprecision(2) << rmse_gps << " m\n";
    std::cout << "  Error at end:   " << traj.pos_error_m[outage_start_idx] << " m\n\n";

    std::cout << "Outage Phase (" << config.gps_outage_start_s << "-"
              << (config.gps_outage_start_s + config.gps_outage_duration_s) << "s):\n";
    std::cout << "  Max error:      " << max_outage_error << " m\n";
    std::cout << "  Error at end:   " << traj.pos_error_m[outage_end_idx] << " m\n\n";

    std::cout << "Recovery Phase (" << (config.gps_outage_start_s + config.gps_outage_duration_s)
              << "-" << config.duration_s << "s):\n";
    std::cout << "  Final error:    " << final_error << " m\n";
    if (convergence_time > 0) {
        std::cout << "  Convergence:    " << convergence_time << " s to <500m\n";
    } else {
        std::cout << "  Convergence:    Did not converge to <500m\n";
    }
    std::cout << "\n";

    // Save trajectory
    sim.saveTrajectoryCSV(output_file, traj);

    std::cout << "\nDone.\n";

    return 0;
}
