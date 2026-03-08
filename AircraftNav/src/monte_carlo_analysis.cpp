/**
 * @file monte_carlo_analysis.cpp
 * @brief Monte Carlo analysis for aircraft navigation simulation
 *
 * Runs multiple simulation trials in parallel and computes
 * aggregate statistics for filter performance analysis.
 */

#include <iostream>
#include <string>
#include <cstdlib>

#include "AircraftNavSimulation.h"
#include "MonteCarloRunner.h"

using namespace AircraftNav;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  --trials <count>       Number of Monte Carlo trials (default: 1000)\n"
              << "  --duration <seconds>   Simulation duration (default: 300)\n"
              << "  --outage-start <sec>   GPS outage start time (default: 60)\n"
              << "  --outage-duration <s>  GPS outage duration (default: 30)\n"
              << "  --turbulence <level>   Turbulence: light/moderate/severe (default: moderate)\n"
              << "  --threads <count>      Number of parallel threads (default: auto)\n"
              << "  --output <prefix>      Output file prefix (default: monte_carlo)\n"
              << "  --seed <value>         Base random seed (default: random)\n"
              << "  --verbose              Enable verbose progress output\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char* argv[]) {
    MonteCarloConfig mc_config;
    SimulationConfig sim_config = SimulationConfig::default_config();
    std::string output_prefix = "monte_carlo";
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--trials" && i + 1 < argc) {
            mc_config.num_trials = std::atoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            mc_config.simulation_duration_s = std::atof(argv[++i]);
            sim_config.duration_s = mc_config.simulation_duration_s;
        } else if (arg == "--outage-start" && i + 1 < argc) {
            mc_config.gps_outage_start_s = std::atof(argv[++i]);
            sim_config.gps_outage_start_s = mc_config.gps_outage_start_s;
        } else if (arg == "--outage-duration" && i + 1 < argc) {
            mc_config.gps_outage_duration_s = std::atof(argv[++i]);
            sim_config.gps_outage_duration_s = mc_config.gps_outage_duration_s;
        } else if (arg == "--turbulence" && i + 1 < argc) {
            std::string level = argv[++i];
            if (level == "light") {
                sim_config.aircraft.turbulence_severity = DrydenConfig::Severity::LIGHT;
            } else if (level == "moderate") {
                sim_config.aircraft.turbulence_severity = DrydenConfig::Severity::MODERATE;
            } else if (level == "severe") {
                sim_config.aircraft.turbulence_severity = DrydenConfig::Severity::SEVERE;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            mc_config.num_threads = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            mc_config.base_seed = std::stoull(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
            mc_config.verbose = true;
        }
    }

    // Disable verbose output in simulation for MC runs
    sim_config.verbose = false;
    sim_config.save_trajectory = false;

    // Print configuration
    std::cout << "Monte Carlo Analysis - Aircraft Navigation\n";
    std::cout << "==========================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Trials:           " << mc_config.num_trials << "\n";
    std::cout << "  Duration:         " << mc_config.simulation_duration_s << " s\n";
    std::cout << "  GPS outage:       " << mc_config.gps_outage_start_s << " - "
              << (mc_config.gps_outage_start_s + mc_config.gps_outage_duration_s) << " s\n";
    std::cout << "  Threads:          " << (mc_config.num_threads == 0 ? "auto" :
                                            std::to_string(mc_config.num_threads)) << "\n";
    std::cout << "  Turbulence:       ";
    switch (sim_config.aircraft.turbulence_severity) {
        case DrydenConfig::Severity::LIGHT: std::cout << "light\n"; break;
        case DrydenConfig::Severity::MODERATE: std::cout << "moderate\n"; break;
        case DrydenConfig::Severity::SEVERE: std::cout << "severe\n"; break;
    }
    std::cout << "  Output prefix:    " << output_prefix << "\n";
    std::cout << "  Base seed:        " << (mc_config.base_seed == 0 ? "random" :
                                            std::to_string(mc_config.base_seed)) << "\n";
    std::cout << "\n";

    // Create Monte Carlo runner
    MonteCarloRunner runner(mc_config);

    // Define trial function
    auto trial_func = [&sim_config](int trial_id, uint64_t seed) -> TrialResult {
        AircraftNavSimulation sim(sim_config, seed);
        return sim.runTrial(trial_id, seed);
    };

    // Run Monte Carlo simulation
    std::cout << "Running Monte Carlo simulation...\n\n";

    MonteCarloStatistics stats = runner.run(trial_func);

    // Print results
    runner.printStatistics(stats);

    // Save results to CSV
    std::string summary_file = output_prefix + "_summary.csv";
    std::string stats_file = output_prefix + "_statistics.csv";

    runner.saveSummaryCSV(summary_file);
    runner.saveStatisticsCSV(stats_file, stats);

    // Print convergence analysis
    std::cout << "\nConvergence Analysis\n";
    std::cout << "====================\n\n";

    const auto& results = runner.getResults();

    // Count trials by convergence time
    int converged_15s = 0, converged_30s = 0, converged_60s = 0;
    for (const auto& r : results) {
        if (r.converged) {
            if (r.convergence_time_s <= 15.0) converged_15s++;
            if (r.convergence_time_s <= 30.0) converged_30s++;
            if (r.convergence_time_s <= 60.0) converged_60s++;
        }
    }

    std::cout << "  Converged within 15s:  " << converged_15s << " ("
              << std::fixed << std::setprecision(1)
              << (100.0 * converged_15s / mc_config.num_trials) << "%)\n";
    std::cout << "  Converged within 30s:  " << converged_30s << " ("
              << (100.0 * converged_30s / mc_config.num_trials) << "%)\n";
    std::cout << "  Converged within 60s:  " << converged_60s << " ("
              << (100.0 * converged_60s / mc_config.num_trials) << "%)\n";
    std::cout << "  Did not converge:      " << (mc_config.num_trials - stats.num_converged) << " ("
              << (100.0 * (mc_config.num_trials - stats.num_converged) / mc_config.num_trials) << "%)\n";
    std::cout << "  Diverged:              " << stats.num_diverged << " ("
              << (100.0 * stats.num_diverged / mc_config.num_trials) << "%)\n";

    // Print expected performance targets
    std::cout << "\nPerformance vs Targets\n";
    std::cout << "======================\n\n";

    bool target_gps = stats.mean_rmse_gps_m < 5.0;
    bool target_outage = stats.mean_rmse_outage_m < 500.0;
    bool target_recovery = stats.mean_final_error_m < 100.0;
    bool target_convergence = stats.mean_convergence_time_s < 60.0;

    std::cout << "  GPS phase <5m:         " << (target_gps ? "PASS" : "FAIL")
              << " (" << stats.mean_rmse_gps_m << " m)\n";
    std::cout << "  Outage <500m:          " << (target_outage ? "PASS" : "FAIL")
              << " (" << stats.mean_rmse_outage_m << " m)\n";
    std::cout << "  Recovery <100m:        " << (target_recovery ? "PASS" : "FAIL")
              << " (" << stats.mean_final_error_m << " m)\n";
    std::cout << "  Convergence <60s:      " << (target_convergence ? "PASS" : "FAIL")
              << " (" << stats.mean_convergence_time_s << " s)\n";
    std::cout << "  95th percentile <500m: "
              << (stats.percentile_95_error_m < 500.0 ? "PASS" : "FAIL")
              << " (" << stats.percentile_95_error_m << " m)\n";

    std::cout << "\nDone.\n";

    return 0;
}
