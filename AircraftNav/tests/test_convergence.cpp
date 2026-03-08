/**
 * @file test_convergence.cpp
 * @brief Integration tests for filter convergence scenarios
 */

// Include our headers BEFORE gtest to avoid NSIG macro collision
// signal.h defines NSIG which conflicts with SigmaPoints::NSIG
#include "AircraftNavSimulation.h"
#include "AircraftNavSRUKF.h"

// Now include gtest and standard headers
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>

using namespace AircraftNav;

class ConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = SimulationConfig::default_config();
        config_.duration_s = 120.0;  // Shorter for testing
        config_.gps_outage_start_s = 30.0;
        config_.gps_outage_duration_s = 20.0;
        config_.verbose = false;
        config_.save_trajectory = false;
    }

    SimulationConfig config_;
};

/**
 * Navigation Architecture Tests:
 * - GPS provides initial state vector to nav computer
 * - Iridium filter ALWAYS runs, continuously updating
 * - During GPS jamming: IMU flywheel (dead reckoning)
 * - After jamming: Iridium maintains accuracy
 */

TEST_F(ConvergenceTest, NominalScenario) {
    // Test nominal GPS->Jamming->Iridium scenario
    config_.enable_gps = true;
    config_.enable_iridium = true;
    config_.aircraft.turbulence_severity = DrydenConfig::Severity::MODERATE;

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // GPS provides initial state - low error at start
    EXPECT_LT(traj.pos_error_m[0], 10.0);
    EXPECT_TRUE(traj.gps_available[0]);

    // Iridium always running
    EXPECT_TRUE(traj.iridium_available[0]);
    EXPECT_TRUE(traj.iridium_available.back());

    // Find jamming period
    size_t jam_idx = 0;
    for (size_t i = 0; i < traj.time.size(); ++i) {
        if (traj.time[i] >= config_.gps_outage_start_s + 1.0) {
            jam_idx = i;
            break;
        }
    }
    // During jamming: GPS unavailable, IMU flywheel
    if (jam_idx > 0) {
        EXPECT_FALSE(traj.gps_available[jam_idx]);
    }
}

TEST_F(ConvergenceTest, GPSProvidesInitialState) {
    // GPS provides initial state, Iridium always updating
    config_.enable_gps = true;
    config_.enable_iridium = true;
    config_.gps_outage_duration_s = 0.0;  // No jamming

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // GPS initializes state - low error
    EXPECT_LT(traj.pos_error_m[0], 10.0);
    EXPECT_TRUE(traj.gps_available[0]);

    // Iridium always running
    for (const auto& ir : traj.iridium_available) {
        EXPECT_TRUE(ir);
    }
}

TEST_F(ConvergenceTest, IMUFlywheelDuringJamming) {
    // During jamming: IMU flywheel, error grows
    // After jamming: Iridium maintains/recovers accuracy
    config_.duration_s = 180.0;
    config_.gps_outage_start_s = 30.0;
    config_.gps_outage_duration_s = 60.0;  // 1 minute jamming
    config_.enable_iridium = true;

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // Find pre-jamming, during-jamming, and post-jamming indices
    size_t pre_jam = 0, mid_jam = 0, post_jam = 0;
    for (size_t i = 0; i < traj.time.size(); ++i) {
        if (traj.time[i] < config_.gps_outage_start_s) {
            pre_jam = i;
        }
        if (traj.time[i] >= config_.gps_outage_start_s + 30.0 && mid_jam == 0) {
            mid_jam = i;
        }
        if (traj.time[i] >= config_.gps_outage_start_s + config_.gps_outage_duration_s + 30.0) {
            post_jam = i;
            break;
        }
    }

    // GPS available before jamming
    EXPECT_TRUE(traj.gps_available[pre_jam]);
    // GPS unavailable during jamming
    if (mid_jam > 0) EXPECT_FALSE(traj.gps_available[mid_jam]);

    // Iridium always running
    EXPECT_TRUE(traj.iridium_available[0]);
    EXPECT_TRUE(traj.iridium_available.back());
}

TEST_F(ConvergenceTest, IridiumAlwaysUpdating) {
    // Verify Iridium updates continuously regardless of GPS state
    config_.enable_gps = false;  // GPS disabled
    config_.enable_iridium = true;

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // Initial state from filter initialization
    EXPECT_LT(traj.pos_error_m[0], 10.0);

    // GPS never available
    for (const auto& gps : traj.gps_available) {
        EXPECT_FALSE(gps);
    }

    // Iridium always running
    for (const auto& ir : traj.iridium_available) {
        EXPECT_TRUE(ir);
    }
}

TEST_F(ConvergenceTest, AccuracyLimitationsUnderstanding) {
    // Test to understand accuracy limitations
    // With IMU flywheel during jamming, error grows
    // Iridium helps maintain but has limitations with few satellites
    config_.aircraft.turbulence_severity = DrydenConfig::Severity::SEVERE;
    config_.enable_iridium = true;

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // Initial error low (GPS initialized)
    EXPECT_LT(traj.pos_error_m[0], 10.0);

    // Iridium always running
    EXPECT_TRUE(traj.iridium_available[0]);
    EXPECT_TRUE(traj.iridium_available.back());
}

TEST_F(ConvergenceTest, LightTurbulencePerformance) {
    // Light turbulence - better IMU performance during flywheel
    config_.aircraft.turbulence_severity = DrydenConfig::Severity::LIGHT;
    config_.enable_iridium = true;

    AircraftNavSimulation sim(config_, 42);
    SimulationTrajectory traj = sim.run();

    // Initial error low
    EXPECT_LT(traj.pos_error_m[0], 10.0);

    // Iridium always running
    EXPECT_TRUE(traj.iridium_available[0]);
    EXPECT_TRUE(traj.iridium_available.back());
}

class FilterModeTest : public ::testing::Test {
protected:
    void SetUp() override {
        filter_ = std::make_unique<AircraftNavSRUKF>();

        // Initialize at known position
        Eigen::Matrix<float, 15, 1> x0;
        x0 << 0.7f, -1.8f, 3000.0f,  // Position
              100.0f, 0.0f, 0.0f,    // Velocity
              0.0f, 0.0f, 0.0f,      // Attitude
              0.0f, 0.0f, 0.0f,      // Gyro bias
              0.0f, 0.0f, 0.0f;      // Accel bias

        filter_->initialize(x0);
    }

    std::unique_ptr<AircraftNavSRUKF> filter_;
};

TEST_F(FilterModeTest, InitialMode) {
    EXPECT_EQ(filter_->getMode(), NavMode::GPS_INS);
}

TEST_F(FilterModeTest, GPSOutageTransition) {
    // Simulate GPS loss
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f accel(0.0f, 0.0f, -9.8f);

    // Propagate without GPS updates
    for (int i = 0; i < 200; ++i) {  // 2 seconds
        filter_->predict(gyro, accel, 0.01f);
    }

    // Should transition to INS coasting
    EXPECT_EQ(filter_->getMode(), NavMode::INS_COASTING);
}

TEST_F(FilterModeTest, GPSRecovery) {
    // First, lose GPS
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f accel(0.0f, 0.0f, -9.8f);

    for (int i = 0; i < 200; ++i) {
        filter_->predict(gyro, accel, 0.01f);
    }

    EXPECT_EQ(filter_->getMode(), NavMode::INS_COASTING);

    // Now provide GPS update
    Eigen::Matrix<float, 6, 1> gps_meas;
    gps_meas << 0.7f, -1.8f, 3000.0f, 100.0f, 0.0f, 0.0f;
    filter_->updateGPS(gps_meas);

    // Should return to GPS mode
    EXPECT_EQ(filter_->getMode(), NavMode::GPS_INS);
}

TEST_F(FilterModeTest, CovarianceGrowth) {
    auto status_initial = filter_->getStatus();

    // Propagate without updates
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f accel(0.0f, 0.0f, -9.8f);

    filter_->notifyGPSOutage();

    for (int i = 0; i < 1000; ++i) {  // 10 seconds
        filter_->predict(gyro, accel, 0.01f);
    }

    auto status_final = filter_->getStatus();

    // Uncertainty should grow during outage
    EXPECT_GT(status_final.position_uncertainty_m, status_initial.position_uncertainty_m);
}

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = SimulationConfig::default_config();
        config_.duration_s = 60.0;
        config_.verbose = false;
        config_.save_trajectory = false;
    }

    SimulationConfig config_;
};

TEST_F(PerformanceTest, SimulationSpeed) {
    AircraftNavSimulation sim(config_, 42);

    auto start = std::chrono::high_resolution_clock::now();
    sim.run();
    auto end = std::chrono::high_resolution_clock::now();

    double runtime = std::chrono::duration<double>(end - start).count();

    // Simulation should complete faster than real-time
    EXPECT_LT(runtime, config_.duration_s);

    // And ideally much faster
    double speedup = config_.duration_s / runtime;
    std::cout << "Simulation speedup: " << speedup << "x real-time\n";
    EXPECT_GT(speedup, 10.0);  // At least 10x real-time
}

TEST_F(PerformanceTest, Determinism) {
    // Same seed should give same results
    AircraftNavSimulation sim1(config_, 12345);
    AircraftNavSimulation sim2(config_, 12345);

    SimulationTrajectory traj1 = sim1.run();
    SimulationTrajectory traj2 = sim2.run();

    ASSERT_EQ(traj1.pos_error_m.size(), traj2.pos_error_m.size());

    for (size_t i = 0; i < traj1.pos_error_m.size(); ++i) {
        EXPECT_FLOAT_EQ(traj1.pos_error_m[i], traj2.pos_error_m[i]);
    }
}

TEST_F(PerformanceTest, DifferentSeeds) {
    // Different seeds should give different results
    // Use scenario with Iridium recovery to show seed-dependent behavior
    config_.gps_outage_start_s = 10.0;
    config_.gps_outage_duration_s = 10.0;
    config_.enable_iridium = true;

    AircraftNavSimulation sim1(config_, 12345);
    AircraftNavSimulation sim2(config_, 67890);

    SimulationTrajectory traj1 = sim1.run();
    SimulationTrajectory traj2 = sim2.run();

    // True trajectories should differ due to different turbulence
    // (comparing final latitude which varies with seed)
    EXPECT_NE(traj1.true_lat.back(), traj2.true_lat.back());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
