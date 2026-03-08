/**
 * @file test_aircraft_dynamics.cpp
 * @brief Unit tests for aircraft dynamics model
 */

#include <gtest/gtest.h>
#include <cmath>

#include "AircraftDynamicsModel.h"
#include "DrydenTurbulenceModel.h"

using namespace AircraftNav;

class AircraftDynamicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.latitude = 40.0 * M_PI / 180.0;
        config_.longitude = -105.0 * M_PI / 180.0;
        config_.altitude = 3048.0;  // 10,000 ft
        config_.airspeed = 103.0;   // 200 knots
        config_.heading = 0.0;      // North
        config_.enable_turbulence = false;
    }

    AircraftDynamicsConfig config_;
};

TEST_F(AircraftDynamicsTest, Initialization) {
    AircraftDynamicsModel model(config_);
    const auto& state = model.state();

    EXPECT_NEAR(state.lat, config_.latitude, 1e-6);
    EXPECT_NEAR(state.lon, config_.longitude, 1e-6);
    EXPECT_NEAR(state.alt, config_.altitude, 1e-6);
    EXPECT_NEAR(state.v_n, config_.airspeed, 1e-3);
    EXPECT_NEAR(state.v_e, 0.0, 1e-3);
}

TEST_F(AircraftDynamicsTest, SteadyFlight) {
    AircraftDynamicsModel model(config_);

    // Propagate for 10 seconds
    for (int i = 0; i < 1000; ++i) {
        model.propagate(0.01);
    }

    const auto& state = model.state();

    // Should maintain approximately level flight
    EXPECT_NEAR(state.roll, 0.0, 0.1);
    EXPECT_NEAR(state.pitch, 0.0, 0.1);

    // Altitude should be stable (within 200m due to simplified dynamics)
    EXPECT_NEAR(state.alt, config_.altitude, 200.0);

    // Airspeed should be maintained
    double speed = std::sqrt(state.v_n * state.v_n + state.v_e * state.v_e);
    EXPECT_NEAR(speed, config_.airspeed, 5.0);
}

TEST_F(AircraftDynamicsTest, PositionProgression) {
    config_.heading = 0.0;  // North
    AircraftDynamicsModel model(config_);

    double initial_lat = model.state().lat;

    // Propagate for 60 seconds
    for (int i = 0; i < 6000; ++i) {
        model.propagate(0.01);
    }

    // Should have moved north
    EXPECT_GT(model.state().lat, initial_lat);

    // Distance traveled should be approximately airspeed * time
    double R = 6371000.0;
    double dist = (model.state().lat - initial_lat) * R;
    double expected_dist = config_.airspeed * 60.0;
    EXPECT_NEAR(dist, expected_dist, expected_dist * 0.1);  // 10% tolerance
}

TEST_F(AircraftDynamicsTest, GravityComputation) {
    // At equator, sea level
    float g_equator = AircraftDynamicsModel::computeGravity(0.0f, 0.0f);
    EXPECT_NEAR(g_equator, 9.78f, 0.05f);

    // At poles, sea level
    float g_pole = AircraftDynamicsModel::computeGravity(M_PI / 2.0f, 0.0f);
    EXPECT_NEAR(g_pole, 9.83f, 0.05f);

    // At altitude
    float g_altitude = AircraftDynamicsModel::computeGravity(0.0f, 10000.0f);
    EXPECT_LT(g_altitude, g_equator);
}

TEST_F(AircraftDynamicsTest, RadiiOfCurvature) {
    float R_M, R_N;

    // At equator
    AircraftDynamicsModel::computeRadii(0.0f, R_M, R_N);
    EXPECT_NEAR(R_M, 6335439.0f, 1000.0f);  // Meridian
    EXPECT_NEAR(R_N, 6378137.0f, 1.0f);     // Transverse

    // At poles (R_M should increase, R_N stays similar)
    float R_M_pole, R_N_pole;
    AircraftDynamicsModel::computeRadii(static_cast<float>(M_PI / 2.0), R_M_pole, R_N_pole);
    EXPECT_GT(R_M_pole, R_M);
}

class DrydenTurbulenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = DrydenConfig::create(3048.0f, 103.0f, DrydenConfig::Severity::MODERATE);
    }

    DrydenConfig config_;
};

TEST_F(DrydenTurbulenceTest, ZeroMean) {
    DrydenTurbulenceModel model(config_, 42);

    // Generate many samples
    const int N = 10000;
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();

    for (int i = 0; i < N; ++i) {
        sum += model.update(0.01f);
    }

    Eigen::Vector3f mean = sum / N;

    // Mean should be approximately zero
    EXPECT_NEAR(mean(0), 0.0f, 0.5f);
    EXPECT_NEAR(mean(1), 0.0f, 0.5f);
    EXPECT_NEAR(mean(2), 0.0f, 0.5f);
}

TEST_F(DrydenTurbulenceTest, Correlation) {
    DrydenTurbulenceModel model(config_, 42);

    // Turbulence should be correlated (not white noise)
    Eigen::Vector3f prev = model.update(0.01f);
    int sign_changes = 0;

    for (int i = 0; i < 100; ++i) {
        Eigen::Vector3f curr = model.update(0.01f);
        if ((prev(0) > 0) != (curr(0) > 0)) {
            sign_changes++;
        }
        prev = curr;
    }

    // Correlated signal should have fewer sign changes than white noise
    EXPECT_LT(sign_changes, 40);  // White noise would have ~50
}

TEST_F(DrydenTurbulenceTest, SeverityScaling) {
    auto config_light = DrydenConfig::create(3048.0f, 103.0f, DrydenConfig::Severity::LIGHT);
    auto config_severe = DrydenConfig::create(3048.0f, 103.0f, DrydenConfig::Severity::SEVERE);

    EXPECT_LT(config_light.sigma_u, config_.sigma_u);
    EXPECT_GT(config_severe.sigma_u, config_.sigma_u);
}

class INSErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = INSErrorConfig::create(INSErrorConfig::Grade::TACTICAL);
    }

    INSErrorConfig config_;
};

TEST_F(INSErrorTest, BiasRandomWalk) {
    INSErrorModel model(config_, 42);

    Eigen::Vector3f initial_bias = model.getGyroBias();

    // Propagate for 1 hour
    for (int i = 0; i < 3600 * 100; ++i) {
        model.update(0.01f);
    }

    Eigen::Vector3f final_bias = model.getGyroBias();
    Eigen::Vector3f drift = final_bias - initial_bias;

    // Drift should be on the order of bias stability
    float drift_mag = drift.norm();
    float expected_drift = config_.gyro_bias_stability_rad_s * std::sqrt(3600.0f);
    EXPECT_LT(drift_mag, expected_drift * 10.0f);  // Wide tolerance
}

TEST_F(INSErrorTest, MeasurementCorruption) {
    INSErrorModel model(config_, 42);

    Eigen::Vector3f true_gyro(0.0f, 0.0f, 0.1f);
    Eigen::Vector3f true_accel(0.0f, 0.0f, 9.8f);

    Eigen::Vector3f corrupted_gyro = model.corruptGyro(true_gyro);
    Eigen::Vector3f corrupted_accel = model.corruptAccel(true_accel);

    // Corrupted values should be different but close
    EXPECT_NE(corrupted_gyro, true_gyro);
    EXPECT_NE(corrupted_accel, true_accel);

    float gyro_error = (corrupted_gyro - true_gyro).norm();
    float accel_error = (corrupted_accel - true_accel).norm();

    EXPECT_LT(gyro_error, 0.05f);  // Small error (includes bias + noise)
    EXPECT_LT(accel_error, 0.5f);  // Includes bias + noise
}

TEST_F(INSErrorTest, PositionErrorGrowth) {
    INSErrorModel model(config_, 42);

    float error_1min = model.getExpectedPositionError(60.0f);
    float error_10min = model.getExpectedPositionError(600.0f);
    float error_1hr = model.getExpectedPositionError(3600.0f);

    // Error should grow with time
    EXPECT_LT(error_1min, error_10min);
    EXPECT_LT(error_10min, error_1hr);

    // Tactical INS error growth varies based on configuration
    // Error model uses simplified quadratic growth
    EXPECT_GT(error_1hr, 100.0f);
    EXPECT_LT(error_1hr, 50000.0f);  // Wide tolerance for simplified model
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
