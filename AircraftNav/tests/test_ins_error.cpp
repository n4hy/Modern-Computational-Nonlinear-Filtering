/**
 * @file test_ins_error.cpp
 * @brief Unit tests for INS error model and state space model
 */

#include <gtest/gtest.h>
#include <cmath>

#include "AircraftNavStateSpaceModel.h"
#include "AircraftAntennaModel.h"

using namespace AircraftNav;

class StateSpaceModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ = std::make_unique<AircraftNavStateSpaceModel>();
    }

    std::unique_ptr<AircraftNavStateSpaceModel> model_;
};

TEST_F(StateSpaceModelTest, InitialCovariance) {
    auto P0 = AircraftNavStateSpaceModel::getInitialCovariance();

    // Check dimensions
    EXPECT_EQ(P0.rows(), 15);
    EXPECT_EQ(P0.cols(), 15);

    // Check positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 15, 15>> solver(P0);
    for (int i = 0; i < 15; ++i) {
        EXPECT_GT(solver.eigenvalues()(i), 0.0f);
    }

    // Check reasonable initial uncertainties
    float R_M = 6371000.0f;
    float pos_std_m = std::sqrt(P0(LAT, LAT)) * R_M;
    EXPECT_NEAR(pos_std_m, 10.0f, 5.0f);  // ~10m position uncertainty

    float vel_std = std::sqrt(P0(VN, VN));
    EXPECT_NEAR(vel_std, 1.0f, 0.5f);  // ~1.0 m/s velocity uncertainty
}

TEST_F(StateSpaceModelTest, ProcessNoise) {
    auto Q = model_->Q(0.0f);

    // Check positive semi-definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 15, 15>> solver(Q);
    for (int i = 0; i < 15; ++i) {
        EXPECT_GE(solver.eigenvalues()(i), 0.0f);
    }
}

TEST_F(StateSpaceModelTest, GPSMeasurement) {
    // Create test state
    Eigen::Matrix<float, 15, 1> x;
    x << 0.7f, -1.8f, 3000.0f,  // Position
         100.0f, 0.0f, 0.0f,     // Velocity
         0.0f, 0.0f, 0.0f,       // Attitude
         0.0f, 0.0f, 0.0f,       // Gyro bias
         0.0f, 0.0f, 0.0f;       // Accel bias

    auto y = model_->h(x, 0.0f);

    // GPS measurement should observe position and velocity
    EXPECT_EQ(y.size(), 6);
    EXPECT_FLOAT_EQ(y(0), x(LAT));
    EXPECT_FLOAT_EQ(y(1), x(LON));
    EXPECT_FLOAT_EQ(y(2), x(ALT));
    EXPECT_FLOAT_EQ(y(3), x(VN));
    EXPECT_FLOAT_EQ(y(4), x(VE));
    EXPECT_FLOAT_EQ(y(5), x(VD));
}

TEST_F(StateSpaceModelTest, MeasurementNoise) {
    auto R = model_->R(0.0f);

    // Check dimensions
    EXPECT_EQ(R.rows(), 6);
    EXPECT_EQ(R.cols(), 6);

    // Check positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> solver(R);
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(solver.eigenvalues()(i), 0.0f);
    }

    // Check reasonable noise levels
    float R_M = 6371000.0f;
    float pos_noise_m = std::sqrt(R(0, 0)) * R_M;
    EXPECT_NEAR(pos_noise_m, 4.0f, 1.0f);  // ~4m position noise
}

class IridiumMeasurementTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ = std::make_unique<IridiumMeasurementModel>(15.0f);
    }

    std::unique_ptr<IridiumMeasurementModel> model_;
};

TEST_F(IridiumMeasurementTest, NoiseFromSNR) {
    // Test that noise values vary with SNR (within clamp limits)
    model_->updateNoiseFromSNR(20.0f);  // High SNR
    auto R_high = model_->R(0.0f);

    model_->updateNoiseFromSNR(5.0f);  // Low SNR
    auto R_low = model_->R(0.0f);

    // Lower SNR should have higher (or equal if clamped) noise
    EXPECT_GE(R_low(0, 0), R_high(0, 0));
    EXPECT_GE(R_low(1, 1), R_high(1, 1));
    EXPECT_GE(R_low(2, 2), R_high(2, 2));

    // Doppler noise should vary more (wider range)
    EXPECT_GT(R_low(2, 2), R_high(2, 2));
}

TEST_F(IridiumMeasurementTest, MeasurementModel) {
    // Set satellite directly overhead
    IridiumMeasurementModel::SatellitePosition sat;
    sat.lat = 0.7f;   // Same as aircraft
    sat.lon = -1.8f;
    sat.alt = 780000.0f;
    sat.v_lat = 0.0f;
    sat.v_lon = 1e-5f;  // Moving eastward
    sat.v_alt = 0.0f;

    model_->setSatellitePosition(sat);

    // Aircraft state (under satellite)
    Eigen::Matrix<float, 15, 1> x;
    x << 0.7f, -1.8f, 3000.0f,
         100.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f;

    auto y = model_->h(x, 0.0f);

    // Elevation should be high (satellite nearly overhead)
    EXPECT_GT(y(1), 1.0f);  // > 60 degrees
}

class AntennaModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        antenna_ = std::make_unique<AircraftAntennaModel>(AntennaConfig::default_aircraft(), 42);
    }

    std::unique_ptr<AircraftAntennaModel> antenna_;
};

TEST_F(AntennaModelTest, VisibilityCheck) {
    // Aircraft at Boulder, CO
    float ac_lat = 40.0f * M_PI / 180.0f;
    float ac_lon = -105.0f * M_PI / 180.0f;
    float ac_alt = 3048.0f;

    // Satellite directly overhead
    float sat_lat = ac_lat;
    float sat_lon = ac_lon;
    float sat_alt = 780000.0f;

    auto vis = antenna_->checkVisibility(
        ac_lat, ac_lon, ac_alt,
        0.0f, 0.0f, 0.0f,  // Level flight
        sat_lat, sat_lon, sat_alt);

    EXPECT_TRUE(vis.visible);
    EXPECT_GT(vis.elevation_rad, 1.4f);  // > 80 degrees
    EXPECT_GT(vis.snr_dB, 10.0f);        // Good SNR
}

TEST_F(AntennaModelTest, HorizonCheck) {
    float ac_lat = 40.0f * M_PI / 180.0f;
    float ac_lon = -105.0f * M_PI / 180.0f;
    float ac_alt = 3048.0f;

    // Satellite well below horizon
    float sat_lat = -40.0f * M_PI / 180.0f;  // Southern hemisphere
    float sat_lon = ac_lon;
    float sat_alt = 780000.0f;

    auto vis = antenna_->checkVisibility(
        ac_lat, ac_lon, ac_alt,
        0.0f, 0.0f, 0.0f,
        sat_lat, sat_lon, sat_alt);

    EXPECT_FALSE(vis.visible);
}

TEST_F(AntennaModelTest, AOAMeasurement) {
    float ac_lat = 40.0f * M_PI / 180.0f;
    float ac_lon = -105.0f * M_PI / 180.0f;
    float ac_alt = 3048.0f;
    float sat_lat = ac_lat + 0.1f;  // Slightly north
    float sat_lon = ac_lon;
    float sat_alt = 780000.0f;

    auto vis = antenna_->checkVisibility(
        ac_lat, ac_lon, ac_alt,
        0.0f, 0.0f, 0.0f,
        sat_lat, sat_lon, sat_alt);

    ASSERT_TRUE(vis.visible);

    // Measure AOA multiple times
    float az_sum = 0.0f, el_sum = 0.0f;
    const int N = 100;

    for (int i = 0; i < N; ++i) {
        auto [az, el] = antenna_->measureAOA(vis);
        az_sum += az;
        el_sum += el;
    }

    float az_mean = az_sum / N;
    float el_mean = el_sum / N;

    // Mean should be close to true value
    EXPECT_NEAR(az_mean, vis.azimuth_rad, 0.1f);
    EXPECT_NEAR(el_mean, vis.elevation_rad, 0.1f);
}

TEST_F(AntennaModelTest, PhaseAmbiguity) {
    // Baseline = 10cm, wavelength = 18.44cm
    // Baseline < λ/2, so no ambiguity expected for default config
    int n_amb = antenna_->getPhaseAmbiguityCount();

    // For 10cm baseline at 1626 MHz (λ ≈ 18.4cm):
    // d/λ ≈ 0.54, so we expect ~1 ambiguity
    EXPECT_LE(n_amb, 2);
}

TEST_F(AntennaModelTest, LinkBudget) {
    float ac_lat = 40.0f * M_PI / 180.0f;
    float ac_lon = -105.0f * M_PI / 180.0f;
    float ac_alt = 3048.0f;

    // High elevation satellite
    auto vis_high = antenna_->checkVisibility(
        ac_lat, ac_lon, ac_alt,
        0.0f, 0.0f, 0.0f,
        ac_lat, ac_lon, 780000.0f);

    // Low elevation satellite (30° off)
    auto vis_low = antenna_->checkVisibility(
        ac_lat, ac_lon, ac_alt,
        0.0f, 0.0f, 0.0f,
        ac_lat + 0.5f, ac_lon, 780000.0f);

    if (vis_high.visible && vis_low.visible) {
        // Higher elevation should have better SNR
        EXPECT_GT(vis_high.snr_dB, vis_low.snr_dB);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
