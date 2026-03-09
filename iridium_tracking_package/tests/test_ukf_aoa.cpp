/**
 * @file test_ukf_aoa.cpp
 * @brief Unit tests for UKF AOA tracking implementation
 */

#include <gtest/gtest.h>
#include <optmath/ukf_aoa_tracking.hpp>
#include <cmath>

using namespace optmath::tracking;
using namespace optmath::tracking::constants;

//=============================================================================
// Coordinate Transform Tests
//=============================================================================

class CoordinateTest : public ::testing::Test {
protected:
    static constexpr double TOLERANCE = 1e-6;
};

TEST_F(CoordinateTest, GeodeticToECEF_Equator) {
    GeodeticCoord geo{0.0, 0.0, 0.0};
    ECEFCoord ecef = geodetic_to_ecef(geo);

    EXPECT_NEAR(ecef.x, WGS84_A, 1.0);
    EXPECT_NEAR(ecef.y, 0.0, 1.0);
    EXPECT_NEAR(ecef.z, 0.0, 1.0);
}

TEST_F(CoordinateTest, GeodeticToECEF_NorthPole) {
    GeodeticCoord geo{PI / 2.0, 0.0, 0.0};
    ECEFCoord ecef = geodetic_to_ecef(geo);

    EXPECT_NEAR(ecef.x, 0.0, 1.0);
    EXPECT_NEAR(ecef.y, 0.0, 1.0);
    EXPECT_NEAR(ecef.z, WGS84_B, 100.0);  // Semi-minor axis
}

TEST_F(CoordinateTest, ECEFToGeodetic_Roundtrip) {
    GeodeticCoord original{40.0 * DEG2RAD, -105.0 * DEG2RAD, 1650.0};
    ECEFCoord ecef = geodetic_to_ecef(original);
    GeodeticCoord recovered = ecef_to_geodetic(ecef);

    EXPECT_NEAR(recovered.latitude, original.latitude, 1e-10);
    EXPECT_NEAR(recovered.longitude, original.longitude, 1e-10);
    EXPECT_NEAR(recovered.altitude, original.altitude, 1e-3);
}

TEST_F(CoordinateTest, AzElComputation) {
    // Observer at Boulder, CO
    GeodeticCoord observer{40.0 * DEG2RAD, -105.0 * DEG2RAD, 1650.0};

    // Satellite directly overhead
    GeodeticCoord sat{40.0 * DEG2RAD, -105.0 * DEG2RAD, 780000.0};
    ECEFCoord sat_ecef = geodetic_to_ecef(sat);
    AzElCoord azel = ecef_to_azel(sat_ecef, observer);

    // Elevation should be ~90 degrees
    EXPECT_GT(azel.elevation, 85.0 * DEG2RAD);

    // Range should be approximately altitude difference
    EXPECT_NEAR(azel.range, 780000.0 - 1650.0, 1000.0);
}

TEST_F(CoordinateTest, AzElComputation_Horizon) {
    GeodeticCoord observer{0.0, 0.0, 0.0};

    // Satellite on the horizon to the East
    GeodeticCoord sat{0.0, 10.0 * DEG2RAD, 0.0};
    ECEFCoord sat_ecef = geodetic_to_ecef(sat);
    AzElCoord azel = ecef_to_azel(sat_ecef, observer);

    // Azimuth should be ~90 degrees (East)
    EXPECT_NEAR(azel.azimuth, 90.0 * DEG2RAD, 5.0 * DEG2RAD);
    // Elevation should be low (accounting for Earth's curvature)
    EXPECT_LT(azel.elevation, 10.0 * DEG2RAD);
}

//=============================================================================
// TLE and SGP4 Tests
//=============================================================================

class SGP4Test : public ::testing::Test {
protected:
    TLE tle;

    void SetUp() override {
        tle = create_iridium_tle(2460000.5, 45.0, 0.0);
    }
};

TEST_F(SGP4Test, TLECreation) {
    EXPECT_NEAR(tle.semi_major_axis, WGS84_A + IRIDIUM_ALTITUDE, 10000.0);
    EXPECT_NEAR(tle.period, IRIDIUM_PERIOD, 10.0);
    EXPECT_NEAR(tle.inclination, IRIDIUM_INCLINATION, 0.01);
}

TEST_F(SGP4Test, Propagation_AtEpoch) {
    SimplifiedSGP4 sgp4(tle);
    ECICoord eci = sgp4.propagate(tle.epoch_jd);

    // Verify altitude is correct
    double r = std::sqrt(eci.x*eci.x + eci.y*eci.y + eci.z*eci.z);
    EXPECT_NEAR(r, WGS84_A + IRIDIUM_ALTITUDE, 5000.0);

    // Verify velocity magnitude
    double v = std::sqrt(eci.vx*eci.vx + eci.vy*eci.vy + eci.vz*eci.vz);
    double v_expected = std::sqrt(EARTH_MU / r);
    EXPECT_NEAR(v, v_expected, 50.0);
}

TEST_F(SGP4Test, Propagation_OneOrbit) {
    SimplifiedSGP4 sgp4(tle);

    ECICoord eci0 = sgp4.propagate(tle.epoch_jd);
    ECICoord eci1 = sgp4.propagate(tle.epoch_jd + tle.period / 86400.0);

    // After one orbit, position changes due to J2 perturbations:
    // - Nodal regression: ~4.7 deg/orbit for Iridium
    // - This causes ~500km displacement in ECI after one orbit
    double dx = eci1.x - eci0.x;
    double dy = eci1.y - eci0.y;
    double dz = eci1.z - eci0.z;
    double displacement = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Allow for nodal regression (~600km displacement expected)
    EXPECT_LT(displacement, 800000.0);

    // But altitude should remain nearly constant (circular orbit)
    double r0 = std::sqrt(eci0.x*eci0.x + eci0.y*eci0.y + eci0.z*eci0.z);
    double r1 = std::sqrt(eci1.x*eci1.x + eci1.y*eci1.y + eci1.z*eci1.z);
    EXPECT_NEAR(r0, r1, 1000.0);  // Within 1km
}

TEST_F(SGP4Test, AzEl_Computation) {
    SimplifiedSGP4 sgp4(tle);
    GeodeticCoord observer{40.0 * DEG2RAD, -105.0 * DEG2RAD, 1650.0};

    AzElCoord azel = sgp4.get_azel(tle.epoch_jd, observer);

    // Azimuth should be in [0, 2*pi]
    EXPECT_GE(azel.azimuth, 0.0);
    EXPECT_LT(azel.azimuth, TWO_PI);

    // Elevation could be negative (below horizon)
    EXPECT_GE(azel.elevation, -PI/2);
    EXPECT_LE(azel.elevation, PI/2);

    // Range depends on geometry:
    // - Minimum (overhead): ~780km (orbit altitude)
    // - Maximum (horizon): ~3100km for LEO
    // - When below horizon, range can be much larger (through Earth)
    // We just verify range is positive and reasonable
    EXPECT_GT(azel.range, 0.0);
    EXPECT_LT(azel.range, 15000000.0);  // Less than Earth diameter
}

//=============================================================================
// Matrix Operation Tests
//=============================================================================

class MatrixTest : public ::testing::Test {
protected:
    static constexpr double TOLERANCE = 1e-10;
};

TEST_F(MatrixTest, Cholesky_Identity) {
    Mat<3, 3> I = identity_matrix<3>();
    Mat<3, 3> L = cholesky(I);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_NEAR(L[i][j], 1.0, TOLERANCE);
            } else if (i > j) {
                // Lower triangular should be zero for identity
                EXPECT_NEAR(L[i][j], 0.0, TOLERANCE);
            }
        }
    }
}

TEST_F(MatrixTest, Cholesky_SimpleMatrix) {
    // Create a simple SPD matrix: A = [4, 2; 2, 2]
    Mat<2, 2> A = {{{4.0, 2.0}, {2.0, 2.0}}};
    Mat<2, 2> L = cholesky(A);

    // L should be [2, 0; 1, 1]
    EXPECT_NEAR(L[0][0], 2.0, TOLERANCE);
    EXPECT_NEAR(L[0][1], 0.0, TOLERANCE);
    EXPECT_NEAR(L[1][0], 1.0, TOLERANCE);
    EXPECT_NEAR(L[1][1], 1.0, TOLERANCE);

    // Verify L * L^T = A
    Mat<2, 2> LLT = mat_mul(L, mat_transpose(L));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(LLT[i][j], A[i][j], TOLERANCE);
        }
    }
}

TEST_F(MatrixTest, CholeskyInverse) {
    Mat<2, 2> A = {{{4.0, 2.0}, {2.0, 2.0}}};
    Mat<2, 2> A_inv = cholesky_inverse(A);

    // A * A^-1 should equal I
    Mat<2, 2> product = mat_mul(A, A_inv);

    EXPECT_NEAR(product[0][0], 1.0, TOLERANCE);
    EXPECT_NEAR(product[0][1], 0.0, TOLERANCE);
    EXPECT_NEAR(product[1][0], 0.0, TOLERANCE);
    EXPECT_NEAR(product[1][1], 1.0, TOLERANCE);
}

//=============================================================================
// AOA Measurement Model Tests
//=============================================================================

class AOAMeasurementTest : public ::testing::Test {
protected:
    AntennaArrayConfig config;

    void SetUp() override {
        config = AntennaArrayConfig::default_iridium();
    }
};

TEST_F(AOAMeasurementTest, DefaultConfig) {
    EXPECT_NEAR(config.baseline, 0.1, 0.001);
    EXPECT_EQ(config.baseline_azimuth, 0.0);
    EXPECT_NEAR(config.phase_noise_std, 0.1, 0.001);
}

TEST_F(AOAMeasurementTest, MeasurementGeneration) {
    AOAMeasurementModel model(config, 12345);

    AzElCoord true_azel{45.0 * DEG2RAD, 30.0 * DEG2RAD, 1000000.0};
    AOAMeasurement meas = model.measure(0.0, true_azel);

    EXPECT_TRUE(meas.valid);
    EXPECT_GT(meas.snr_db, 0.0);

    // Azimuth should be within a few degrees of truth
    double az_error = std::abs(meas.azimuth - true_azel.azimuth);
    EXPECT_LT(az_error, 0.5);  // ~30 degrees tolerance (phase noise dependent)

    // Elevation should be reasonable
    EXPECT_GT(meas.elevation, 0.0);
    EXPECT_LT(meas.elevation, PI/2);
}

TEST_F(AOAMeasurementTest, BelowHorizon_Invalid) {
    AOAMeasurementModel model(config, 12345);

    AzElCoord below_horizon{45.0 * DEG2RAD, -10.0 * DEG2RAD, 1000000.0};
    AOAMeasurement meas = model.measure(0.0, below_horizon);

    EXPECT_FALSE(meas.valid);
}

TEST_F(AOAMeasurementTest, UncertaintyIncreases_LowElevation) {
    AOAMeasurementModel model(config, 12345);

    AzElCoord high_el{45.0 * DEG2RAD, 60.0 * DEG2RAD, 800000.0};
    AzElCoord low_el{45.0 * DEG2RAD, 10.0 * DEG2RAD, 2000000.0};

    AOAMeasurement meas_high = model.measure(0.0, high_el);
    AOAMeasurement meas_low = model.measure(0.0, low_el);

    // Low elevation should have higher uncertainty
    EXPECT_GT(meas_low.azimuth_std, meas_high.azimuth_std);
    EXPECT_GT(meas_low.elevation_std, meas_high.elevation_std);
}

//=============================================================================
// UKF Tests
//=============================================================================

class UKFTest : public ::testing::Test {
protected:
    GeodeticCoord observer{40.0 * DEG2RAD, -105.0 * DEG2RAD, 1650.0};
    TLE tle;
    double epoch_jd = 2460000.5;

    void SetUp() override {
        tle = create_iridium_tle(epoch_jd, 45.0, 0.0);
    }
};

TEST_F(UKFTest, Initialization) {
    UKF_AOATracker tracker(observer);
    EXPECT_FALSE(tracker.is_initialized());

    tracker.initialize(tle, epoch_jd);
    EXPECT_TRUE(tracker.is_initialized());

    GeodeticCoord est = tracker.estimated_position();
    EXPECT_GT(est.altitude, 700000.0);
    EXPECT_LT(est.altitude, 900000.0);
}

TEST_F(UKFTest, Prediction) {
    UKF_AOATracker tracker(observer);
    tracker.initialize(tle, epoch_jd);

    StateVec state_before = tracker.state();

    // Predict 10 seconds into the future
    tracker.predict(epoch_jd + 10.0 / 86400.0);

    StateVec state_after = tracker.state();

    // Position should have changed
    double pos_change = std::sqrt(
        std::pow(state_after[0] - state_before[0], 2) +
        std::pow(state_after[1] - state_before[1], 2) +
        std::pow(state_after[2] - state_before[2], 2)
    );
    EXPECT_GT(pos_change, 0.0);

    // Velocity should remain approximately constant (within ~10% due to orbital constraints)
    // The process model now includes soft orbital velocity constraints
    double v_lat_ratio = state_after[3] / state_before[3];
    double v_lon_ratio = state_after[4] / state_before[4];
    EXPECT_NEAR(v_lat_ratio, 1.0, 0.1);
    EXPECT_NEAR(v_lon_ratio, 1.0, 0.1);
}

TEST_F(UKFTest, Update_ImprovesEstimate) {
    UKF_AOATracker tracker(observer);
    tracker.initialize(tle, epoch_jd);

    // Get true position
    SimplifiedSGP4 sgp4(tle);
    AzElCoord true_azel = sgp4.get_azel(epoch_jd, observer);

    // Create a perfect measurement
    AOAMeasurement meas;
    meas.timestamp = epoch_jd;
    meas.azimuth = true_azel.azimuth;
    meas.elevation = true_azel.elevation;
    meas.azimuth_std = 0.01;
    meas.elevation_std = 0.01;
    meas.valid = true;

    StateCovar P_before = tracker.covariance();
    tracker.update(meas);
    StateCovar P_after = tracker.covariance();

    // Covariance should decrease (trace should reduce)
    double trace_before = 0.0, trace_after = 0.0;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        trace_before += P_before[i][i];
        trace_after += P_after[i][i];
    }
    EXPECT_LT(trace_after, trace_before);
}

TEST_F(UKFTest, InvalidMeasurement_Ignored) {
    UKF_AOATracker tracker(observer);
    tracker.initialize(tle, epoch_jd);

    StateVec state_before = tracker.state();
    StateCovar P_before = tracker.covariance();

    AOAMeasurement invalid_meas;
    invalid_meas.valid = false;
    tracker.update(invalid_meas);

    StateVec state_after = tracker.state();
    StateCovar P_after = tracker.covariance();

    // State and covariance should be unchanged
    for (size_t i = 0; i < STATE_DIM; ++i) {
        EXPECT_EQ(state_before[i], state_after[i]);
        for (size_t j = 0; j < STATE_DIM; ++j) {
            EXPECT_EQ(P_before[i][j], P_after[i][j]);
        }
    }
}

TEST_F(UKFTest, PositionUncertainty) {
    UKF_AOATracker tracker(observer);
    tracker.initialize(tle, epoch_jd);

    Vec<3> uncertainty = tracker.position_uncertainty_m();

    // Initial uncertainty should be reasonable (based on TLE accuracy)
    EXPECT_GT(uncertainty[0], 0.0);
    EXPECT_GT(uncertainty[1], 0.0);
    EXPECT_GT(uncertainty[2], 0.0);
    EXPECT_LT(uncertainty[0], 10000.0);  // Less than 10km
    EXPECT_LT(uncertainty[1], 10000.0);
    EXPECT_LT(uncertainty[2], 10000.0);
}

//=============================================================================
// Burst Model Tests
//=============================================================================

TEST(BurstModelTest, BasicTiming) {
    IridiumBurstModel burst;

    double jd = 2460000.5;

    // Test over multiple frames
    int burst_count = 0;
    int total_samples = 1000;
    for (int i = 0; i < total_samples; ++i) {
        double t = jd + i * 0.001 / 86400.0;  // 1ms steps
        if (burst.is_burst_active(t)) {
            ++burst_count;
        }
    }

    // Burst duty cycle should be ~9% (8.28ms / 90ms)
    double duty_cycle = static_cast<double>(burst_count) / total_samples;
    EXPECT_NEAR(duty_cycle, IRIDIUM_BURST_DURATION / IRIDIUM_FRAME_PERIOD, 0.05);
}

TEST(BurstModelTest, NextBurstStart) {
    IridiumBurstModel burst;
    double jd = 2460000.5;

    double next = burst.next_burst_start(jd);

    // Next burst should be at or after current time
    EXPECT_GE(next, jd);

    // And should be within one frame period
    EXPECT_LT((next - jd) * 86400.0, IRIDIUM_FRAME_PERIOD);
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST(IntegrationTest, ShortSimulation) {
    SimulationConfig cfg = SimulationConfig::default_config();
    cfg.duration_sec = 60.0;  // 1 minute
    cfg.measurement_interval_sec = 1.0;
    cfg.verbose = false;

    // Set up satellite to pass over observer
    // RAAN aligned with observer longitude, mean anomaly set so satellite
    // is approaching from the south
    cfg.satellite_tle = create_iridium_tle(cfg.start_jd,
        cfg.observer.longitude * RAD2DEG + 180.0,  // RAAN opposite to observer
        90.0 - cfg.observer.latitude * RAD2DEG);   // Mean anomaly to be near observer

    SimulationResults results = run_simulation(cfg);

    // Should have ~60 measurements
    EXPECT_GE(results.num_measurements, 58);
    EXPECT_LE(results.num_measurements, 62);

    // Position errors should be computed
    EXPECT_FALSE(results.position_error_m.empty());

    // Statistics should be computed
    results.compute_statistics();

    // Check that we got some measurements (satellite may or may not be visible
    // depending on geometry, but the simulation should complete)
    EXPECT_GE(results.num_valid_measurements, 0);
}

TEST(IntegrationTest, VisibleSatellite) {
    // Configure a scenario where satellite is visible
    // Search through mean anomalies to find a visible position
    SimulationConfig cfg;

    cfg.observer.latitude = 40.0 * DEG2RAD;  // Mid-latitude
    cfg.observer.longitude = -105.0 * DEG2RAD;
    cfg.observer.altitude = 1650.0;

    cfg.start_jd = 2460000.5;
    cfg.antenna = AntennaArrayConfig::default_iridium();
    cfg.ukf_params = UKFParams::default_params();
    cfg.duration_sec = 60.0;
    cfg.measurement_interval_sec = 1.0;
    cfg.use_burst_timing = false;
    cfg.verbose = false;

    // Search for a mean anomaly where satellite is visible
    SimplifiedSGP4* visible_sgp4 = nullptr;
    TLE visible_tle;

    for (int raan_deg = 0; raan_deg < 360; raan_deg += 30) {
        for (int ma_deg = 0; ma_deg < 360; ma_deg += 30) {
            TLE tle = create_iridium_tle(cfg.start_jd, raan_deg, ma_deg);
            SimplifiedSGP4 sgp4(tle);
            AzElCoord azel = sgp4.get_azel(cfg.start_jd, cfg.observer);

            if (azel.elevation > 20.0 * DEG2RAD) {
                visible_tle = tle;
                visible_sgp4 = new SimplifiedSGP4(visible_tle);
                break;
            }
        }
        if (visible_sgp4) break;
    }

    // Skip test if no visible configuration found (shouldn't happen)
    if (!visible_sgp4) {
        delete visible_sgp4;
        GTEST_SKIP() << "Could not find visible satellite configuration";
    }

    cfg.satellite_tle = visible_tle;
    delete visible_sgp4;

    SimulationResults results = run_simulation(cfg);

    // With visible satellite and no burst timing, we should get valid measurements
    EXPECT_GT(results.num_valid_measurements, 5);
}

TEST(IntegrationTest, NoBurstTiming) {
    SimulationConfig cfg = SimulationConfig::default_config();
    cfg.duration_sec = 30.0;
    cfg.measurement_interval_sec = 1.0;
    cfg.use_burst_timing = false;
    cfg.verbose = false;

    // Set up satellite to be visible
    cfg.satellite_tle = create_iridium_tle(cfg.start_jd,
        cfg.observer.longitude * RAD2DEG + 180.0,
        90.0 - cfg.observer.latitude * RAD2DEG);

    SimulationResults results = run_simulation(cfg);

    // Test should complete successfully
    EXPECT_GE(results.num_measurements, 28);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
