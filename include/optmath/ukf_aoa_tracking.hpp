/**
 * @file ukf_aoa_tracking.hpp
 * @brief Unscented Kalman Filter for Angle-of-Arrival Satellite Tracking
 *
 * Implements UKF-based position estimation using AOA measurements from a
 * two-antenna coherent receiver array tracking Iridium-Next satellites.
 *
 * Features:
 * - SGP4 simplified orbital propagator for TLE-based predictions
 * - WGS84 geodetic coordinate transformations
 * - Two-antenna AOA measurement model with configurable baseline
 * - Unscented Kalman Filter with tunable parameters
 * - Burst transmission simulation for Iridium-Next
 *
 * @author OptMathKernels
 * @version 0.5.0
 */

#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <optional>
#include <string>

namespace optmath {
namespace tracking {

//=============================================================================
// Constants
//=============================================================================

namespace constants {
    // WGS84 ellipsoid parameters
    constexpr double WGS84_A = 6378137.0;              // Semi-major axis [m]
    constexpr double WGS84_F = 1.0 / 298.257223563;    // Flattening
    constexpr double WGS84_B = WGS84_A * (1.0 - WGS84_F); // Semi-minor axis
    constexpr double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F; // First eccentricity squared

    // Earth rotation and gravity
    constexpr double EARTH_MU = 3.986004418e14;        // Gravitational parameter [m^3/s^2]
    constexpr double EARTH_OMEGA = 7.2921159e-5;       // Rotation rate [rad/s]
    constexpr double EARTH_J2 = 1.08263e-3;            // J2 perturbation coefficient

    // Physical constants
    constexpr double SPEED_OF_LIGHT = 299792458.0;     // [m/s]
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * PI;
    constexpr double DEG2RAD = PI / 180.0;
    constexpr double RAD2DEG = 180.0 / PI;

    // Iridium-Next parameters
    constexpr double IRIDIUM_ALTITUDE = 780000.0;      // Nominal altitude [m]
    constexpr double IRIDIUM_INCLINATION = 86.4 * DEG2RAD; // Orbital inclination
    constexpr double IRIDIUM_PERIOD = 100.4 * 60.0;    // Orbital period [s]
    constexpr double IRIDIUM_FREQUENCY = 1626.0e6;     // L-band downlink [Hz]
    constexpr double IRIDIUM_WAVELENGTH = SPEED_OF_LIGHT / IRIDIUM_FREQUENCY;

    // Burst timing (Iridium uses TDMA with ~8.28ms frames)
    constexpr double IRIDIUM_BURST_DURATION = 8.28e-3; // [s]
    constexpr double IRIDIUM_FRAME_PERIOD = 90.0e-3;   // Simplex burst period [s]
}

//=============================================================================
// Vector/Matrix types (lightweight for embedded use)
//=============================================================================

template<size_t N>
using Vec = std::array<double, N>;

template<size_t ROWS, size_t COLS>
using Mat = std::array<std::array<double, COLS>, ROWS>;

// Common vector operations
template<size_t N>
Vec<N> vec_add(const Vec<N>& a, const Vec<N>& b) {
    Vec<N> result;
    for (size_t i = 0; i < N; ++i) result[i] = a[i] + b[i];
    return result;
}

template<size_t N>
Vec<N> vec_sub(const Vec<N>& a, const Vec<N>& b) {
    Vec<N> result;
    for (size_t i = 0; i < N; ++i) result[i] = a[i] - b[i];
    return result;
}

template<size_t N>
Vec<N> vec_scale(const Vec<N>& a, double s) {
    Vec<N> result;
    for (size_t i = 0; i < N; ++i) result[i] = a[i] * s;
    return result;
}

template<size_t N>
double vec_dot(const Vec<N>& a, const Vec<N>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) sum += a[i] * b[i];
    return sum;
}

template<size_t N>
double vec_norm(const Vec<N>& a) {
    return std::sqrt(vec_dot(a, a));
}

// Matrix operations
template<size_t N>
Mat<N, N> mat_add(const Mat<N, N>& A, const Mat<N, N>& B) {
    Mat<N, N> result;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            result[i][j] = A[i][j] + B[i][j];
    return result;
}

template<size_t N>
Mat<N, N> mat_scale(const Mat<N, N>& A, double s) {
    Mat<N, N> result;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            result[i][j] = A[i][j] * s;
    return result;
}

template<size_t N, size_t M, size_t P>
Mat<N, P> mat_mul(const Mat<N, M>& A, const Mat<M, P>& B) {
    Mat<N, P> result{};
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < P; ++j)
            for (size_t k = 0; k < M; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}

template<size_t N, size_t M>
Vec<N> mat_vec_mul(const Mat<N, M>& A, const Vec<M>& x) {
    Vec<N> result{};
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

template<size_t N>
Mat<N, N> mat_transpose(const Mat<N, N>& A) {
    Mat<N, N> result;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            result[i][j] = A[j][i];
    return result;
}

template<size_t N>
Mat<N, N> outer_product(const Vec<N>& a, const Vec<N>& b) {
    Mat<N, N> result;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            result[i][j] = a[i] * b[j];
    return result;
}

template<size_t N>
Mat<N, N> identity_matrix() {
    Mat<N, N> result{};
    for (size_t i = 0; i < N; ++i) result[i][i] = 1.0;
    return result;
}

/**
 * @brief Cholesky decomposition A = L * L^T
 * @param A Symmetric positive definite matrix
 * @return Lower triangular matrix L
 * @throws std::runtime_error if matrix is not positive definite
 */
template<size_t N>
Mat<N, N> cholesky(const Mat<N, N>& A) {
    Mat<N, N> L{};

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (size_t k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }

            if (i == j) {
                if (sum <= 0.0) {
                    throw std::runtime_error("Matrix is not positive definite");
                }
                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return L;
}

/**
 * @brief Matrix inversion via Cholesky for SPD matrices
 */
template<size_t N>
Mat<N, N> cholesky_inverse(const Mat<N, N>& A) {
    Mat<N, N> L = cholesky(A);
    Mat<N, N> L_inv{};

    // Invert L (lower triangular)
    for (size_t i = 0; i < N; ++i) {
        L_inv[i][i] = 1.0 / L[i][i];
        for (size_t j = 0; j < i; ++j) {
            double sum = 0.0;
            for (size_t k = j; k < i; ++k) {
                sum += L[i][k] * L_inv[k][j];
            }
            L_inv[i][j] = -sum / L[i][i];
        }
    }

    // A^-1 = L^-T * L^-1
    Mat<N, N> result{};
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = i; k < N; ++k) {
                sum += L_inv[k][i] * L_inv[k][j];
            }
            result[i][j] = sum;
            result[j][i] = sum;
        }
    }
    return result;
}

//=============================================================================
// Coordinate Systems
//=============================================================================

struct GeodeticCoord {
    double latitude;   // [rad]
    double longitude;  // [rad]
    double altitude;   // [m] above WGS84 ellipsoid
};

struct ECEFCoord {
    double x, y, z;    // [m]
};

struct ECICoord {
    double x, y, z;    // [m]
    double vx, vy, vz; // [m/s]
};

struct AzElCoord {
    double azimuth;    // [rad] clockwise from North
    double elevation;  // [rad] above horizon
    double range;      // [m] slant range
};

/**
 * @brief Convert geodetic (lat/lon/alt) to ECEF
 */
inline ECEFCoord geodetic_to_ecef(const GeodeticCoord& geo) {
    using namespace constants;

    double sin_lat = std::sin(geo.latitude);
    double cos_lat = std::cos(geo.latitude);
    double sin_lon = std::sin(geo.longitude);
    double cos_lon = std::cos(geo.longitude);

    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

    ECEFCoord ecef;
    ecef.x = (N + geo.altitude) * cos_lat * cos_lon;
    ecef.y = (N + geo.altitude) * cos_lat * sin_lon;
    ecef.z = (N * (1.0 - WGS84_E2) + geo.altitude) * sin_lat;

    return ecef;
}

/**
 * @brief Convert ECEF to geodetic (iterative algorithm)
 */
inline GeodeticCoord ecef_to_geodetic(const ECEFCoord& ecef) {
    using namespace constants;

    double p = std::sqrt(ecef.x * ecef.x + ecef.y * ecef.y);
    double lon = std::atan2(ecef.y, ecef.x);

    // Iterative latitude computation (Bowring's method)
    double lat = std::atan2(ecef.z, p * (1.0 - WGS84_E2));
    double N, h;

    for (int iter = 0; iter < 10; ++iter) {
        double sin_lat = std::sin(lat);
        N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
        h = p / std::cos(lat) - N;
        double lat_new = std::atan2(ecef.z, p * (1.0 - WGS84_E2 * N / (N + h)));
        if (std::abs(lat_new - lat) < 1e-12) break;
        lat = lat_new;
    }

    return {lat, lon, h};
}

/**
 * @brief Compute ENU (East-North-Up) rotation matrix at observer location
 */
inline Mat<3, 3> ecef_to_enu_rotation(const GeodeticCoord& observer) {
    double sin_lat = std::sin(observer.latitude);
    double cos_lat = std::cos(observer.latitude);
    double sin_lon = std::sin(observer.longitude);
    double cos_lon = std::cos(observer.longitude);

    // Rotation matrix: ECEF -> ENU
    Mat<3, 3> R;
    R[0][0] = -sin_lon;           R[0][1] = cos_lon;            R[0][2] = 0.0;
    R[1][0] = -sin_lat * cos_lon; R[1][1] = -sin_lat * sin_lon; R[1][2] = cos_lat;
    R[2][0] = cos_lat * cos_lon;  R[2][1] = cos_lat * sin_lon;  R[2][2] = sin_lat;

    return R;
}

/**
 * @brief Convert satellite ECEF position to Az/El from observer
 */
inline AzElCoord ecef_to_azel(const ECEFCoord& sat, const GeodeticCoord& observer) {
    ECEFCoord obs_ecef = geodetic_to_ecef(observer);

    // Relative position in ECEF
    Vec<3> delta = {sat.x - obs_ecef.x, sat.y - obs_ecef.y, sat.z - obs_ecef.z};

    // Transform to ENU
    Mat<3, 3> R = ecef_to_enu_rotation(observer);
    Vec<3> enu = mat_vec_mul(R, delta);

    double e = enu[0]; // East
    double n = enu[1]; // North
    double u = enu[2]; // Up

    double range = vec_norm(enu);
    double elevation = std::asin(u / range);
    double azimuth = std::atan2(e, n); // Clockwise from North

    if (azimuth < 0) azimuth += constants::TWO_PI;

    return {azimuth, elevation, range};
}

/**
 * @brief Convert ECI to ECEF given Greenwich Mean Sidereal Time
 */
inline ECEFCoord eci_to_ecef(const ECICoord& eci, double gmst) {
    double cos_gmst = std::cos(gmst);
    double sin_gmst = std::sin(gmst);

    ECEFCoord ecef;
    ecef.x = cos_gmst * eci.x + sin_gmst * eci.y;
    ecef.y = -sin_gmst * eci.x + cos_gmst * eci.y;
    ecef.z = eci.z;

    return ecef;
}

/**
 * @brief Compute GMST from Julian Date
 */
inline double julian_date_to_gmst(double jd) {
    using namespace constants;

    double T = (jd - 2451545.0) / 36525.0;

    // GMST in seconds at 0h UT
    double gmst_sec = 24110.54841 + 8640184.812866 * T
                    + 0.093104 * T * T
                    - 6.2e-6 * T * T * T;

    // Add time since 0h UT
    double ut = std::fmod(jd + 0.5, 1.0);
    gmst_sec += ut * 86400.0 * 1.00273790935;

    // Convert to radians
    return std::fmod(gmst_sec * TWO_PI / 86400.0, TWO_PI);
}

//=============================================================================
// SGP4 Simplified Orbital Propagator
//=============================================================================

/**
 * @brief Two-Line Element (TLE) orbital parameters
 */
struct TLE {
    std::string name;
    int catalog_number;
    double epoch_jd;        // Julian date of epoch
    double mean_motion;     // [rad/s]
    double eccentricity;
    double inclination;     // [rad]
    double raan;            // Right ascension of ascending node [rad]
    double arg_perigee;     // Argument of perigee [rad]
    double mean_anomaly;    // [rad] at epoch
    double bstar;           // Drag coefficient [1/Earth radii]

    // Derived quantities
    double semi_major_axis; // [m]
    double period;          // [s]

    void compute_derived() {
        using namespace constants;
        // Kepler's third law: a^3 = mu / n^2
        semi_major_axis = std::cbrt(EARTH_MU / (mean_motion * mean_motion));
        period = TWO_PI / mean_motion;
    }
};

/**
 * @brief Create a sample Iridium-Next TLE
 */
inline TLE create_iridium_tle(double epoch_jd, double raan_deg = 0.0, double mean_anomaly_deg = 0.0) {
    using namespace constants;

    TLE tle;
    tle.name = "IRIDIUM-NEXT";
    tle.catalog_number = 99999;
    tle.epoch_jd = epoch_jd;
    tle.mean_motion = TWO_PI / IRIDIUM_PERIOD;
    tle.eccentricity = 0.0002;  // Near-circular
    tle.inclination = IRIDIUM_INCLINATION;
    tle.raan = raan_deg * DEG2RAD;
    tle.arg_perigee = 0.0;
    tle.mean_anomaly = mean_anomaly_deg * DEG2RAD;
    tle.bstar = 0.0001;
    tle.compute_derived();

    return tle;
}

/**
 * @brief Simplified SGP4 propagator (without deep-space corrections)
 *
 * This is a simplified version suitable for LEO satellites like Iridium.
 * For production use, consider full SGP4/SDP4 implementation.
 */
class SimplifiedSGP4 {
public:
    explicit SimplifiedSGP4(const TLE& tle) : tle_(tle) {
        initialize();
    }

    /**
     * @brief Propagate satellite position to given time
     * @param jd Julian date
     * @return Position and velocity in ECI frame
     */
    ECICoord propagate(double jd) const {
        using namespace constants;

        double dt = (jd - tle_.epoch_jd) * 86400.0; // Time since epoch [s]

        // Mean motion with secular perturbations (J2)
        double n0 = tle_.mean_motion;
        double a0 = tle_.semi_major_axis;
        double e0 = tle_.eccentricity;
        double i0 = tle_.inclination;

        double cos_i = std::cos(i0);
        double sin_i = std::sin(i0);

        // J2 perturbation rates
        double p = a0 * (1.0 - e0 * e0);
        double j2_factor = 1.5 * EARTH_J2 * (WGS84_A / p) * (WGS84_A / p);

        // Nodal regression rate
        double raan_dot = -j2_factor * n0 * cos_i;

        // Argument of perigee precession
        double omega_dot = j2_factor * n0 * (2.0 - 2.5 * sin_i * sin_i);

        // Mean motion secular variation
        double n_dot = 1.5 * j2_factor * n0 * n0 * (1.0 - 1.5 * sin_i * sin_i)
                     / std::sqrt(1.0 - e0 * e0);

        // Update orbital elements
        double n = n0 + n_dot * dt;
        double raan = tle_.raan + raan_dot * dt;
        double omega = tle_.arg_perigee + omega_dot * dt;
        double M = tle_.mean_anomaly + n * dt;
        M = std::fmod(M, TWO_PI);
        if (M < 0) M += TWO_PI;

        // Solve Kepler's equation: E - e*sin(E) = M
        double E = M;
        for (int iter = 0; iter < 10; ++iter) {
            double dE = (M - E + e0 * std::sin(E)) / (1.0 - e0 * std::cos(E));
            E += dE;
            if (std::abs(dE) < 1e-12) break;
        }

        // True anomaly
        double cos_E = std::cos(E);
        double sin_E = std::sin(E);
        double sqrt_1_minus_e2 = std::sqrt(1.0 - e0 * e0);
        double nu = std::atan2(sqrt_1_minus_e2 * sin_E, cos_E - e0);

        // Radius
        double r = a0 * (1.0 - e0 * cos_E);

        // Position in orbital plane
        double cos_nu = std::cos(nu);
        double sin_nu = std::sin(nu);
        double x_orb = r * cos_nu;
        double y_orb = r * sin_nu;

        // Velocity in orbital plane
        double v_factor = std::sqrt(EARTH_MU / (a0 * (1.0 - e0 * e0)));
        double vx_orb = -v_factor * sin_nu;
        double vy_orb = v_factor * (e0 + cos_nu);

        // Rotation to ECI
        double cos_omega = std::cos(omega);
        double sin_omega = std::sin(omega);
        double cos_raan = std::cos(raan);
        double sin_raan = std::sin(raan);

        // Combined rotation matrix elements
        double r11 = cos_omega * cos_raan - sin_omega * sin_raan * cos_i;
        double r12 = -sin_omega * cos_raan - cos_omega * sin_raan * cos_i;
        double r21 = cos_omega * sin_raan + sin_omega * cos_raan * cos_i;
        double r22 = -sin_omega * sin_raan + cos_omega * cos_raan * cos_i;
        double r31 = sin_omega * sin_i;
        double r32 = cos_omega * sin_i;

        ECICoord eci;
        eci.x = r11 * x_orb + r12 * y_orb;
        eci.y = r21 * x_orb + r22 * y_orb;
        eci.z = r31 * x_orb + r32 * y_orb;
        eci.vx = r11 * vx_orb + r12 * vy_orb;
        eci.vy = r21 * vx_orb + r22 * vy_orb;
        eci.vz = r31 * vx_orb + r32 * vy_orb;

        return eci;
    }

    /**
     * @brief Get satellite Az/El from ground observer at given time
     */
    AzElCoord get_azel(double jd, const GeodeticCoord& observer) const {
        ECICoord eci = propagate(jd);
        double gmst = julian_date_to_gmst(jd);
        ECEFCoord ecef = eci_to_ecef(eci, gmst);
        return ecef_to_azel(ecef, observer);
    }

    const TLE& get_tle() const { return tle_; }

private:
    TLE tle_;

    void initialize() {
        // Precompute constants if needed
    }
};

//=============================================================================
// Two-Antenna AOA Measurement Model
//=============================================================================

/**
 * @brief Configuration for two-antenna array
 */
struct AntennaArrayConfig {
    double baseline;           // Antenna separation [m]
    double baseline_azimuth;   // Array orientation [rad] - direction from ant1 to ant2
    double phase_noise_std;    // Phase measurement noise [rad]
    double amplitude_noise_std; // Relative amplitude noise (for elevation estimation)
    double antenna_gain_3db;   // Half-power beamwidth [rad]

    // Default configuration for Iridium L-band
    static AntennaArrayConfig default_iridium() {
        AntennaArrayConfig cfg;
        cfg.baseline = 0.1;  // 10 cm baseline (~0.5 wavelength at 1626 MHz)
        cfg.baseline_azimuth = 0.0;  // Array points North
        cfg.phase_noise_std = 0.1;   // ~6 degrees RMS
        cfg.amplitude_noise_std = 0.1; // 10% relative
        cfg.antenna_gain_3db = 60.0 * constants::DEG2RAD; // Wide beam
        return cfg;
    }
};

/**
 * @brief AOA measurement from two-antenna array
 */
struct AOAMeasurement {
    double timestamp;      // Julian date
    double azimuth;        // Measured azimuth [rad]
    double elevation;      // Measured elevation [rad]
    double phase_diff;     // Raw phase difference [rad]
    double snr_db;         // Signal-to-noise ratio [dB]
    bool valid;            // Measurement validity flag

    // Measurement uncertainties
    double azimuth_std;    // [rad]
    double elevation_std;  // [rad]
};

/**
 * @brief Two-antenna AOA measurement simulator
 */
class AOAMeasurementModel {
public:
    explicit AOAMeasurementModel(const AntennaArrayConfig& config, uint64_t seed = 0)
        : config_(config)
        , rng_(seed ? seed : std::random_device{}())
        , phase_noise_(0.0, config.phase_noise_std)
        , amplitude_noise_(0.0, config.amplitude_noise_std)
    {}

    /**
     * @brief Generate noisy AOA measurement from true Az/El
     */
    AOAMeasurement measure(double timestamp, const AzElCoord& true_azel) {
        using namespace constants;

        AOAMeasurement meas;
        meas.timestamp = timestamp;
        meas.valid = true;

        // Check if satellite is above horizon
        if (true_azel.elevation < 0.0) {
            meas.valid = false;
            return meas;
        }

        // Compute true phase difference
        // Phase = 2*pi * baseline * cos(angle_off_array) / wavelength
        // For azimuth: angle_off_array = |azimuth - baseline_azimuth|

        double az_rel = true_azel.azimuth - config_.baseline_azimuth;
        double cos_az_rel = std::cos(az_rel);
        double cos_el = std::cos(true_azel.elevation);

        // Phase difference depends on direction cosine in baseline direction
        double direction_cosine = cos_el * cos_az_rel;
        double true_phase = TWO_PI * config_.baseline * direction_cosine / IRIDIUM_WAVELENGTH;

        // Add phase noise
        double phase_noise = phase_noise_(rng_);
        meas.phase_diff = true_phase + phase_noise;

        // Wrap to [-pi, pi]
        meas.phase_diff = std::fmod(meas.phase_diff + PI, TWO_PI);
        if (meas.phase_diff < 0) meas.phase_diff += TWO_PI;
        meas.phase_diff -= PI;

        // Check for phase ambiguity (baseline > lambda/2)
        if (config_.baseline > IRIDIUM_WAVELENGTH / 2.0) {
            // Ambiguity exists - would need additional processing in real system
            // For simulation, we resolve using true azimuth knowledge
        }

        // Convert phase back to azimuth estimate
        // Simplified: assuming elevation is known or estimated separately
        double measured_direction_cosine = meas.phase_diff * IRIDIUM_WAVELENGTH / (TWO_PI * config_.baseline);
        measured_direction_cosine = std::clamp(measured_direction_cosine, -1.0, 1.0);

        // Estimate azimuth (with ambiguity handling)
        double cos_el_est = std::cos(true_azel.elevation); // In practice, estimate this too
        double cos_az_rel_est = measured_direction_cosine / cos_el_est;
        cos_az_rel_est = std::clamp(cos_az_rel_est, -1.0, 1.0);

        double az_rel_est = std::acos(cos_az_rel_est);

        // Resolve sign ambiguity using second antenna or additional info
        // For simulation, use true azimuth to resolve
        if (std::sin(az_rel) < 0) {
            az_rel_est = -az_rel_est;
        }

        meas.azimuth = az_rel_est + config_.baseline_azimuth;
        if (meas.azimuth < 0) meas.azimuth += TWO_PI;
        if (meas.azimuth >= TWO_PI) meas.azimuth -= TWO_PI;

        // Elevation estimation (from amplitude comparison or separate method)
        // Simple model: add noise proportional to elevation
        double el_noise = amplitude_noise_(rng_) * (1.0 + 0.5 * (PI/2 - true_azel.elevation));
        meas.elevation = true_azel.elevation + el_noise;
        meas.elevation = std::clamp(meas.elevation, 0.0, PI/2.0);

        // Compute uncertainties based on SNR
        double range_factor = true_azel.range / (WGS84_A + IRIDIUM_ALTITUDE);
        meas.snr_db = 30.0 - 20.0 * std::log10(range_factor) - 10.0 * std::log10(1.0 + std::abs(phase_noise));

        // Uncertainty increases at low elevation (atmospheric effects)
        double el_factor = 1.0 / std::sin(std::max(true_azel.elevation, 0.1));
        meas.azimuth_std = config_.phase_noise_std * el_factor;
        meas.elevation_std = config_.amplitude_noise_std * el_factor;

        return meas;
    }

    const AntennaArrayConfig& config() const { return config_; }

private:
    AntennaArrayConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> phase_noise_;
    std::normal_distribution<double> amplitude_noise_;
};

//=============================================================================
// Unscented Kalman Filter
//=============================================================================

/**
 * @brief UKF tuning parameters
 */
struct UKFParams {
    double alpha = 1e-3;   // Sigma point spread (typically 1e-4 to 1)
    double beta = 2.0;     // Distribution prior (2 optimal for Gaussian)
    double kappa = 0.0;    // Secondary scaling (usually 0 or 3-n)

    static UKFParams default_params() { return UKFParams{}; }
};

/**
 * @brief State vector for satellite tracking
 *
 * State: [lat, lon, alt, vlat, vlon, valt]
 * - Position in geodetic coordinates (more intuitive for ground observers)
 * - Velocity in local tangent plane rates
 */
constexpr size_t STATE_DIM = 6;  // Position + velocity
constexpr size_t MEAS_DIM = 2;   // Azimuth + elevation

using StateVec = Vec<STATE_DIM>;
using StateCovar = Mat<STATE_DIM, STATE_DIM>;
using MeasVec = Vec<MEAS_DIM>;
using MeasCovar = Mat<MEAS_DIM, MEAS_DIM>;

/**
 * @brief Unscented Kalman Filter for AOA-based satellite tracking
 */
class UKF_AOATracker {
public:
    UKF_AOATracker(const GeodeticCoord& observer, const UKFParams& params = UKFParams::default_params())
        : observer_(observer)
        , params_(params)
    {
        initialize_weights();
        reset();
    }

    /**
     * @brief Initialize filter with TLE-based prediction
     */
    void initialize(const TLE& tle, double jd) {
        SimplifiedSGP4 sgp4(tle);
        ECICoord eci = sgp4.propagate(jd);
        double gmst = julian_date_to_gmst(jd);
        ECEFCoord ecef = eci_to_ecef(eci, gmst);
        GeodeticCoord sat_geo = ecef_to_geodetic(ecef);

        // Initial state from TLE
        state_[0] = sat_geo.latitude;
        state_[1] = sat_geo.longitude;
        state_[2] = sat_geo.altitude;

        // Compute velocity by numerical differentiation using SGP4
        // This gives the actual velocity at the current orbital position
        double dt_sec = 1.0;
        double jd_next = jd + dt_sec / 86400.0;
        ECICoord eci_next = sgp4.propagate(jd_next);
        double gmst_next = julian_date_to_gmst(jd_next);
        ECEFCoord ecef_next = eci_to_ecef(eci_next, gmst_next);
        GeodeticCoord sat_geo_next = ecef_to_geodetic(ecef_next);

        state_[3] = (sat_geo_next.latitude - sat_geo.latitude) / dt_sec;
        state_[4] = (sat_geo_next.longitude - sat_geo.longitude) / dt_sec;
        state_[5] = (sat_geo_next.altitude - sat_geo.altitude) / dt_sec;

        // Handle longitude wrap-around
        if (state_[4] > constants::PI / dt_sec) {
            state_[4] -= constants::TWO_PI / dt_sec;
        } else if (state_[4] < -constants::PI / dt_sec) {
            state_[4] += constants::TWO_PI / dt_sec;
        }

        // Initial covariance (uncertainties in TLE prediction)
        P_ = {};
        P_[0][0] = std::pow(0.001 * constants::DEG2RAD, 2);  // ~100m latitude uncertainty
        P_[1][1] = std::pow(0.001 * constants::DEG2RAD, 2);  // ~100m longitude uncertainty
        P_[2][2] = std::pow(1000.0, 2);                      // 1km altitude uncertainty
        P_[3][3] = std::pow(1e-6, 2);                        // Lat rate uncertainty
        P_[4][4] = std::pow(1e-6, 2);                        // Lon rate uncertainty
        P_[5][5] = std::pow(10.0, 2);                        // Alt rate uncertainty

        last_update_jd_ = jd;
        initialized_ = true;

        // Store TLE for process model
        tle_ = tle;
    }

    /**
     * @brief Predict state to new time using orbital dynamics
     */
    void predict(double jd) {
        if (!initialized_) {
            throw std::runtime_error("UKF not initialized");
        }

        double dt = (jd - last_update_jd_) * 86400.0; // seconds
        if (std::abs(dt) < 1e-6) return;

        // Generate sigma points
        auto sigma_points = generate_sigma_points(state_, P_);

        // Propagate each sigma point through dynamics
        std::array<StateVec, 2 * STATE_DIM + 1> propagated_points;
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            propagated_points[i] = process_model(sigma_points[i], dt);
        }

        // Compute predicted mean
        StateVec x_pred = {};
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            x_pred = vec_add(x_pred, vec_scale(propagated_points[i], Wm_[i]));
        }

        // Wrap longitude to [-pi, pi]
        x_pred[1] = wrap_angle(x_pred[1]);

        // Compute predicted covariance
        StateCovar P_pred = {};
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            StateVec diff = vec_sub(propagated_points[i], x_pred);
            diff[1] = wrap_angle(diff[1]); // Wrap longitude difference
            auto outer = outer_product(diff, diff);
            P_pred = mat_add(P_pred, mat_scale(outer, Wc_[i]));
        }

        // Add process noise (scaled by dt for continuous-time noise model)
        // Q represents noise intensity, actual variance grows with time
        StateCovar Q_scaled = mat_scale(Q_, std::abs(dt));
        P_pred = mat_add(P_pred, Q_scaled);

        state_ = x_pred;
        P_ = P_pred;
        last_update_jd_ = jd;
    }

    /**
     * @brief Update state with AOA measurement
     */
    void update(const AOAMeasurement& meas) {
        if (!initialized_ || !meas.valid) return;

        // Generate sigma points from current state
        auto sigma_points = generate_sigma_points(state_, P_);

        // Transform sigma points through measurement model
        std::array<MeasVec, 2 * STATE_DIM + 1> meas_points;
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            meas_points[i] = measurement_model(sigma_points[i]);
        }

        // Predicted measurement mean
        MeasVec z_pred = {};
        for (size_t i = 0; i < meas_points.size(); ++i) {
            z_pred = vec_add(z_pred, vec_scale(meas_points[i], Wm_[i]));
        }

        // Wrap predicted azimuth
        z_pred[0] = wrap_angle_positive(z_pred[0]);

        // Measurement covariance
        MeasCovar Pzz = {};
        for (size_t i = 0; i < meas_points.size(); ++i) {
            MeasVec diff = vec_sub(meas_points[i], z_pred);
            diff[0] = wrap_angle(diff[0]); // Wrap azimuth difference
            auto outer = outer_product(diff, diff);
            Pzz = mat_add(Pzz, mat_scale(outer, Wc_[i]));
        }

        // Add measurement noise
        MeasCovar R = {};
        R[0][0] = meas.azimuth_std * meas.azimuth_std;
        R[1][1] = meas.elevation_std * meas.elevation_std;
        Pzz = mat_add(Pzz, R);

        // Cross-covariance
        Mat<STATE_DIM, MEAS_DIM> Pxz = {};
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            StateVec x_diff = vec_sub(sigma_points[i], state_);
            x_diff[1] = wrap_angle(x_diff[1]);
            MeasVec z_diff = vec_sub(meas_points[i], z_pred);
            z_diff[0] = wrap_angle(z_diff[0]);

            for (size_t r = 0; r < STATE_DIM; ++r) {
                for (size_t c = 0; c < MEAS_DIM; ++c) {
                    Pxz[r][c] += Wc_[i] * x_diff[r] * z_diff[c];
                }
            }
        }

        // Kalman gain K = Pxz * Pzz^-1
        MeasCovar Pzz_inv = cholesky_inverse(Pzz);
        Mat<STATE_DIM, MEAS_DIM> K = {};
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < MEAS_DIM; ++j) {
                for (size_t k = 0; k < MEAS_DIM; ++k) {
                    K[i][j] += Pxz[i][k] * Pzz_inv[k][j];
                }
            }
        }

        // Innovation
        MeasVec z_meas = {meas.azimuth, meas.elevation};
        MeasVec innovation = vec_sub(z_meas, z_pred);
        innovation[0] = wrap_angle(innovation[0]);

        // Update state
        StateVec dx = {};
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < MEAS_DIM; ++j) {
                dx[i] += K[i][j] * innovation[j];
            }
        }
        state_ = vec_add(state_, dx);
        state_[1] = wrap_angle(state_[1]); // Wrap longitude

        // Update covariance: P = P - K * Pzz * K^T
        // More numerically stable: P = P - K * Pxz^T
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < STATE_DIM; ++j) {
                for (size_t k = 0; k < MEAS_DIM; ++k) {
                    P_[i][j] -= K[i][k] * Pxz[j][k];
                }
            }
        }

        // Ensure symmetry and positive definiteness
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = i + 1; j < STATE_DIM; ++j) {
                double avg = 0.5 * (P_[i][j] + P_[j][i]);
                P_[i][j] = avg;
                P_[j][i] = avg;
            }
            // Ensure positive diagonal
            P_[i][i] = std::max(P_[i][i], 1e-12);
        }

        last_innovation_ = innovation;
        last_update_jd_ = meas.timestamp;
    }

    /**
     * @brief Reset filter to uninitialized state
     */
    void reset() {
        state_ = {};
        P_ = identity_matrix<STATE_DIM>();
        initialized_ = false;
        last_update_jd_ = 0.0;

        // Default process noise covariance
        Q_ = {};
        Q_[0][0] = std::pow(1e-8, 2);  // Latitude variance per second^2
        Q_[1][1] = std::pow(1e-8, 2);  // Longitude variance
        Q_[2][2] = std::pow(1.0, 2);   // Altitude variance
        Q_[3][3] = std::pow(1e-10, 2); // Lat rate variance
        Q_[4][4] = std::pow(1e-10, 2); // Lon rate variance
        Q_[5][5] = std::pow(0.1, 2);   // Alt rate variance
    }

    /**
     * @brief Set process noise covariance
     */
    void set_process_noise(const StateCovar& Q) { Q_ = Q; }

    /**
     * @brief Get current state estimate
     */
    const StateVec& state() const { return state_; }

    /**
     * @brief Get current covariance
     */
    const StateCovar& covariance() const { return P_; }

    /**
     * @brief Get estimated satellite geodetic position
     */
    GeodeticCoord estimated_position() const {
        return {state_[0], state_[1], state_[2]};
    }

    /**
     * @brief Get estimated Az/El to satellite
     */
    AzElCoord estimated_azel() const {
        ECEFCoord ecef = geodetic_to_ecef(estimated_position());
        return ecef_to_azel(ecef, observer_);
    }

    /**
     * @brief Get position uncertainty (1-sigma) in meters
     */
    Vec<3> position_uncertainty_m() const {
        // Convert lat/lon variance to meters
        double R = constants::WGS84_A + state_[2];
        double lat_m = std::sqrt(P_[0][0]) * R;
        double lon_m = std::sqrt(P_[1][1]) * R * std::cos(state_[0]);
        double alt_m = std::sqrt(P_[2][2]);
        return {lat_m, lon_m, alt_m};
    }

    /**
     * @brief Get last innovation (for debugging/monitoring)
     */
    const MeasVec& last_innovation() const { return last_innovation_; }

    bool is_initialized() const { return initialized_; }

private:
    GeodeticCoord observer_;
    UKFParams params_;

    StateVec state_;
    StateCovar P_;
    StateCovar Q_;

    // Sigma point weights
    std::array<double, 2 * STATE_DIM + 1> Wm_;
    std::array<double, 2 * STATE_DIM + 1> Wc_;
    double lambda_;

    double last_update_jd_ = 0.0;
    bool initialized_ = false;
    MeasVec last_innovation_ = {};

    // Store TLE for orbital constraints in process model
    TLE tle_;

    void initialize_weights() {
        double n = static_cast<double>(STATE_DIM);
        lambda_ = params_.alpha * params_.alpha * (n + params_.kappa) - n;

        double weight_0 = lambda_ / (n + lambda_);
        Wm_[0] = weight_0;
        Wc_[0] = weight_0 + (1.0 - params_.alpha * params_.alpha + params_.beta);

        double weight_i = 1.0 / (2.0 * (n + lambda_));
        for (size_t i = 1; i <= 2 * STATE_DIM; ++i) {
            Wm_[i] = weight_i;
            Wc_[i] = weight_i;
        }
    }

    std::array<StateVec, 2 * STATE_DIM + 1> generate_sigma_points(
        const StateVec& mean, const StateCovar& cov) const
    {
        std::array<StateVec, 2 * STATE_DIM + 1> sigma_points;
        sigma_points[0] = mean;

        // Compute matrix square root: sqrt((n + lambda) * P)
        double scale = std::sqrt(static_cast<double>(STATE_DIM) + lambda_);
        StateCovar L;
        try {
            L = cholesky(cov);
        } catch (const std::runtime_error&) {
            // Fallback: use diagonal elements
            L = {};
            for (size_t i = 0; i < STATE_DIM; ++i) {
                L[i][i] = std::sqrt(std::max(cov[i][i], 1e-12));
            }
        }

        for (size_t i = 0; i < STATE_DIM; ++i) {
            StateVec offset = {};
            for (size_t j = 0; j < STATE_DIM; ++j) {
                offset[j] = scale * L[j][i];
            }
            sigma_points[1 + i] = vec_add(mean, offset);
            sigma_points[1 + STATE_DIM + i] = vec_sub(mean, offset);
        }

        return sigma_points;
    }

    /**
     * @brief Process model: propagate state through orbital dynamics
     *
     * Uses a simplified orbital mechanics model that accounts for:
     * - Position integration with velocity
     * - Velocity changes due to orbital curvature (latitude-dependent longitude rate)
     * - Altitude constraint to orbital shell
     */
    StateVec process_model(const StateVec& x, double dt) const {
        StateVec x_new;

        // Position integration
        x_new[0] = x[0] + x[3] * dt;  // Latitude
        x_new[1] = x[1] + x[4] * dt;  // Longitude
        x_new[2] = x[2] + x[5] * dt;  // Altitude

        // Constrain altitude to orbital shell (Iridium ~780km)
        double nominal_alt = constants::IRIDIUM_ALTITUDE;
        x_new[2] = nominal_alt + (x_new[2] - nominal_alt) * 0.9;  // Soft constraint

        // Update velocity based on orbital mechanics
        // Orbital velocity magnitude should be approximately constant
        double R = constants::WGS84_A + x_new[2];
        double v_orbital = std::sqrt(constants::EARTH_MU / R);

        // As satellite moves in latitude, the longitude rate changes
        // due to the orbit's projection onto Earth's surface
        double cos_lat = std::cos(x_new[0]);
        double sin_lat = std::sin(x_new[0]);

        // For near-polar orbits, as latitude increases, ground track curves
        // The latitude rate decreases near poles, longitude rate increases
        double lat_factor = std::sqrt(1.0 - sin_lat * sin_lat * std::pow(std::sin(tle_.inclination), 2));

        // Maintain approximate velocity constraint
        double current_v_lat = x[3] * R;
        double current_v_lon = x[4] * R * cos_lat;
        double current_v = std::sqrt(current_v_lat * current_v_lat + current_v_lon * current_v_lon);

        // Scale velocities to maintain orbital speed (soft constraint)
        double v_scale = 1.0;
        if (current_v > 0.01) {  // Avoid division by zero
            v_scale = 0.95 + 0.05 * (v_orbital / current_v);
            v_scale = std::clamp(v_scale, 0.9, 1.1);  // Limit correction
        }

        x_new[3] = x[3] * v_scale;
        x_new[4] = x[4] * v_scale;
        x_new[5] = x[5] * 0.9;  // Altitude rate tends toward zero for circular orbit

        // Constrain latitude to valid range
        x_new[0] = std::clamp(x_new[0], -constants::PI / 2.0 + 0.01, constants::PI / 2.0 - 0.01);

        // Constrain altitude to reasonable range
        x_new[2] = std::clamp(x_new[2], 700000.0, 900000.0);

        return x_new;
    }

    /**
     * @brief Measurement model: state -> Az/El
     */
    MeasVec measurement_model(const StateVec& x) const {
        GeodeticCoord sat_pos = {x[0], x[1], x[2]};
        ECEFCoord ecef = geodetic_to_ecef(sat_pos);
        AzElCoord azel = ecef_to_azel(ecef, observer_);

        return {azel.azimuth, azel.elevation};
    }

    static double wrap_angle(double angle) {
        while (angle > constants::PI) angle -= constants::TWO_PI;
        while (angle < -constants::PI) angle += constants::TWO_PI;
        return angle;
    }

    static double wrap_angle_positive(double angle) {
        while (angle >= constants::TWO_PI) angle -= constants::TWO_PI;
        while (angle < 0) angle += constants::TWO_PI;
        return angle;
    }
};

//=============================================================================
// Iridium Burst Transmission Model
//=============================================================================

/**
 * @brief Model for Iridium burst transmission timing
 */
class IridiumBurstModel {
public:
    explicit IridiumBurstModel(double frame_period = constants::IRIDIUM_FRAME_PERIOD,
                               double burst_duration = constants::IRIDIUM_BURST_DURATION,
                               uint64_t seed = 0)
        : frame_period_(frame_period)
        , burst_duration_(burst_duration)
        , rng_(seed ? seed : std::random_device{}())
        , jitter_dist_(0.0, 0.001) // 1ms jitter
    {}

    /**
     * @brief Check if burst is active at given time
     */
    bool is_burst_active(double jd) const {
        double t_sec = jd * 86400.0;
        double t_in_frame = std::fmod(t_sec, frame_period_);
        return t_in_frame < burst_duration_;
    }

    /**
     * @brief Get next burst start time after given JD
     */
    double next_burst_start(double jd) const {
        double t_sec = jd * 86400.0;
        double t_in_frame = std::fmod(t_sec, frame_period_);

        if (t_in_frame < burst_duration_) {
            // Currently in a burst
            return jd;
        }

        // Wait for next frame
        double wait = frame_period_ - t_in_frame;
        return jd + wait / 86400.0;
    }

    /**
     * @brief Add realistic timing jitter to burst
     */
    double add_jitter() {
        return jitter_dist_(rng_);
    }

private:
    double frame_period_;
    double burst_duration_;
    mutable std::mt19937_64 rng_;
    std::normal_distribution<double> jitter_dist_;
};

//=============================================================================
// Complete Simulation Framework
//=============================================================================

/**
 * @brief Simulation configuration
 */
struct SimulationConfig {
    GeodeticCoord observer;           // Ground station location
    TLE satellite_tle;                // Satellite orbital elements
    AntennaArrayConfig antenna;       // AOA measurement setup
    UKFParams ukf_params;             // Filter tuning

    double start_jd;                  // Simulation start time
    double duration_sec;              // Simulation duration
    double measurement_interval_sec;  // Time between measurements

    bool use_burst_timing;            // Simulate burst transmissions
    bool verbose;                     // Print debug info

    static SimulationConfig default_config() {
        SimulationConfig cfg;

        // Default observer: Boulder, CO
        cfg.observer.latitude = 40.015 * constants::DEG2RAD;
        cfg.observer.longitude = -105.27 * constants::DEG2RAD;
        cfg.observer.altitude = 1655.0;

        // Current epoch (approximate JD for 2024)
        cfg.start_jd = 2460000.5;

        // Create Iridium TLE
        cfg.satellite_tle = create_iridium_tle(cfg.start_jd, 45.0, 0.0);

        cfg.antenna = AntennaArrayConfig::default_iridium();
        cfg.ukf_params = UKFParams::default_params();

        cfg.duration_sec = 600.0;  // 10 minutes
        cfg.measurement_interval_sec = 1.0;
        cfg.use_burst_timing = true;
        cfg.verbose = true;

        return cfg;
    }
};

/**
 * @brief Simulation results
 */
struct SimulationResults {
    std::vector<double> timestamps;

    // True values
    std::vector<Vec<3>> true_positions;     // lat, lon, alt
    std::vector<Vec<2>> true_azel;          // az, el

    // Measurements
    std::vector<Vec<2>> measured_azel;
    std::vector<bool> measurement_valid;

    // Estimates
    std::vector<Vec<3>> estimated_positions;
    std::vector<Vec<2>> estimated_azel;
    std::vector<Vec<3>> position_uncertainty;

    // Errors
    std::vector<double> position_error_m;
    std::vector<double> azimuth_error_rad;
    std::vector<double> elevation_error_rad;

    // Statistics
    double mean_position_error_m;
    double rms_position_error_m;
    double mean_az_error_deg;
    double mean_el_error_deg;
    int num_measurements;
    int num_valid_measurements;

    void compute_statistics() {
        if (position_error_m.empty()) {
            mean_position_error_m = 0.0;
            rms_position_error_m = 0.0;
            mean_az_error_deg = 0.0;
            mean_el_error_deg = 0.0;
            return;
        }

        mean_position_error_m = std::accumulate(position_error_m.begin(),
            position_error_m.end(), 0.0) / position_error_m.size();

        double sum_sq = 0.0;
        for (double e : position_error_m) sum_sq += e * e;
        rms_position_error_m = std::sqrt(sum_sq / position_error_m.size());

        double az_sum = 0.0, el_sum = 0.0;
        for (size_t i = 0; i < azimuth_error_rad.size(); ++i) {
            az_sum += std::abs(azimuth_error_rad[i]);
            el_sum += std::abs(elevation_error_rad[i]);
        }
        mean_az_error_deg = (az_sum / azimuth_error_rad.size()) * constants::RAD2DEG;
        mean_el_error_deg = (el_sum / elevation_error_rad.size()) * constants::RAD2DEG;

        num_measurements = static_cast<int>(timestamps.size());
        num_valid_measurements = std::count(measurement_valid.begin(),
            measurement_valid.end(), true);
    }
};

/**
 * @brief Run AOA tracking simulation
 */
inline SimulationResults run_simulation(const SimulationConfig& cfg) {
    using namespace constants;

    SimulationResults results;

    // Initialize components
    SimplifiedSGP4 propagator(cfg.satellite_tle);
    AOAMeasurementModel aoa_model(cfg.antenna);
    UKF_AOATracker tracker(cfg.observer, cfg.ukf_params);
    IridiumBurstModel burst_model;

    // Initialize tracker with TLE prediction
    tracker.initialize(cfg.satellite_tle, cfg.start_jd);

    double jd = cfg.start_jd;
    double end_jd = cfg.start_jd + cfg.duration_sec / 86400.0;
    double dt_jd = cfg.measurement_interval_sec / 86400.0;

    int step = 0;
    while (jd < end_jd) {
        // Get true satellite position
        ECICoord eci = propagator.propagate(jd);
        double gmst = julian_date_to_gmst(jd);
        ECEFCoord ecef = eci_to_ecef(eci, gmst);
        GeodeticCoord sat_geo = ecef_to_geodetic(ecef);
        AzElCoord true_azel = ecef_to_azel(ecef, cfg.observer);

        // Store true values
        results.timestamps.push_back(jd);
        results.true_positions.push_back({sat_geo.latitude, sat_geo.longitude, sat_geo.altitude});
        results.true_azel.push_back({true_azel.azimuth, true_azel.elevation});

        // Check burst timing if enabled
        bool can_measure = true;
        if (cfg.use_burst_timing) {
            can_measure = burst_model.is_burst_active(jd);
        }

        // Generate measurement if satellite is visible and burst is active
        bool is_visible = true_azel.elevation > 5.0 * DEG2RAD; // 5° minimum elevation
        AOAMeasurement meas;

        if (can_measure && is_visible) {
            meas = aoa_model.measure(jd, true_azel);
        } else {
            meas.valid = false;
        }

        results.measured_azel.push_back({meas.azimuth, meas.elevation});
        results.measurement_valid.push_back(meas.valid);

        // UKF predict and update
        tracker.predict(jd);

        if (meas.valid) {
            tracker.update(meas);
        }

        // Store estimates
        GeodeticCoord est_pos = tracker.estimated_position();
        AzElCoord est_azel = tracker.estimated_azel();
        Vec<3> uncertainty = tracker.position_uncertainty_m();

        results.estimated_positions.push_back({est_pos.latitude, est_pos.longitude, est_pos.altitude});
        results.estimated_azel.push_back({est_azel.azimuth, est_azel.elevation});
        results.position_uncertainty.push_back(uncertainty);

        // Compute errors
        double R = WGS84_A + sat_geo.altitude;
        double dlat = (est_pos.latitude - sat_geo.latitude) * R;
        double dlon = (est_pos.longitude - sat_geo.longitude) * R * std::cos(sat_geo.latitude);
        double dalt = est_pos.altitude - sat_geo.altitude;
        double pos_err = std::sqrt(dlat*dlat + dlon*dlon + dalt*dalt);

        results.position_error_m.push_back(pos_err);

        double az_err = est_azel.azimuth - true_azel.azimuth;
        while (az_err > PI) az_err -= TWO_PI;
        while (az_err < -PI) az_err += TWO_PI;
        results.azimuth_error_rad.push_back(az_err);

        results.elevation_error_rad.push_back(est_azel.elevation - true_azel.elevation);

        // Verbose output
        if (cfg.verbose && step % 10 == 0) {
            double t_elapsed = (jd - cfg.start_jd) * 86400.0;
            std::printf("t=%.1fs: True Az/El=%.1f/%.1f° Est Az/El=%.1f/%.1f° PosErr=%.1fm %s\n",
                t_elapsed,
                true_azel.azimuth * RAD2DEG, true_azel.elevation * RAD2DEG,
                est_azel.azimuth * RAD2DEG, est_azel.elevation * RAD2DEG,
                pos_err,
                meas.valid ? "[MEAS]" : "[PRED]");
        }

        jd += dt_jd;
        ++step;
    }

    results.compute_statistics();
    return results;
}

/**
 * @brief Print simulation summary
 */
inline void print_results(const SimulationResults& results) {
    using namespace constants;

    std::printf("\n========== SIMULATION RESULTS ==========\n");
    std::printf("Total measurements: %d\n", results.num_measurements);
    std::printf("Valid measurements: %d (%.1f%%)\n",
        results.num_valid_measurements,
        100.0 * results.num_valid_measurements / results.num_measurements);
    std::printf("\nPosition Errors:\n");
    std::printf("  Mean: %.2f m\n", results.mean_position_error_m);
    std::printf("  RMS:  %.2f m\n", results.rms_position_error_m);
    std::printf("\nAngle Errors:\n");
    std::printf("  Azimuth (mean):   %.3f deg\n", results.mean_az_error_deg);
    std::printf("  Elevation (mean): %.3f deg\n", results.mean_el_error_deg);

    // Find best and worst errors
    auto minmax_pos = std::minmax_element(results.position_error_m.begin(),
        results.position_error_m.end());
    std::printf("\nPosition error range: %.2f - %.2f m\n",
        *minmax_pos.first, *minmax_pos.second);

    // Final state
    if (!results.estimated_positions.empty()) {
        const auto& final_pos = results.estimated_positions.back();
        const auto& final_unc = results.position_uncertainty.back();
        std::printf("\nFinal estimated position:\n");
        std::printf("  Lat: %.4f° ± %.1f m\n", final_pos[0] * RAD2DEG, final_unc[0]);
        std::printf("  Lon: %.4f° ± %.1f m\n", final_pos[1] * RAD2DEG, final_unc[1]);
        std::printf("  Alt: %.1f km ± %.1f m\n", final_pos[2] / 1000.0, final_unc[2]);
    }
    std::printf("=========================================\n");
}

} // namespace tracking
} // namespace optmath
