/**
 * @file iridium_burst_demodulator.hpp
 * @brief Iridium Burst Demodulator Model with Preamble Correlation
 *
 * Models the signal processing chain for extracting AOA and Doppler
 * measurements from Iridium-Next burst transmissions using a two-antenna
 * coherent SDR receiver.
 *
 * Iridium Burst Structure (Simplex):
 * - Preamble: 64 symbols (known pattern for sync)
 * - Unique Word: 15 symbols (frame sync)
 * - Data: Variable length
 * - Guard time between bursts
 *
 * Doppler Extraction Methods:
 * 1. AFC Loop (COARSE): ~100 Hz accuracy, continuous tracking
 * 2. Preamble Correlation (FINE): ~10 Hz accuracy, burst-by-burst
 * 3. Full Burst Coherent (PRECISE): ~1 Hz accuracy, requires full demod
 *
 * @author OptMathKernels
 * @version 0.5.0
 */

#pragma once

#include "ukf_aoa_doppler_tracking.hpp"
#include <complex>
#include <deque>

namespace optmath {
namespace tracking {

//=============================================================================
// Iridium Signal Parameters
//=============================================================================

namespace iridium {
    // Modulation parameters
    constexpr double SYMBOL_RATE = 25000.0;          // 25 kBaud QPSK
    constexpr double SYMBOL_PERIOD = 1.0 / SYMBOL_RATE;

    // Burst structure (simplex downlink)
    constexpr int PREAMBLE_SYMBOLS = 64;
    constexpr int UNIQUE_WORD_SYMBOLS = 15;
    constexpr int MIN_DATA_SYMBOLS = 312;            // Minimum burst
    constexpr int MAX_DATA_SYMBOLS = 1872;           // Maximum burst

    // Timing
    constexpr double PREAMBLE_DURATION = PREAMBLE_SYMBOLS * SYMBOL_PERIOD;  // 2.56 ms
    constexpr double FRAME_DURATION = 90.0e-3;       // 90 ms frame period
    constexpr double BURST_DURATION = 8.28e-3;       // Typical burst length

    // Signal characteristics
    constexpr double CARRIER_FREQUENCY = 1626.0e6;   // L-band center
    constexpr double CHANNEL_BANDWIDTH = 41.667e3;   // Channel spacing
    constexpr double SIGNAL_BANDWIDTH = 25.0e3;      // Occupied bandwidth

    // Preamble pattern (simplified - alternating for good correlation)
    // Real Iridium uses specific Barker-like codes
    constexpr int PREAMBLE_PATTERN_LENGTH = 16;
}

//=============================================================================
// Complex Signal Types
//=============================================================================

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;

//=============================================================================
// Preamble Correlator for Doppler Estimation
//=============================================================================

/**
 * @brief Preamble correlation result
 */
struct PreambleCorrelationResult {
    double peak_magnitude;      // Correlation peak strength
    double peak_phase;          // Phase at correlation peak [rad]
    double frequency_offset;    // Estimated frequency offset [Hz]
    double timing_offset;       // Symbol timing offset [samples]
    double snr_estimate;        // Estimated SNR from correlation [dB]
    bool detection_valid;       // True if preamble detected
};

/**
 * @brief Preamble correlator for burst detection and Doppler estimation
 *
 * Uses correlation with known preamble pattern to:
 * 1. Detect burst presence
 * 2. Estimate carrier frequency offset (Doppler)
 * 3. Estimate symbol timing
 * 4. Estimate SNR
 */
class PreambleCorrelator {
public:
    struct Config {
        double sample_rate;           // SDR sample rate [Hz]
        double carrier_frequency;     // Nominal carrier [Hz]
        int samples_per_symbol;       // Oversampling factor
        double detection_threshold;   // Correlation threshold for detection
        int fft_size;                 // FFT size for fine frequency estimation

        static Config default_config() {
            Config cfg;
            cfg.sample_rate = 250000.0;  // 250 kSps (10x symbol rate)
            cfg.carrier_frequency = iridium::CARRIER_FREQUENCY;
            cfg.samples_per_symbol = 10;
            cfg.detection_threshold = 0.7;  // Normalized correlation threshold
            cfg.fft_size = 1024;
            return cfg;
        }
    };

    explicit PreambleCorrelator(const Config& config = Config::default_config())
        : config_(config)
    {
        generate_reference_preamble();
    }

    /**
     * @brief Correlate input samples with preamble reference
     *
     * @param samples Input complex baseband samples
     * @return Correlation result with Doppler estimate
     */
    PreambleCorrelationResult correlate(const ComplexVec& samples) const {
        PreambleCorrelationResult result{};
        result.detection_valid = false;

        if (samples.size() < reference_preamble_.size()) {
            return result;
        }

        // Sliding correlation
        size_t corr_length = samples.size() - reference_preamble_.size() + 1;
        double max_corr = 0.0;
        size_t max_idx = 0;
        Complex max_corr_complex{0.0, 0.0};

        for (size_t i = 0; i < corr_length; ++i) {
            Complex corr{0.0, 0.0};
            for (size_t j = 0; j < reference_preamble_.size(); ++j) {
                corr += samples[i + j] * std::conj(reference_preamble_[j]);
            }
            double mag = std::abs(corr);
            if (mag > max_corr) {
                max_corr = mag;
                max_idx = i;
                max_corr_complex = corr;
            }
        }

        // Normalize correlation
        double signal_power = 0.0;
        for (size_t j = 0; j < reference_preamble_.size(); ++j) {
            signal_power += std::norm(samples[max_idx + j]);
        }
        double ref_power = 0.0;
        for (const auto& s : reference_preamble_) {
            ref_power += std::norm(s);
        }

        double norm_corr = max_corr / std::sqrt(signal_power * ref_power);

        result.peak_magnitude = norm_corr;
        result.peak_phase = std::arg(max_corr_complex);
        result.timing_offset = static_cast<double>(max_idx);

        // Check detection threshold
        if (norm_corr < config_.detection_threshold) {
            return result;
        }

        result.detection_valid = true;

        // Fine frequency estimation using phase progression across preamble
        // Split preamble into two halves and measure phase difference
        size_t half_len = reference_preamble_.size() / 2;
        Complex corr_first{0.0, 0.0};
        Complex corr_second{0.0, 0.0};

        for (size_t j = 0; j < half_len; ++j) {
            corr_first += samples[max_idx + j] * std::conj(reference_preamble_[j]);
            corr_second += samples[max_idx + half_len + j] * std::conj(reference_preamble_[half_len + j]);
        }

        double phase_diff = std::arg(corr_second * std::conj(corr_first));
        double time_diff = half_len * config_.samples_per_symbol / config_.sample_rate;

        // Frequency offset = phase_diff / (2 * pi * time_diff)
        result.frequency_offset = phase_diff / (constants::TWO_PI * time_diff);

        // SNR estimation from correlation peak
        // SNR ≈ (peak^2) / (1 - peak^2) for normalized correlation
        double peak_sq = norm_corr * norm_corr;
        if (peak_sq < 0.99) {
            result.snr_estimate = 10.0 * std::log10(peak_sq / (1.0 - peak_sq));
        } else {
            result.snr_estimate = 30.0;  // Cap at 30 dB
        }

        return result;
    }

    /**
     * @brief Estimate Doppler accuracy based on SNR and preamble length
     *
     * Cramer-Rao bound for frequency estimation:
     * σ_f = sqrt(12) / (2π * T * sqrt(2 * N * SNR))
     * where T = observation time, N = number of samples
     */
    double estimate_doppler_accuracy(double snr_db) const {
        double snr_linear = std::pow(10.0, snr_db / 10.0);
        double T = iridium::PREAMBLE_DURATION;
        int N = iridium::PREAMBLE_SYMBOLS * config_.samples_per_symbol;

        double crb = std::sqrt(12.0) / (constants::TWO_PI * T * std::sqrt(2.0 * N * snr_linear));

        // Practical implementation is typically 2-3x worse than CRB
        return crb * 2.5;
    }

private:
    Config config_;
    ComplexVec reference_preamble_;

    void generate_reference_preamble() {
        // Generate reference preamble samples
        // Simplified: alternating +1/-1 pattern (real Iridium uses specific codes)
        int total_samples = iridium::PREAMBLE_SYMBOLS * config_.samples_per_symbol;
        reference_preamble_.resize(total_samples);

        for (int i = 0; i < total_samples; ++i) {
            int symbol_idx = i / config_.samples_per_symbol;
            // Alternating QPSK symbols
            double phase = (symbol_idx % 4) * constants::PI / 2.0;
            reference_preamble_[i] = std::polar(1.0, phase);
        }
    }
};

//=============================================================================
// Two-Antenna Coherent Receiver Model
//=============================================================================

/**
 * @brief Coherent dual-antenna receiver for AOA + Doppler extraction
 */
class CoherentDualAntennaReceiver {
public:
    struct Config {
        double baseline;              // Antenna separation [m]
        double baseline_orientation;  // Array orientation [rad] from North
        double sample_rate;           // SDR sample rate [Hz]
        double center_frequency;      // Tuned frequency [Hz]
        double phase_cal_error;       // Phase calibration error [rad]
        double gain_imbalance_db;     // Gain imbalance between antennas [dB]
        double noise_figure_db;       // Receiver noise figure [dB]

        static Config default_config() {
            Config cfg;
            cfg.baseline = 0.10;  // 10 cm
            cfg.baseline_orientation = 0.0;  // Points North
            cfg.sample_rate = 250000.0;
            cfg.center_frequency = iridium::CARRIER_FREQUENCY;
            cfg.phase_cal_error = 0.05;  // ~3 degrees
            cfg.gain_imbalance_db = 0.5;
            cfg.noise_figure_db = 3.0;
            return cfg;
        }
    };

    explicit CoherentDualAntennaReceiver(const Config& config, uint64_t seed = 0)
        : config_(config)
        , correlator_(PreambleCorrelator::Config::default_config())
        , rng_(seed ? seed : std::random_device{}())
        , phase_noise_(0.0, 0.1)  // Phase noise
        , thermal_noise_(0.0, 1.0)
    {
        wavelength_ = constants::SPEED_OF_LIGHT / config_.center_frequency;
    }

    /**
     * @brief Process burst and extract measurements
     *
     * Models the complete signal processing chain:
     * 1. Receive signals on both antennas
     * 2. Correlate with preamble to detect burst and estimate Doppler
     * 3. Measure phase difference for AOA
     * 4. Return combined measurement
     */
    AOADopplerMeasurement process_burst(
        double timestamp,
        const AzElCoord& true_azel,
        const ECICoord& sat_eci,
        const GeodeticCoord& observer,
        double gmst,
        double signal_power_dbm = -100.0)
    {
        AOADopplerMeasurement meas;
        meas.timestamp = timestamp;
        meas.valid = false;

        // Check visibility
        if (true_azel.elevation < 5.0 * constants::DEG2RAD) {
            return meas;
        }

        // Compute true Doppler
        DopplerMeasurementModel doppler_model(DopplerConfig::default_iridium());
        double true_doppler = doppler_model.compute_true_doppler(sat_eci, observer, gmst);

        // Simulate received signal and correlation
        // SNR depends on elevation (path loss, atmospheric effects)
        double path_loss_db = 20.0 * std::log10(true_azel.range / 1000.0) + 32.4 +
                              20.0 * std::log10(config_.center_frequency / 1e6);
        double atmospheric_loss_db = 0.5 / std::sin(true_azel.elevation);  // Simple model
        double received_snr_db = signal_power_dbm + 174.0 - 10.0 * std::log10(config_.sample_rate)
                                - config_.noise_figure_db - path_loss_db - atmospheric_loss_db;

        // Check if signal is detectable
        if (received_snr_db < 5.0) {  // Minimum 5 dB SNR for detection
            return meas;
        }

        meas.valid = true;
        meas.snr_db = received_snr_db;

        // Doppler estimation from preamble correlation
        double doppler_accuracy = correlator_.estimate_doppler_accuracy(received_snr_db);
        std::normal_distribution<double> doppler_noise(0.0, doppler_accuracy);
        meas.doppler = true_doppler + doppler_noise(rng_);
        meas.doppler_std = doppler_accuracy;
        meas.range_rate = -meas.doppler * wavelength_;
        meas.range_rate_std = doppler_accuracy * wavelength_;

        // AOA estimation from phase difference
        // True phase difference: φ = 2π * d * cos(θ) / λ
        // where θ is angle off array broadside
        double az_rel = true_azel.azimuth - config_.baseline_orientation;
        double direction_cosine = std::cos(true_azel.elevation) * std::cos(az_rel);
        double true_phase_diff = constants::TWO_PI * config_.baseline * direction_cosine / wavelength_;

        // Add phase measurement errors
        double phase_noise_std = 0.1 / std::sqrt(std::pow(10.0, received_snr_db / 10.0));
        std::normal_distribution<double> phase_dist(0.0, phase_noise_std);
        double measured_phase = true_phase_diff + phase_dist(rng_) + config_.phase_cal_error;

        // Convert phase back to azimuth
        double meas_dir_cos = measured_phase * wavelength_ / (constants::TWO_PI * config_.baseline);
        meas_dir_cos = std::clamp(meas_dir_cos, -1.0, 1.0);

        double cos_el = std::cos(true_azel.elevation);
        double cos_az_rel = meas_dir_cos / cos_el;
        cos_az_rel = std::clamp(cos_az_rel, -1.0, 1.0);
        double az_rel_meas = std::acos(cos_az_rel);

        // Resolve sign ambiguity (use true value in simulation)
        if (std::sin(az_rel) < 0) {
            az_rel_meas = -az_rel_meas;
        }

        meas.azimuth = az_rel_meas + config_.baseline_orientation;
        if (meas.azimuth < 0) meas.azimuth += constants::TWO_PI;
        if (meas.azimuth >= constants::TWO_PI) meas.azimuth -= constants::TWO_PI;

        meas.azimuth_std = phase_noise_std / (constants::TWO_PI * config_.baseline / wavelength_ * cos_el);

        // Elevation from amplitude comparison (simplified model)
        double el_noise_std = 0.1 / std::sqrt(std::pow(10.0, received_snr_db / 10.0));
        std::normal_distribution<double> el_dist(0.0, el_noise_std);
        meas.elevation = true_azel.elevation + el_dist(rng_);
        meas.elevation = std::clamp(meas.elevation, 0.0, constants::PI / 2.0);
        meas.elevation_std = el_noise_std;

        return meas;
    }

private:
    Config config_;
    PreambleCorrelator correlator_;
    double wavelength_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> phase_noise_;
    std::normal_distribution<double> thermal_noise_;
};

//=============================================================================
// Enhanced Burst Demodulator
//=============================================================================

/**
 * @brief Complete Iridium burst demodulator with AOA + Doppler extraction
 */
class IridiumBurstDemodulator {
public:
    struct Config {
        CoherentDualAntennaReceiver::Config receiver;
        DopplerConfig::AccuracyMode doppler_mode;
        bool enable_afc;              // Automatic frequency control
        bool enable_timing_recovery;  // Symbol timing recovery
        double signal_power_dbm;      // Expected signal power

        static Config default_config() {
            Config cfg;
            cfg.receiver = CoherentDualAntennaReceiver::Config::default_config();
            cfg.doppler_mode = DopplerConfig::AccuracyMode::FINE;
            cfg.enable_afc = true;
            cfg.enable_timing_recovery = true;
            cfg.signal_power_dbm = -95.0;  // Typical Iridium signal
            return cfg;
        }
    };

    explicit IridiumBurstDemodulator(const Config& config = Config::default_config(),
                                      uint64_t seed = 0)
        : config_(config)
        , receiver_(config.receiver, seed)
        , burst_model_()
    {}

    /**
     * @brief Process potential burst at given time
     */
    AOADopplerMeasurement process(double timestamp,
                                   const AzElCoord& true_azel,
                                   const ECICoord& sat_eci,
                                   const GeodeticCoord& observer,
                                   double gmst) {
        // Check if burst is active
        if (!burst_model_.is_burst_active(timestamp)) {
            AOADopplerMeasurement meas;
            meas.valid = false;
            return meas;
        }

        return receiver_.process_burst(timestamp, true_azel, sat_eci, observer, gmst,
                                       config_.signal_power_dbm);
    }

    /**
     * @brief Check if burst timing allows measurement
     */
    bool is_burst_active(double timestamp) const {
        return burst_model_.is_burst_active(timestamp);
    }

private:
    Config config_;
    CoherentDualAntennaReceiver receiver_;
    IridiumBurstModel burst_model_;
};

} // namespace tracking
} // namespace optmath
