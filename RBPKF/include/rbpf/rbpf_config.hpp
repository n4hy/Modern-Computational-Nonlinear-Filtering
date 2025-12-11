#ifndef RBPF_CONFIG_HPP
#define RBPF_CONFIG_HPP

namespace rbpf {

/**
 * @brief Configuration parameters for the Rao-Blackwellized Particle Filter.
 */
struct RbpfConfig {
    int    num_particles = 100;
    double resampling_threshold = 0.5;  // Resample if N_eff / N < threshold
    bool   use_systematic_resampling = true; // true for systematic, false for stratified
    int    fixed_lag = 0;             // L for fixed-lag smoothing (0 = filter only)
    unsigned long long seed = 123456789;      // RNG seed
};

} // namespace rbpf

#endif // RBPF_CONFIG_HPP
