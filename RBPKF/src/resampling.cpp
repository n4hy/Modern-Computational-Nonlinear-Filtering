#include "rbpf/resampling.hpp"
#include <numeric>
#include <cmath>

namespace rbpf {

/**
 * Systematic resampling: draw N particles using a single uniform random offset
 * and N evenly spaced points through the CDF. Offsets and the cumulative weight
 * sum are accumulated in double.
 */
std::vector<int> systematic_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    if (N == 0) return std::vector<int>();
    if (N == 1) return std::vector<int>{0};

    std::vector<int> parents(N);

    // Offsets and the running CDF are double. The previous float u (and a
    // Kahan-compensated but float-STORED csum) has a spacing of ~6e-8 near 1,
    // within a few ulps of the 1/N stratum width at N ~ 1e6 and coarser than it
    // beyond ~1e7, so strata collapse onto the same parent.
    std::uniform_real_distribution<double> dist(0.0, 1.0 / static_cast<double>(N));
    const double u0 = dist(rng);

    double csum = weights[0];
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        const double u = u0 + static_cast<double>(i) / static_cast<double>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

/**
 * Stratified resampling: divide [0,1] into N strata, draw one uniform sample
 * per stratum, then map through the CDF. Lower variance than systematic.
 * Offsets and the cumulative sum are accumulated in double.
 */
std::vector<int> stratified_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    if (N == 0) return std::vector<int>();
    if (N == 1) return std::vector<int>{0};

    std::vector<int> parents(N);

    // Double offsets and running CDF -- see systematic_resampling above.
    double csum = weights[0];
    size_t k = 0;

    // Create distribution once, not per iteration
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        const double u = (static_cast<double>(i) + dist(rng)) / static_cast<double>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

} // namespace rbpf
