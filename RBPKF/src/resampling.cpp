#include "rbpf/resampling.hpp"
#include <numeric>
#include <cmath>

namespace rbpf {

std::vector<int> systematic_resampling(const std::vector<double>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    std::uniform_real_distribution<double> dist(0.0, 1.0 / static_cast<double>(N));
    double u0 = dist(rng);

    double csum = weights[0];
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        double u = u0 + static_cast<double>(i) / static_cast<double>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

std::vector<int> stratified_resampling(const std::vector<double>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    double csum = weights[0];
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = (static_cast<double>(i) + dist(rng)) / static_cast<double>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

} // namespace rbpf
