#include "rbpf/resampling.hpp"
#include <numeric>
#include <cmath>

namespace rbpf {

std::vector<int> systematic_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f / static_cast<float>(N));
    float u0 = dist(rng);

    float csum = weights[0];
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        float u = u0 + static_cast<float>(i) / static_cast<float>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

std::vector<int> stratified_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    float csum = weights[0];
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float u = (static_cast<float>(i) + dist(rng)) / static_cast<float>(N);

        while (u > csum && k < N - 1) {
            k++;
            csum += weights[k];
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

} // namespace rbpf
