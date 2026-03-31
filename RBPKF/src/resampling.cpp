#include "rbpf/resampling.hpp"
#include <numeric>
#include <cmath>

namespace rbpf {

std::vector<int> systematic_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    if (N == 0) return std::vector<int>();
    if (N == 1) return std::vector<int>{0};

    std::vector<int> parents(N);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f / static_cast<float>(N));
    float u0 = dist(rng);

    // Use Kahan summation for cumulative sum to reduce float precision loss
    float csum = 0.0f;
    float comp = 0.0f;  // Kahan compensation
    {
        float y = weights[0] - comp;
        float t = csum + y;
        comp = (t - csum) - y;
        csum = t;
    }
    size_t k = 0;

    for (size_t i = 0; i < N; ++i) {
        float u = u0 + static_cast<float>(i) / static_cast<float>(N);

        while (u > csum && k < N - 1) {
            k++;
            float y = weights[k] - comp;
            float t = csum + y;
            comp = (t - csum) - y;
            csum = t;
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

std::vector<int> stratified_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    if (N == 0) return std::vector<int>();
    if (N == 1) return std::vector<int>{0};

    std::vector<int> parents(N);

    // Use Kahan summation for cumulative sum
    float csum = 0.0f;
    float comp = 0.0f;
    {
        float y = weights[0] - comp;
        float t = csum + y;
        comp = (t - csum) - y;
        csum = t;
    }
    size_t k = 0;

    // Create distribution once, not per iteration
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        float u = (static_cast<float>(i) + dist(rng)) / static_cast<float>(N);

        while (u > csum && k < N - 1) {
            k++;
            float y = weights[k] - comp;
            float t = csum + y;
            comp = (t - csum) - y;
            csum = t;
        }
        parents[i] = static_cast<int>(k);
    }

    return parents;
}

} // namespace rbpf
