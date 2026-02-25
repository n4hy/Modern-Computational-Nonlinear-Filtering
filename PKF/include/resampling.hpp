#ifndef PKF_RESAMPLING_HPP
#define PKF_RESAMPLING_HPP

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace PKF {

namespace Resampling {

    /**
     * @brief Systematic Resampling
     *
     * Uses a single random number to generate N evenly spaced points.
     * Complexity: O(N)
     *
     * @param weights Normalized linear weights (sum to 1)
     * @param rng Random number generator
     * @return std::vector<size_t> Indices of selected particles (parents)
     */
    inline std::vector<size_t> systematic(const std::vector<double>& weights, std::mt19937_64& rng) {
        size_t N = weights.size();
        if (N == 0) return std::vector<size_t>();
        if (N == 1) return std::vector<size_t>{0};

        std::vector<size_t> parents(N);

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
            parents[i] = k;
        }

        return parents;
    }

    // Overload for float weights
    inline std::vector<size_t> systematic(const std::vector<float>& weights, std::mt19937_64& rng) {
        size_t N = weights.size();
        if (N == 0) return std::vector<size_t>();
        if (N == 1) return std::vector<size_t>{0};

        std::vector<size_t> parents(N);

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
            parents[i] = k;
        }

        return parents;
    }

    /**
     * @brief Stratified Resampling
     *
     * Divides the range [0,1] into N strata and samples one point from each.
     * Lower variance than multinomial, often better than systematic.
     * Complexity: O(N)
     *
     * @param weights Normalized linear weights (sum to 1)
     * @param rng Random number generator
     * @return std::vector<size_t> Indices of selected particles (parents)
     */
    inline std::vector<size_t> stratified(const std::vector<double>& weights, std::mt19937_64& rng) {
        size_t N = weights.size();
        if (N == 0) return std::vector<size_t>();
        if (N == 1) return std::vector<size_t>{0};

        std::vector<size_t> parents(N);

        double csum = weights[0];
        size_t k = 0;

        for (size_t i = 0; i < N; ++i) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double u = (static_cast<double>(i) + dist(rng)) / static_cast<double>(N);

            while (u > csum && k < N - 1) {
                k++;
                csum += weights[k];
            }
            parents[i] = k;
        }

        return parents;
    }

    // Overload for float weights
    inline std::vector<size_t> stratified(const std::vector<float>& weights, std::mt19937_64& rng) {
        size_t N = weights.size();
        if (N == 0) return std::vector<size_t>();
        if (N == 1) return std::vector<size_t>{0};

        std::vector<size_t> parents(N);

        float csum = weights[0];
        size_t k = 0;

        for (size_t i = 0; i < N; ++i) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float u = (static_cast<float>(i) + dist(rng)) / static_cast<float>(N);

            while (u > csum && k < N - 1) {
                k++;
                csum += weights[k];
            }
            parents[i] = k;
        }

        return parents;
    }

} // namespace Resampling
} // namespace PKF

#endif // PKF_RESAMPLING_HPP
