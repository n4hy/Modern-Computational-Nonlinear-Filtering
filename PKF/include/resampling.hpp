#ifndef PKF_RESAMPLING_HPP
#define PKF_RESAMPLING_HPP

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <optmath/vulkan_backend.hpp>

namespace PKF {

namespace Resampling {

    // Helper to compute inclusive CDF on GPU if available
    inline std::vector<float> compute_cdf_vulkan(const std::vector<float>& weights) {
        if (!optmath::vulkan::is_available()) return {};

        size_t N = weights.size();
        // Current implementation of scan_prefix_sum in OptimizedKernels is limited to 256 elements
        if (N > 256) return {};

        // Map to Eigen
        Eigen::Map<const Eigen::VectorXf> w_map(weights.data(), N);

        // 1. Exclusive Scan
        Eigen::VectorXf exclusive = optmath::vulkan::vulkan_scan_prefix_sum(w_map);

        // 2. Inclusive Scan = Exclusive + Weights
        Eigen::VectorXf inclusive = optmath::vulkan::vulkan_vec_add(exclusive, w_map);

        // Copy back
        std::vector<float> cdf(N);
        Eigen::Map<Eigen::VectorXf>(cdf.data(), N) = inclusive;

        // Ensure last element is 1.0 (or sum) to avoid precision issues
        // (But we shouldn't modify unless we know it's normalized.
        //  The algorithm usually handles u > 1.0 logic by clamping or just loop ending.)
        return cdf;
    }

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
        std::vector<size_t> parents(N);

        std::uniform_real_distribution<float> dist(0.0f, 1.0f / static_cast<float>(N));
        float u0 = dist(rng);

        // Use GPU for CDF if available and size supported
        std::vector<float> cdf = (N <= 256) ? compute_cdf_vulkan(weights) : std::vector<float>{};

        if (!cdf.empty()) {
            // Search using CDF
            // u starts at u0.
            // We can use std::upper_bound or manual march. Manual march is O(N) which is fine.

            size_t k = 0;
            for (size_t i = 0; i < N; ++i) {
                float u = u0 + static_cast<float>(i) / static_cast<float>(N);

                // March
                while (k < N - 1 && u > cdf[k]) {
                    k++;
                }
                parents[i] = k;
            }
        } else {
            // CPU fallback
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
        std::vector<size_t> parents(N);

        std::vector<float> cdf = (N <= 256) ? compute_cdf_vulkan(weights) : std::vector<float>{};

        if (!cdf.empty()) {
             size_t k = 0;

             for (size_t i = 0; i < N; ++i) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float u = (static_cast<float>(i) + dist(rng)) / static_cast<float>(N);

                while (k < N - 1 && u > cdf[k]) {
                    k++;
                }
                parents[i] = k;
             }
        } else {
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
        }

        return parents;
    }

} // namespace Resampling
} // namespace PKF

#endif // PKF_RESAMPLING_HPP
