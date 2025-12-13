#include "rbpf/resampling.hpp"
#include <numeric>
#include <cmath>
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>

namespace rbpf {

// Helper to compute inclusive CDF on GPU
inline std::vector<float> compute_cdf_vulkan(const std::vector<float>& weights) {
    if (!optmath::vulkan::is_available()) return {};

    size_t N = weights.size();
    if (N > 256) return {};

    Eigen::Map<const Eigen::VectorXf> w_map(weights.data(), N);

    // Exclusive
    Eigen::VectorXf exclusive = optmath::vulkan::vulkan_scan_prefix_sum(w_map);
    // Inclusive
    Eigen::VectorXf inclusive = optmath::vulkan::vulkan_vec_add(exclusive, w_map);

    std::vector<float> cdf(N);
    Eigen::Map<Eigen::VectorXf>(cdf.data(), N) = inclusive;
    return cdf;
}

std::vector<int> systematic_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f / static_cast<float>(N));
    float u0 = dist(rng);

    std::vector<float> cdf = (N <= 256) ? compute_cdf_vulkan(weights) : std::vector<float>{};

    if (!cdf.empty()) {
        size_t k = 0;
        for (size_t i = 0; i < N; ++i) {
            float u = u0 + static_cast<float>(i) / static_cast<float>(N);
            while (k < N - 1 && u > cdf[k]) {
                k++;
            }
            parents[i] = static_cast<int>(k);
        }
    } else {
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
    }

    return parents;
}

std::vector<int> stratified_resampling(const std::vector<float>& weights, std::mt19937_64& rng) {
    size_t N = weights.size();
    std::vector<int> parents(N);

    std::vector<float> cdf = (N <= 256) ? compute_cdf_vulkan(weights) : std::vector<float>{};

    if (!cdf.empty()) {
        size_t k = 0;
        for (size_t i = 0; i < N; ++i) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float u = (static_cast<float>(i) + dist(rng)) / static_cast<float>(N);
            while (k < N - 1 && u > cdf[k]) {
                k++;
            }
            parents[i] = static_cast<int>(k);
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
            parents[i] = static_cast<int>(k);
        }
    }

    return parents;
}

} // namespace rbpf
