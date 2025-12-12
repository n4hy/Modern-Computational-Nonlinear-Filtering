#ifndef RBPF_RESAMPLING_HPP
#define RBPF_RESAMPLING_HPP

#include <vector>
#include <random>

namespace rbpf {

/**
 * @brief Systematic Resampling
 *
 * @param weights Normalized linear weights (sum to 1)
 * @param rng Random number generator
 * @return std::vector<int> Indices of selected particles (parents)
 */
std::vector<int> systematic_resampling(const std::vector<float>& weights, std::mt19937_64& rng);

/**
 * @brief Stratified Resampling
 *
 * @param weights Normalized linear weights (sum to 1)
 * @param rng Random number generator
 * @return std::vector<int> Indices of selected particles (parents)
 */
std::vector<int> stratified_resampling(const std::vector<float>& weights, std::mt19937_64& rng);

} // namespace rbpf

#endif // RBPF_RESAMPLING_HPP
