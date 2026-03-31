#ifndef PKF_PARTICLE_FILTER_HPP
#define PKF_PARTICLE_FILTER_HPP

#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "state_space_model.hpp"
#include "resampling.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <optmath/vulkan_backend.hpp>
#define PKF_HAS_VULKAN 1
#else
#define PKF_HAS_VULKAN 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PKF {

/**
 * @class ParticleFilter
 * @brief Bootstrap Particle Filter implementation.
 *
 * @tparam NX State dimension
 * @tparam NY Observation dimension
 */
template<int NX, int NY>
class ParticleFilter {
public:
    using Model = StateSpaceModel<NX, NY>;
    using State = typename Model::State;
    using StateMat = typename Model::StateMat;
    using Observation = typename Model::Observation;

    /**
     * @brief Constructor
     *
     * @param model Pointer to the state space model
     * @param num_particles Number of particles
     * @param resampling_threshold Threshold for resampling (fraction of N)
     */
    ParticleFilter(const Model* model, size_t num_particles, float resampling_threshold = 0.5f)
        : model_(model), N_(num_particles), resampling_threshold_(resampling_threshold * static_cast<float>(num_particles)) {

        particles_.resize(N_);
        log_weights_.resize(N_, -std::log(static_cast<float>(N_))); // Initialize with uniform weights

        // Pre-allocate temporary vectors for step()
        props_.resize(N_);
        noises_.resize(N_);

        // Seed RNG
        std::random_device rd;
        rng_.seed(rd());
    }

    // Allow setting fixed seed
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

    /**
     * @brief Initialize particles using a prior distribution sampler
     *
     * @tparam Sampler Function object: State sampler(std::mt19937_64&)
     */
    template<typename Sampler>
    void initialize(Sampler&& prior_sampler) {
        for (size_t i = 0; i < N_; ++i) {
            particles_[i] = prior_sampler(rng_);
        }
        std::fill(log_weights_.begin(), log_weights_.end(), -std::log(static_cast<float>(N_)));
    }

    /**
     * @brief Perform one step of the particle filter
     */
    void step(const Observation& y_k, float t_k, const Eigen::Ref<const State>& u_k) {

        // 1. Propagation
        // We compute deterministic part and noise separately to allow potential batching
        // Use pre-allocated props_ and noises_ vectors

#ifdef _OPENMP
        // Parallel propagation with thread-local RNG
        #pragma omp parallel
        {
            thread_local std::mt19937_64 local_rng{std::random_device{}()};
            #pragma omp for
            for (size_t i = 0; i < N_; ++i) {
                props_[i] = model_->propagate(particles_[i], t_k, u_k);
                noises_[i] = model_->sample_process_noise(t_k, local_rng);
            }
        }
#else
        for (size_t i = 0; i < N_; ++i) {
            props_[i] = model_->propagate(particles_[i], t_k, u_k);
            noises_[i] = model_->sample_process_noise(t_k, rng_);
        }
#endif

        // Vulkan Acceleration for Noise Addition
#if PKF_HAS_VULKAN
        if (N_ > 100 && optmath::vulkan::is_available()) {
             // Create large vectors
             Eigen::VectorXf flat_props(N_ * NX);
             Eigen::VectorXf flat_noises(N_ * NX);

             for (size_t i = 0; i < N_; ++i) {
                 flat_props.segment<NX>(i * NX) = props_[i];
                 flat_noises.segment<NX>(i * NX) = noises_[i];
             }

             // Run on GPU
             Eigen::VectorXf flat_result = optmath::vulkan::vulkan_vec_add(flat_props, flat_noises);

             // Copy back
             for (size_t i = 0; i < N_; ++i) {
                 particles_[i] = flat_result.segment<NX>(i * NX);
             }
        } else
#endif
        {
            // CPU fallback (also used on non-Vulkan platforms)
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < N_; ++i) {
                particles_[i] = props_[i] + noises_[i];
            }
        }

        // 2. Weight Update
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < N_; ++i) {
            float log_lik = model_->observation_loglik(y_k, particles_[i], t_k);
            log_weights_[i] += log_lik;
        }

        // 3. Normalization
        normalize_weights();
    }

    /**
     * @brief Calculate Effective Sample Size (ESS)
     *
     * Uses log-sum-exp trick to prevent underflow when log-weights are very negative.
     */
    float get_effective_sample_size() const {
        if (log_weights_.empty()) return 0.0f;

        // Find max of 2*log_weights to use log-sum-exp trick
        double max_2lw = 2.0 * *std::max_element(log_weights_.begin(), log_weights_.end());

        double sum_exp = 0.0;
        for (double lw : log_weights_) {
            sum_exp += std::exp(2.0 * lw - max_2lw);
        }

        // log(sum(w^2)) = max_2lw + log(sum_exp)
        double log_sum_sq = max_2lw + std::log(sum_exp);

        // ESS = 1 / sum(w^2), clamped to valid range [1, N]
        return static_cast<float>(std::clamp(std::exp(-log_sum_sq), 1.0, static_cast<double>(N_)));
    }

    /**
     * @brief Resample particles if ESS is below threshold
     *
     * @return std::vector<size_t> Parent indices (if resampled), or empty/identity if not.
     */
    std::vector<size_t> resample_if_needed() {
        float ess = get_effective_sample_size();

        if (ess < resampling_threshold_) {
            // Convert log weights to linear weights for resampling
            std::vector<double> weights(N_);
            for (size_t i = 0; i < N_; ++i) {
                weights[i] = std::exp(log_weights_[i]);
            }

            // Perform resampling (Stratified by default)
            std::vector<size_t> parents = Resampling::stratified(weights, rng_);

            // Create new particle set
            std::vector<State> new_particles(N_);
            for (size_t i = 0; i < N_; ++i) {
                new_particles[i] = particles_[parents[i]];
            }
            particles_ = std::move(new_particles);

            // Reset weights to uniform
            std::fill(log_weights_.begin(), log_weights_.end(), -std::log(static_cast<float>(N_)));

            return parents;
        }

        // If no resampling, parents are i -> i
        std::vector<size_t> parents(N_);
        std::iota(parents.begin(), parents.end(), 0);
        return parents;
    }

    /**
     * @brief Get filtered mean estimate
     */
    State get_mean() const {
        State mean = State::Zero();
        for (size_t i = 0; i < N_; ++i) {
            mean += std::exp(log_weights_[i]) * particles_[i];
        }
        return mean;
    }

    /**
     * @brief Get filtered covariance estimate
     */
    StateMat get_covariance() const {
        State mean = get_mean();
        StateMat cov = StateMat::Zero();
        for (size_t i = 0; i < N_; ++i) {
            State diff = particles_[i] - mean;
            cov += std::exp(log_weights_[i]) * (diff * diff.transpose());
        }
        return cov;
    }

    // Accessors
    const std::vector<State>& get_particles() const { return particles_; }
    const std::vector<double>& get_log_weights() const { return log_weights_; }

private:
    const Model* model_;
    size_t N_;
    float resampling_threshold_;
    std::vector<State> particles_;
    std::vector<double> log_weights_;
    std::mt19937_64 rng_;

    // Pre-allocated temporary vectors for step()
    std::vector<State> props_;
    std::vector<State> noises_;

    /**
     * @brief Normalize log-weights using log-sum-exp trick
     */
    void normalize_weights() {
        double max_log_w = -std::numeric_limits<double>::infinity();
        for (double w : log_weights_) {
            if (std::isfinite(w) && w > max_log_w) max_log_w = w;
        }

        // Handle degenerate case: all particles dead (weights = -inf)
        if (!std::isfinite(max_log_w)) {
            // Reset to uniform weights
            double uniform_log_w = -std::log(static_cast<double>(N_));
            for (double& w : log_weights_) {
                w = uniform_log_w;
            }
            return;
        }

        double sum_exp = 0.0;
        for (double w : log_weights_) {
            if (std::isfinite(w)) {
                sum_exp += std::exp(w - max_log_w);
            }
        }

        // Prevent log(0)
        if (sum_exp <= 0.0) {
            double uniform_log_w = -std::log(static_cast<double>(N_));
            for (double& w : log_weights_) {
                w = uniform_log_w;
            }
            return;
        }

        double log_sum = max_log_w + std::log(sum_exp);

        for (double& w : log_weights_) {
            w -= log_sum;
        }
    }
};

} // namespace PKF

#endif // PKF_PARTICLE_FILTER_HPP
