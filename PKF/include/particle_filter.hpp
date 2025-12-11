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
    ParticleFilter(const Model* model, size_t num_particles, double resampling_threshold = 0.5)
        : model_(model), N_(num_particles), resampling_threshold_(resampling_threshold * static_cast<double>(num_particles)) {

        particles_.resize(N_);
        log_weights_.resize(N_, -std::log(static_cast<double>(N_))); // Initialize with uniform weights

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
        std::fill(log_weights_.begin(), log_weights_.end(), -std::log(static_cast<double>(N_)));
    }

    /**
     * @brief Perform one step of the particle filter
     *
     * 1. Propagate particles
     * 2. Update weights based on observation
     * 3. Normalize weights
     *
     * @param y_k Observation
     * @param t_k Current time
     * @param u_k Control input
     */
    void step(const Observation& y_k, double t_k, const Eigen::Ref<const State>& u_k) {
        // 1. Propagation
        for (size_t i = 0; i < N_; ++i) {
            State deterministic_x = model_->propagate(particles_[i], t_k, u_k);
            State process_noise = model_->sample_process_noise(t_k, rng_);
            particles_[i] = deterministic_x + process_noise;
        }

        // 2. Weight Update
        for (size_t i = 0; i < N_; ++i) {
            double log_lik = model_->observation_loglik(y_k, particles_[i], t_k);
            log_weights_[i] += log_lik;
        }

        // 3. Normalization
        normalize_weights();
    }

    /**
     * @brief Calculate Effective Sample Size (ESS)
     */
    double get_effective_sample_size() const {
        // ESS = 1 / sum(w^2)
        // Need linear normalized weights
        // Use pre-computed weights from normalize_weights step if available, or recompute.
        // For efficiency, we can compute ESS during normalization or on demand.
        // Here we compute on demand using log_weights which are already normalized in log domain?
        // No, normalize_weights() normalizes them.

        double sum_sq = 0.0;
        for (double lw : log_weights_) {
            sum_sq += std::exp(2.0 * lw);
        }
        return 1.0 / sum_sq;
    }

    /**
     * @brief Resample particles if ESS is below threshold
     *
     * @return std::vector<size_t> Parent indices (if resampled), or empty/identity if not.
     */
    std::vector<size_t> resample_if_needed() {
        double ess = get_effective_sample_size();

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
            std::fill(log_weights_.begin(), log_weights_.end(), -std::log(static_cast<double>(N_)));

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
    double resampling_threshold_;
    std::vector<State> particles_;
    std::vector<double> log_weights_;
    std::mt19937_64 rng_;

    /**
     * @brief Normalize log-weights using log-sum-exp trick
     */
    void normalize_weights() {
        double max_log_w = -std::numeric_limits<double>::infinity();
        for (double w : log_weights_) {
            if (w > max_log_w) max_log_w = w;
        }

        double sum_exp = 0.0;
        for (double w : log_weights_) {
            sum_exp += std::exp(w - max_log_w);
        }

        double log_sum = max_log_w + std::log(sum_exp);

        for (double& w : log_weights_) {
            w -= log_sum;
        }
    }
};

} // namespace PKF

#endif // PKF_PARTICLE_FILTER_HPP
