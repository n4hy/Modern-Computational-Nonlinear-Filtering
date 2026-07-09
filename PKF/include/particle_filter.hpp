#ifndef PKF_PARTICLE_FILTER_HPP
#define PKF_PARTICLE_FILTER_HPP

#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <memory>
#include "state_space_model.hpp"
#include "resampling.hpp"
#include "particle_filter_gpu.hpp"

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
     * @param enable_gpu Enable GPU acceleration if available (default: true)
     */
    ParticleFilter(const Model* model, size_t num_particles,
                   float resampling_threshold = 0.5f, bool enable_gpu = true)
        : model_(model), N_(num_particles),
          resampling_threshold_(resampling_threshold * static_cast<float>(num_particles)),
          use_gpu_(enable_gpu && gpu::should_use_gpu_particles(num_particles))
    {
        particles_.resize(N_);
        log_weights_.resize(N_, -std::log(static_cast<float>(N_))); // Initialize with uniform weights

        // Pre-allocate temporary vectors for step()
        props_.resize(N_);
        noises_.resize(N_);

        // Initialize GPU context if enabled
        if (use_gpu_) {
            gpu_ctx_ = std::make_unique<gpu::GPUParticleContext<NX>>(N_);
            if (!gpu_ctx_->is_active()) {
                use_gpu_ = false;
                gpu_ctx_.reset();
            }
        }

        // Seed RNG. Derive a 64-bit base seed so both the serial rng_ and the
        // per-thread RNGs come from the same source.
        std::random_device rd;
        base_seed_ = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        rng_.seed(base_seed_);
        seed_thread_rngs();
    }

    /** Set a fixed RNG seed for reproducible results (serial AND OpenMP paths). */
    void set_seed(uint64_t seed) {
        base_seed_ = seed;
        rng_.seed(seed);
        seed_thread_rngs();
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
        // Parallel propagation with per-thread RNGs seeded deterministically from
        // base_seed_ (see seed_thread_rngs). schedule(static) fixes the
        // iteration->thread mapping so a given seed + thread count is reproducible.
        // (Previously a std::random_device thread_local was used, ignoring set_seed.)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N_; ++i) {
            std::mt19937_64& local_rng = thread_rngs_[static_cast<size_t>(omp_get_thread_num())];
            props_[i] = model_->propagate(particles_[i], t_k, u_k);
            noises_[i] = model_->sample_process_noise(t_k, local_rng);
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
            // GPU resampling path
            if (use_gpu_ && gpu_ctx_ && gpu_ctx_->is_active()) {
                sync_to_gpu();
                std::vector<size_t> parents = gpu_ctx_->resample_stratified_gpu(rng_);
                sync_from_gpu();  // Get resampled particles back
                return parents;
            }

            // CPU path
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
        // GPU path
        if (use_gpu_ && gpu_ctx_ && gpu_ctx_->is_active()) {
            // Const cast needed for GPU upload (doesn't modify)
            const_cast<ParticleFilter*>(this)->sync_to_gpu();
            return const_cast<gpu::GPUParticleContext<NX>*>(gpu_ctx_.get())->compute_mean_gpu();
        }

        // CPU path
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

        // GPU path
        if (use_gpu_ && gpu_ctx_ && gpu_ctx_->is_active()) {
            return const_cast<gpu::GPUParticleContext<NX>*>(gpu_ctx_.get())->compute_covariance_gpu(mean);
        }

        // CPU path
        StateMat cov = StateMat::Zero();
        for (size_t i = 0; i < N_; ++i) {
            State diff = particles_[i] - mean;
            cov += std::exp(log_weights_[i]) * (diff * diff.transpose());
        }
        return cov;
    }

    /**
     * @brief Check if GPU acceleration is active.
     */
    bool is_gpu_active() const { return use_gpu_ && gpu_ctx_ && gpu_ctx_->is_active(); }

    /**
     * @brief Sync particles to GPU (call before GPU operations).
     */
    void sync_to_gpu() {
        if (gpu_ctx_ && gpu_ctx_->is_active()) {
            gpu_ctx_->upload_particles(particles_);
            gpu_ctx_->upload_log_weights(log_weights_);
        }
    }

    /**
     * @brief Sync particles from GPU (call after GPU operations).
     */
    void sync_from_gpu() {
        if (gpu_ctx_ && gpu_ctx_->is_active()) {
            gpu_ctx_->download_particles(particles_);
            gpu_ctx_->download_log_weights(log_weights_);
        }
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
    uint64_t base_seed_ = 0;

#ifdef _OPENMP
    // One RNG per OpenMP thread, seeded deterministically from base_seed_ so that
    // set_seed() makes the parallel propagation reproducible (it previously used a
    // std::random_device thread_local that ignored the seed entirely).
    std::vector<std::mt19937_64> thread_rngs_;
#endif

    /** Seed one mt19937_64 per OpenMP thread deterministically from base_seed_. */
    void seed_thread_rngs() {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        thread_rngs_.resize(static_cast<size_t>(nthreads));
        for (int t = 0; t < nthreads; ++t) {
            uint64_t s = base_seed_ + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(t + 1);
            s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
            s ^= s >> 27; s *= 0x94D049BB133111EBULL;
            s ^= s >> 31;
            thread_rngs_[static_cast<size_t>(t)].seed(s);
        }
#endif
    }

    // Pre-allocated temporary vectors for step()
    std::vector<State> props_;
    std::vector<State> noises_;

    // GPU acceleration
    bool use_gpu_ = false;
    std::unique_ptr<gpu::GPUParticleContext<NX>> gpu_ctx_;

    /**
     * @brief Normalize log-weights using the log-sum-exp trick for numerical stability.
     *
     * After normalization, sum(exp(log_weights)) == 1. Handles degenerate cases
     * where all weights are -inf (particle death) by resetting to uniform.
     * Dispatches to GPU when the GPU context is active.
     */
    void normalize_weights() {
        // GPU path - normalize on GPU and sync back
        if (use_gpu_ && gpu_ctx_ && gpu_ctx_->is_active()) {
            gpu_ctx_->upload_log_weights(log_weights_);
            gpu_ctx_->normalize_weights_gpu();
            gpu_ctx_->download_log_weights(log_weights_);
            return;
        }

        // CPU path
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
