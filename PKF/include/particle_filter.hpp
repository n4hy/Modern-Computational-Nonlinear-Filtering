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
     *
     * CPU only (OpenMP across particles). The former CUDA "GPU context" and the
     * aarch64 Vulkan noise-add were removed on 2026-11-16: neither ran the
     * model on the device (propagate/observation_loglik are virtual host
     * calls), so each was host loops plus PCIe/Vulkan copies, and the CUDA
     * path was slower than this one at every N measured (N = 200k: 427 vs
     * 226 ms per 50 steps).
     */
    ParticleFilter(const Model* model, size_t num_particles,
                   float resampling_threshold = 0.5f)
        : model_(model), N_(num_particles),
          resampling_threshold_(resampling_threshold * static_cast<float>(num_particles))
    {
        particles_.resize(N_);
        log_weights_.resize(N_, -std::log(static_cast<float>(N_))); // Initialize with uniform weights

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
        step_fused(y_k, t_k, u_k);
        normalize_weights();
    }

    /**
     * Below this many particles every per-particle loop runs serially. Measured
     * on a 24-core x86 host (libgomp, default passive wait): at N = 1000 one
     * thread took 2.1 ms / 50 steps against 3.2-37 ms for 8-24 threads -- the
     * cost is waking sleeping workers, not the work.
     */
    static constexpr size_t kParallelMinParticles = 2048;

    /**
     * The mean/covariance sums do far less work per particle than a step, so
     * their crossover is higher: parallel was slower at N = 4000 (cov 2.1 ->
     * 4.4 ms / 50 calls) and faster from N = 16000 (cov 9.6 -> 4.1 ms).
     */
    static constexpr size_t kParallelMinMoments = 16384;

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
            // Double-precision weights and CDF: with float log-weights at
            // N = 4e6 the CDF boundaries shift by several strata (equal
            // weights: 711k of 4M particles got no offspring; this path: 197).
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

    /** @brief Filtered (weighted) mean. */
    State get_mean() const { return compute_mean_cpu(); }

    /** @brief Filtered (weighted) covariance about the mean. */
    StateMat get_covariance() const { return compute_covariance_cpu(compute_mean_cpu()); }

    /** Weighted mean (same as get_mean(); kept for existing callers). */
    State compute_mean_cpu() const {
        return parallel_weighted_sum<State>([&](size_t i) {
            return State(std::exp(log_weights_[i]) * particles_[i]);
        });
    }

    /** Weighted covariance about a caller-supplied mean. */
    StateMat compute_covariance_cpu(const State& mean) const {
        return parallel_weighted_sum<StateMat>([&](size_t i) {
            const State diff = particles_[i] - mean;
            return StateMat(std::exp(log_weights_[i]) * (diff * diff.transpose()));
        });
    }

    // Accessors
    const std::vector<State>& get_particles() const { return particles_; }
    const std::vector<double>& get_log_weights() const { return log_weights_; }

private:
    /**
     * Sum term(i) over all particles. Parallel (from kParallelMinMoments)
     * with per-thread partials added in thread order, so the result is
     * deterministic for a given thread count; it differs from the serial sum
     * only by float reassociation.
     */
    template<typename T, typename Term>
    T parallel_weighted_sum(Term term) const {
#ifdef _OPENMP
        if (N_ >= kParallelMinMoments) {
            const int nt = omp_get_max_threads();
            std::vector<T, Eigen::aligned_allocator<T>> partial(static_cast<size_t>(nt), T::Zero());
            #pragma omp parallel num_threads(nt)
            {
                T acc = T::Zero();
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < N_; ++i) acc += term(i);
                partial[static_cast<size_t>(omp_get_thread_num())] = acc;
            }
            T sum = T::Zero();
            for (const T& p : partial) sum += p;
            return sum;
        }
#endif
        T sum = T::Zero();
        for (size_t i = 0; i < N_; ++i) sum += term(i);
        return sum;
    }

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

    /**
     * Grow thread_rngs_ to the thread count the next parallel region will use.
     * It was sized once, from omp_get_max_threads() at construction/seeding, so
     * a caller raising the count later (omp_set_num_threads, OMP_DYNAMIC) made
     * thread_rngs_[omp_get_thread_num()] index past the end: out-of-bounds heap
     * writes, caught by -D_GLIBCXX_ASSERTIONS. Existing streams are left alone;
     * a new slot t gets the same deterministic seed seed_thread_rngs() would
     * have given it.
     */
    void ensure_thread_rngs() {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        const int have = static_cast<int>(thread_rngs_.size());
        if (nthreads <= have) return;
        thread_rngs_.resize(static_cast<size_t>(nthreads));
        for (int t = have; t < nthreads; ++t) {
            uint64_t s = base_seed_ + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(t + 1);
            s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
            s ^= s >> 27; s *= 0x94D049BB133111EBULL;
            s ^= s >> 31;
            thread_rngs_[static_cast<size_t>(t)].seed(s);
        }
#endif
    }

    /**
     * Propagate, add process noise and weight every particle in ONE parallel
     * region. The former staged path used three regions per step; with
     * libgomp's default passive wait each region re-wakes the workers, which
     * on the 24-core test host cost ~0.7 ms per step at any N. Under
     * schedule(static) the same thread draws the same particle's noise, so a
     * given seed and thread count reproduce exactly.
     */
    void step_fused(const Observation& y_k, float t_k, const Eigen::Ref<const State>& u_k) {
#ifdef _OPENMP
        ensure_thread_rngs();
        const bool par = N_ >= kParallelMinParticles;
        #pragma omp parallel for schedule(static) if (par)
        for (size_t i = 0; i < N_; ++i) {
            std::mt19937_64& local_rng = thread_rngs_[static_cast<size_t>(omp_get_thread_num())];
            const State x = model_->propagate(particles_[i], t_k, u_k)
                          + model_->sample_process_noise(t_k, local_rng);
            particles_[i] = x;
            log_weights_[i] += model_->observation_loglik(y_k, x, t_k);
        }
#else
        for (size_t i = 0; i < N_; ++i) {
            const State x = model_->propagate(particles_[i], t_k, u_k)
                          + model_->sample_process_noise(t_k, rng_);
            particles_[i] = x;
            log_weights_[i] += model_->observation_loglik(y_k, x, t_k);
        }
#endif
    }

    /**
     * @brief Normalize log-weights using the log-sum-exp trick for numerical stability.
     *
     * After normalization, sum(exp(log_weights)) == 1. Handles degenerate cases
     * where all weights are -inf (particle death) by resetting to uniform.
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
