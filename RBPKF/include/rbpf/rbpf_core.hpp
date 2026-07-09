#ifndef RBPF_CORE_HPP
#define RBPF_CORE_HPP

#include "types.hpp"
#include "rbpf_config.hpp"
#include "state_space_models.hpp"
#include "kalman_filter.hpp"
#include "resampling.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <numbers>
#include "FilterMath.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rbpf {

template<typename Types>
struct RbpfParticle {
    using NonlinearState = typename Types::NonlinearState;
    using LinearState    = typename Types::LinearState;
    using LinearCov      = typename Types::LinearCov;

    NonlinearState    x_nl;
    LinearKalmanFilter<Types> kf;
    float             log_weight;
};

template<typename Types,
         typename NonlinearModelT,
         typename CondLinModelT>
class RaoBlackwellizedParticleFilter {
public:
    using NonlinearState = typename Types::NonlinearState;
    using LinearState    = typename Types::LinearState;
    using Observation    = typename Types::Observation;
    using LinearCov      = typename Types::LinearCov;
    using ObsCov         = typename Types::ObsCov;

    /**
     * Construct RBPF with nonlinear and conditional-linear models plus config.
     * Pre-allocates particle array and, if fixed_lag > 0, the ancestry buffers.
     */
    RaoBlackwellizedParticleFilter(const NonlinearModelT& nonlinear_model,
                                   const CondLinModelT& conditional_model,
                                   const RbpfConfig& config)
        : nonlinear_model_(nonlinear_model),
          conditional_model_(conditional_model),
          config_(config) {

        particles_.resize(config_.num_particles);
        rng_.seed(config_.seed);
        seed_thread_rngs();

        if (config_.fixed_lag > 0) {
            ancestry_buffer_size_ = config_.fixed_lag + 1;
            parent_indices_.resize(ancestry_buffer_size_);
            particle_history_.resize(ancestry_buffer_size_);
            for(auto& vec : parent_indices_) vec.resize(config_.num_particles);
            for(auto& vec : particle_history_) vec.resize(config_.num_particles);
        }
    }

    /** Initialize all particles to the same nonlinear/linear state with uniform weights. */
    void initialize(const NonlinearState& x_nl0,
                    const LinearState&    x_lin0,
                    const LinearCov&      P_lin0) {
        float init_log_weight = -std::log(static_cast<float>(config_.num_particles));

        // Reseed so a re-run after initialize() with the same config seed reproduces.
        rng_.seed(config_.seed);
        seed_thread_rngs();

        for (auto& p : particles_) {
            p.x_nl = x_nl0;
            p.kf.initialize(x_lin0, P_lin0);
            p.log_weight = init_log_weight;
        }

        if (config_.fixed_lag > 0) {
            // Record the initial state as logical time 0 and advance the counter so
            // the first step() stores at logical time 1 instead of overwriting slot 0.
            // Without this the initial snapshot is clobbered by step 1, and the
            // fixed-lag smoother can never reach the true initial state (off-by-one
            // in smoothing depth). parent_indices at slot 0 are the identity map.
            store_history(0);
            parent_indices_cnt_ = 1;
        }
    }

    /**
     * Advance the RBPF by one time step:
     *  1. Propagate nonlinear state via the proposal (with process noise).
     *  2. Predict the linear sub-state using the conditional KF.
     *  3. Compute observation likelihood and update log-weights.
     *  4. Update the conditional KF with the measurement.
     *  5. Normalize weights, resample if N_eff < threshold, store ancestry.
     */
    void step(float t_k,
              const Observation& y_k,
              const NonlinearState& u_k) {

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < config_.num_particles; ++i) {
            // Per-particle dynamics/observation matrices MUST be declared inside the
            // loop body: under `omp parallel for` any variable declared outside is
            // shared, so hoisting these would let threads stomp each other's
            // get_dynamics()/get_observation() output (a data race). Keep them local.
            Eigen::MatrixXf A(Types::Nlin, Types::Nlin);
            Eigen::MatrixXf B(Types::Nlin, Types::Nlin);
            LinearState bias;
            LinearCov Q;

            Eigen::MatrixXf H(Types::Ny, Types::Nlin);
            Observation offset;
            ObsCov R;

            auto& p = particles_[i];

            NonlinearState x_nl_prev = p.x_nl;
#ifdef _OPENMP
            // Per-thread RNG seeded deterministically from config_.seed (see
            // seed_thread_rngs). schedule(static) above keeps the iteration->thread
            // mapping fixed, so results are reproducible for a given thread count.
            std::mt19937_64& local_rng = thread_rngs_[omp_get_thread_num()];
            p.x_nl = nonlinear_model_.propagate(x_nl_prev, t_k, u_k, local_rng);
#else
            p.x_nl = nonlinear_model_.propagate(x_nl_prev, t_k, u_k, rng_);
#endif

            conditional_model_.get_dynamics(x_nl_prev, t_k, bias, A, B, Q);

            LinearState total_bias = bias;
            if (B.cols() == u_k.rows() && B.rows() == Types::Nlin) {
                total_bias += B * u_k;
            }

            p.kf.predict(A, total_bias, Q);

            conditional_model_.get_observation(p.x_nl, t_k, offset, H, R);

            // Likelihood calculation using accelerated operations
            Eigen::VectorXf Hx = filtermath::mat_vec_mul(H, Eigen::VectorXf(p.kf.x));
            Observation y_pred = Hx + offset;
            Observation innovation = y_k - y_pred;

            // S = H * P * H^T + R
            Eigen::MatrixXf HP = filtermath::gemm(H, p.kf.P);
            ObsCov S = filtermath::gemm(HP, H.transpose());
            S += R;

            // Compute log determinant with singularity protection
            float det = S.determinant();
            float log_det;
            if (!std::isfinite(det) || det <= 1e-30f) {
                Eigen::LDLT<ObsCov> ldlt(S);
                if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
                    log_det = -1e10f;
                } else {
                    auto D = ldlt.vectorD();
                    log_det = 0.0f;
                    for (int k = 0; k < D.size(); ++k)
                        log_det += std::log(std::max(D(k), 1e-30f));
                }
            } else {
                log_det = std::log(det);
            }

            // Mahalanobis distance via SPD solve
            Eigen::VectorXf S_solved = filtermath::solve_spd(S, innovation);
            float mahalanobis;
            if (S_solved.size() > 0) {
                mahalanobis = innovation.transpose() * S_solved;
            } else {
                mahalanobis = innovation.transpose() * S.ldlt().solve(innovation);
            }
            float log_lik = -0.5f * (mahalanobis + log_det + static_cast<float>(Types::Ny) * std::log(2.0f * std::numbers::pi_v<float>));

            p.log_weight += log_lik;

            p.kf.update(y_k, H, offset, R);
        }

        normalize_weights();

        float n_eff = get_effective_sample_size();
        std::vector<int> parents(config_.num_particles);
        std::iota(parents.begin(), parents.end(), 0);

        if (n_eff < config_.resampling_threshold * static_cast<float>(config_.num_particles)) {
            std::vector<float> weights(config_.num_particles);
            for(size_t i=0; i<static_cast<size_t>(config_.num_particles); ++i) weights[i] = std::exp(particles_[i].log_weight);

            if (config_.use_systematic_resampling) {
                parents = systematic_resampling(weights, rng_);
            } else {
                parents = stratified_resampling(weights, rng_);
            }

            std::vector<RbpfParticle<Types>> new_particles(config_.num_particles);
            for(size_t i=0; i<static_cast<size_t>(config_.num_particles); ++i) {
                new_particles[i] = particles_[parents[i]];
                new_particles[i].log_weight = -std::log(static_cast<float>(config_.num_particles));
            }
            particles_ = std::move(new_particles);
        }

        if (config_.fixed_lag > 0) {
            store_history(parent_indices_cnt_, parents);
            parent_indices_cnt_++;
        }
    }

    /** Compute weighted mean of all particles' nonlinear and linear states. */
    void get_filtered_mean(NonlinearState& x_nl_mean,
                           LinearState&    x_lin_mean) const {
        x_nl_mean.setZero();
        x_lin_mean.setZero();

        for (const auto& p : particles_) {
            float w = std::exp(p.log_weight);
            x_nl_mean += w * p.x_nl;
            x_lin_mean += w * p.kf.x;
        }
    }

    /** Return true if enough history has been accumulated for the given lag. */
    bool can_smooth(int lag) const {
        if (config_.fixed_lag <= 0) return false;
        return parent_indices_cnt_ > lag;
    }

    /**
     * Compute the smoothed mean at the given lag by backtracking ancestry
     * indices through the circular buffer to find each particle's ancestor,
     * then weighting by current importance weights.
     */
    void get_smoothed_mean(int lag,
                           NonlinearState& x_nl_mean,
                           LinearState&    x_lin_mean) const {
        if (!can_smooth(lag)) {
            get_filtered_mean(x_nl_mean, x_lin_mean);
            return;
        }

        x_nl_mean.setZero();
        x_lin_mean.setZero();

        for (int i = 0; i < config_.num_particles; ++i) {
            float w = std::exp(particles_[i].log_weight);
            int ancestor_idx = i;

            for (int step = 0; step < lag; ++step) {
                long long logical_idx = parent_indices_cnt_ - 1 - step;
                if (logical_idx < 0) break;

                int buffer_idx = logical_idx % ancestry_buffer_size_;
                const auto& parents_at_step = parent_indices_[buffer_idx];
                ancestor_idx = parents_at_step[ancestor_idx];
            }

            long long state_logical_idx = parent_indices_cnt_ - 1 - lag;
             if (state_logical_idx < 0) continue;

            int state_buffer_idx = state_logical_idx % ancestry_buffer_size_;
            const auto& historical_p = particle_history_[state_buffer_idx][ancestor_idx];

            x_nl_mean += w * historical_p.x_nl;
            x_lin_mean += w * historical_p.kf.x;
        }
    }

private:
    const NonlinearModelT& nonlinear_model_;
    const CondLinModelT&   conditional_model_;
    RbpfConfig             config_;
    std::mt19937_64        rng_;
    std::vector<RbpfParticle<Types>> particles_;

#ifdef _OPENMP
    // One RNG per OpenMP thread, seeded deterministically from config_.seed so the
    // parallel propagation honors the configured seed (previously it used a
    // std::random_device thread_local, ignoring the seed and making parallel runs
    // non-reproducible). Reproducible for a fixed OMP thread count + static schedule.
    std::vector<std::mt19937_64> thread_rngs_;
#endif

    int ancestry_buffer_size_ = 0;
    std::vector<std::vector<int>> parent_indices_;
    std::vector<std::vector<RbpfParticle<Types>>> particle_history_;
    long long parent_indices_cnt_ = 0;

    /** Normalize particle log-weights via log-sum-exp; resets to uniform if all dead. */
    void normalize_weights() {
        float max_log_w = -std::numeric_limits<float>::infinity();
        for (const auto& p : particles_) {
            // FIX: check isfinite to prevent NaN/Inf corruption
            if (std::isfinite(p.log_weight) && p.log_weight > max_log_w)
                max_log_w = p.log_weight;
        }

        // Handle degenerate case: all weights non-finite
        if (!std::isfinite(max_log_w)) {
            float uniform = -std::log(static_cast<float>(config_.num_particles));
            for (auto& p : particles_) p.log_weight = uniform;
            return;
        }

        float sum_exp = 0.0f;
        for (const auto& p : particles_) {
            if (std::isfinite(p.log_weight))
                sum_exp += std::exp(p.log_weight - max_log_w);
        }

        if (sum_exp <= 0.0f) {
            float uniform = -std::log(static_cast<float>(config_.num_particles));
            for (auto& p : particles_) p.log_weight = uniform;
            return;
        }

        float log_sum = max_log_w + std::log(sum_exp);
        for (auto& p : particles_) {
            p.log_weight -= log_sum;
        }
    }

    /** Compute ESS = 1 / sum(w_i^2). Guards against division by zero. */
    float get_effective_sample_size() const {
        float sum_sq = 0.0f;
        for (const auto& p : particles_) {
            sum_sq += std::exp(2.0f * p.log_weight);
        }
        // FIX: guard against division by zero
        if (sum_sq <= 0.0f) return static_cast<float>(config_.num_particles);
        return 1.0f / sum_sq;
    }

    /**
     * Seed one mt19937_64 per OpenMP thread deterministically from config_.seed.
     * Each thread gets a distinct, well-mixed seed (SplitMix64 finalizer) so the
     * per-thread streams are decorrelated yet fully determined by config_.seed.
     * No-op when built without OpenMP (the serial path uses rng_).
     */
    void seed_thread_rngs() {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        thread_rngs_.resize(static_cast<size_t>(nthreads));
        for (int t = 0; t < nthreads; ++t) {
            uint64_t s = config_.seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(t + 1);
            s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
            s ^= s >> 27; s *= 0x94D049BB133111EBULL;
            s ^= s >> 31;
            thread_rngs_[static_cast<size_t>(t)].seed(s);
        }
#endif
    }

    /** Store current particles and parent indices into the circular ancestry buffer. */
    void store_history(long long step_idx, const std::vector<int>& parents = {}) {
        if (ancestry_buffer_size_ == 0) return;
        int buffer_idx = step_idx % ancestry_buffer_size_;
        particle_history_[buffer_idx] = particles_;
        if (!parents.empty()) {
            parent_indices_[buffer_idx] = parents;
        } else {
             std::iota(parent_indices_[buffer_idx].begin(), parent_indices_[buffer_idx].end(), 0);
        }
    }
};

} // namespace rbpf

#endif // RBPF_CORE_HPP
