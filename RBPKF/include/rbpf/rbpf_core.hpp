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
#include <limits>
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

    // The linear transition INTO this epoch, kept for the fixed-lag backward
    // (RTS) pass: x_lin_k = A_in x_lin_{k-1} + bias + w, and the KF predicted
    // moments it produced. Written by step(); unused at the initial epoch.
    typename Types::Matrix_A A_in = Types::Matrix_A::Identity();
    LinearState       x_pred = LinearState::Zero();
    LinearCov         P_pred = LinearCov::Identity();
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

        ensure_thread_rngs();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < config_.num_particles; ++i) {
            // Per-particle dynamics/observation matrices MUST be declared inside the
            // loop body: under `omp parallel for` any variable declared outside is
            // shared, so hoisting these would let threads stomp each other's
            // get_dynamics()/get_observation() output (a data race). Keep them local.
            // Fixed-size A/H (compile-time dims) so the per-particle gemm/mat_vec
            // calls below and inside kf.predict()/update() bind to FilterMath's
            // fixed-size fast path (stack, no dispatch, no heap). They still bind
            // to the model's Eigen::Ref<MatrixXf> out-params (a fixed-size,
            // contiguous matrix is a valid dynamic Ref target). B is dynamic and
            // sized Nlin x Nnl: the control u_k is a NonlinearState (Nnl rows) and
            // the model's Ref<MatrixXf> cannot resize it. It was Nlin x Nlin, so
            // B*u_k was silently skipped whenever Nnl != Nlin. Zeroed so a model
            // that leaves B untouched adds no control term.
            typename Types::Matrix_A A;
            Eigen::MatrixXf B = Eigen::MatrixXf::Zero(Types::Nlin, Types::Nnl);
            LinearState bias;
            LinearCov Q;

            typename Types::Matrix_H H;
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
            p.A_in = A;
            p.x_pred = p.kf.x;
            p.P_pred = p.kf.P;

            conditional_model_.get_observation(p.x_nl, t_k, offset, H, R);

            // Likelihood calculation using accelerated operations (fixed-size fast path)
            Observation Hx = filtermath::mat_vec_mul(H, p.kf.x);
            Observation y_pred = Hx + offset;
            Observation innovation = y_k - y_pred;

            // S = H * P * H^T + R
            Eigen::Matrix<float, Types::Ny, Types::Nlin> HP = filtermath::gemm(H, p.kf.P);
            ObsCov S = filtermath::gemm(HP, H.transpose());
            S += R;

            const float log_lik = predictive_log_likelihood(S, innovation);

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
            // A zero-weight particle may hold a non-finite state (that is why
            // its weight is zero); 0 * NaN = NaN would still poison the mean.
            if (!(w > 0.0f)) continue;
            x_nl_mean += w * p.x_nl;
            x_lin_mean += w * p.kf.x;
        }
    }

    /**
     * Return true if enough history has been accumulated for the given lag.
     * lag must lie in [0, fixed_lag]: the circular buffers hold fixed_lag + 1
     * epochs, so a longer lag would read slots already overwritten (it used to
     * return another epoch's estimate with can_smooth() == true).
     */
    bool can_smooth(int lag) const {
        if (config_.fixed_lag <= 0) return false;
        if (lag < 0 || lag > config_.fixed_lag) return false;
        return parent_indices_cnt_ > lag;
    }

    /**
     * Fixed-lag smoothed means at epoch k - lag given y_{1:k} (k = current).
     *
     * Nonlinear part: each particle's ancestor at k - lag (ancestral lineage),
     * weighted by the current weights.
     *
     * Linear part: conditional on a particle's nonlinear path the linear
     * sub-state is exactly linear-Gaussian, so a Rauch-Tung-Striebel pass back
     * along that path, using the filtered moments and transitions stored with
     * the ancestors,
     *     G_t   = P_{t|t} A_{t+1}^T P_{t+1|t}^{-1}
     *     x^s_t = x_{t|t} + G_t (x^s_{t+1} - x_{t+1|t}),   x^s_k = x_{k|k},
     * gives E[x_lin_{k-lag} | x_nl path, y_{1:k}] exactly (derived); the
     * weighted sum over particles estimates E[x_lin_{k-lag} | y_{1:k}].
     * (Until 2026-11-16 this returned the ancestor's FILTERED linear mean
     * E[x_lin_{k-lag} | path, y_{1:k-lag}], ignoring the last `lag`
     * measurements: 0 where the exact smoothed value was 0.952.)
     * Path degeneracy affects both parts as in any ancestral-lineage smoother.
     */
    void get_smoothed_mean(int lag,
                           NonlinearState& x_nl_mean,
                           LinearState&    x_lin_mean) const {
        if (!can_smooth(lag)) {
            get_filtered_mean(x_nl_mean, x_lin_mean);
            return;
        }
        const int N = config_.num_particles;
        std::vector<NonlinearState, Eigen::aligned_allocator<NonlinearState>> nl(static_cast<size_t>(N));
        std::vector<LinearState, Eigen::aligned_allocator<LinearState>> lin(static_cast<size_t>(N));
        std::vector<float> w(static_cast<size_t>(N));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (N >= 2048)
#endif
        for (int i = 0; i < N; ++i) {
            w[static_cast<size_t>(i)] = std::exp(particles_[i].log_weight);
            if (!(w[static_cast<size_t>(i)] > 0.0f)) continue;   // see get_filtered_mean
            smooth_path(i, lag, nl[static_cast<size_t>(i)], lin[static_cast<size_t>(i)]);
        }
        // Serial sum in particle order: deterministic for any thread count.
        x_nl_mean.setZero();
        x_lin_mean.setZero();
        for (int i = 0; i < N; ++i) {
            const float wi = w[static_cast<size_t>(i)];
            if (!(wi > 0.0f)) continue;
            x_nl_mean += wi * nl[static_cast<size_t>(i)];
            x_lin_mean += wi * lin[static_cast<size_t>(i)];
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

public:
    /**
     * log N(innovation; 0, S) for the particle's predictive measurement density.
     * Public and static so tests exercise the exact code the filter runs.
     *
     * S = H P H^T + R is SPD in exact arithmetic, but float rounding can make
     * it fail a factorisation (e.g. a near-rank-one H P H^T with tiny R). The
     * previous code then set log_det = -1e10, i.e. log_lik ~ +5e9: the
     * numerically failed particle received ALL the weight (and at that
     * magnitude float spacing is 512, so normalisation broke too). Now S is
     * symmetrised and factored by LLT with a trace-relative jitter ladder; a
     * particle whose S still cannot be factored gets log_lik = -inf (zero
     * weight), never a bonus. log_det and the Mahalanobis term come from the
     * same factor: log|S| = 2 sum log L_ii, m = |L^{-1} innovation|^2.
     */
    static float predictive_log_likelihood(const ObsCov& S_in, const Observation& innovation) {
        if (!S_in.allFinite() || !innovation.allFinite()) {
            return -std::numeric_limits<float>::infinity();
        }
        const ObsCov S = 0.5f * (S_in + S_in.transpose());
        Eigen::LLT<ObsCov> llt(S);
        if (llt.info() != Eigen::Success) {
            const float scale = std::max(S.trace() / static_cast<float>(Types::Ny), 1e-30f);
            for (float rel = 1e-7f; rel <= 1e-3f && llt.info() != Eigen::Success; rel *= 100.0f) {
                llt.compute(S + (rel * scale) * ObsCov::Identity());
            }
        }
        if (llt.info() != Eigen::Success) {
            return -std::numeric_limits<float>::infinity();
        }
        const auto L = llt.matrixL();
        float log_det = 0.0f;
        for (int k = 0; k < Types::Ny; ++k) log_det += std::log(llt.matrixLLT()(k, k));
        log_det *= 2.0f;
        const Observation z = L.solve(innovation);
        const float mahalanobis = z.squaredNorm();
        return -0.5f * (mahalanobis + log_det +
                        static_cast<float>(Types::Ny) * std::log(2.0f * std::numbers::pi_v<float>));
    }

private:
    /** Normalize particle log-weights via log-sum-exp; resets to uniform if all dead. */
    void normalize_weights() {
        // Any non-finite log-weight (NaN from a model, +inf) is an invalid
        // particle: give it zero weight. NaN used to survive normalisation
        // (NaN - log_sum), make ESS NaN, block resampling forever (NaN < x is
        // false) and poison every mean.
        for (auto& p : particles_) {
            if (!std::isfinite(p.log_weight))
                p.log_weight = -std::numeric_limits<float>::infinity();
        }
        float max_log_w = -std::numeric_limits<float>::infinity();
        for (const auto& p : particles_) {
            if (p.log_weight > max_log_w) max_log_w = p.log_weight;
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
            uint64_t s = config_.seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(t + 1);
            s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
            s ^= s >> 27; s *= 0x94D049BB133111EBULL;
            s ^= s >> 31;
            thread_rngs_[static_cast<size_t>(t)].seed(s);
        }
#endif
    }

    /**
     * Backward pass along particle i's ancestry, from the current epoch k down
     * to k - lag. history[t][a_t] is the ancestor at t (a_k = i,
     * a_{t-1} = parents[t][a_t]); the transition t -> t+1 is the one stored
     * with history[t+1][a_{t+1}]. Done in double: P_{t+1|t} can be
     * ill-conditioned when Q is small.
     */
    void smooth_path(int i, int lag, NonlinearState& x_nl_out, LinearState& x_lin_out) const {
        using VecD = Eigen::Matrix<double, Types::Nlin, 1>;
        using MatD = Eigen::Matrix<double, Types::Nlin, Types::Nlin>;
        const long long k = parent_indices_cnt_ - 1;
        int a = i;
        const auto* next = &particle_history_[static_cast<size_t>(k % ancestry_buffer_size_)][static_cast<size_t>(a)];
        VecD xs = next->kf.x.template cast<double>();
        for (int s = 1; s <= lag; ++s) {
            a = parent_indices_[static_cast<size_t>((k - s + 1) % ancestry_buffer_size_)][static_cast<size_t>(a)];
            const auto& cur = particle_history_[static_cast<size_t>((k - s) % ancestry_buffer_size_)][static_cast<size_t>(a)];
            const MatD Pf = cur.kf.P.template cast<double>();
            const MatD Pp = next->P_pred.template cast<double>();
            const MatD A = next->A_in.template cast<double>();
            // G^T = Pp^{-1} A Pf  (Pp, Pf symmetric)
            const Eigen::LDLT<MatD> ldlt(0.5 * (Pp + Pp.transpose()));
            const MatD G = ldlt.solve(A * Pf).transpose();
            xs = cur.kf.x.template cast<double>() + G * (xs - next->x_pred.template cast<double>());
            next = &cur;
        }
        x_nl_out = next->x_nl;
        x_lin_out = xs.template cast<float>();
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
