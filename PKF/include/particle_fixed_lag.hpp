#ifndef PKF_PARTICLE_FIXED_LAG_HPP
#define PKF_PARTICLE_FIXED_LAG_HPP

#include "particle_filter.hpp"
#include <deque>

namespace PKF {

/**
 * @class ParticleFilterFixedLag
 * @brief Fixed-Lag Particle Smoother wrapper around ParticleFilter.
 *
 * Implements ancestry tracking to provide smoothed estimates p(x_{k-L} | y_{1:k}).
 *
 * @tparam NX State dimension
 * @tparam NY Observation dimension
 */
template<int NX, int NY>
class ParticleFilterFixedLag {
public:
    using PF = ParticleFilter<NX, NY>;
    using State = typename PF::State;
    using StateMat = typename PF::StateMat;
    using Observation = typename PF::Observation;

    /**
     * @brief Constructor
     *
     * @param model Pointer to model
     * @param num_particles Number of particles
     * @param lag Fixed lag L
     */
    ParticleFilterFixedLag(const typename PF::Model* model, size_t num_particles, size_t lag, double resampling_threshold = 0.5)
        : filter_(model, num_particles, resampling_threshold), lag_(lag), N_(num_particles) {
    }

    // Forward wrapper for initialization
    template<typename Sampler>
    void initialize(Sampler&& prior_sampler) {
        filter_.initialize(prior_sampler);

        // Initialize history with initial particles
        // At k=0 (init), we have x_0.
        // History stores [x_0].
        // Parent history is empty initially.

        history_particles_.clear();
        history_particles_.push_back(filter_.get_particles());

        // No parents for initial step (or identity)
        // history_parents_ stores parents from k-1 to k.
        // So for init step there is no "previous" step.
        history_parents_.clear();
    }

    void set_seed(uint64_t seed) {
        filter_.set_seed(seed);
    }

    /**
     * @brief Perform one step of filtering and update smoother history
     */
    void step(const Observation& y_k, double t_k, const Eigen::Ref<const State>& u_k) {
        // Run Filter Step
        filter_.step(y_k, t_k, u_k);

        // Resample and get ancestry
        // Note: filter_.resample_if_needed() returns indices of parents at k-1 that generated particles at k.
        // If no resampling occurred, it returns 0..N-1.
        std::vector<size_t> parents = filter_.resample_if_needed();

        // Update History

        // 1. Store current particles (at time k)
        history_particles_.push_back(filter_.get_particles());

        // 2. Store parent indices (mapping k -> k-1)
        history_parents_.push_back(parents);

        // 3. Trim history if it exceeds Lag + 1
        // We need x_{k-L} ... x_k.
        // History size needed: L + 1 states.
        // Parent arrays needed: L transitions.

        if (history_particles_.size() > lag_ + 1) {
            history_particles_.pop_front();
            history_parents_.pop_front();
        }
    }

    /**
     * @brief Get Filtered Mean (current time k)
     */
    State get_filtered_mean() const {
        return filter_.get_mean();
    }

    /**
     * @brief Get Filtered Covariance (current time k)
     */
    StateMat get_filtered_covariance() const {
        return filter_.get_covariance();
    }

    /**
     * @brief Get Smoothed Mean at lag L (time k-L)
     *
     * Uses current weights w_k and backtracks ancestry to find x_{k-L}.
     */
    State get_smoothed_mean() const {
        if (history_particles_.size() <= lag_) {
            // Not enough history yet, return filtered mean at that time step or best effort?
            // If k < L, we are effectively smoothing x_0 given y_{1:k}.
            // The history stores x_0, ..., x_k.
            // The "smoothed mean" usually implies the oldest state in the buffer given all data.
            // Let's return the oldest state mean computed with current weights.
            return compute_smoothed_mean_at_index(0);
        }
        return compute_smoothed_mean_at_index(0); // 0 is the oldest in the deque
    }

    /**
     * @brief Get Smoothed Covariance at lag L (time k-L)
     */
    StateMat get_smoothed_covariance() const {
        if (history_particles_.size() <= lag_) {
             return compute_smoothed_cov_at_index(0);
        }
        return compute_smoothed_cov_at_index(0);
    }

private:
    ParticleFilter<NX, NY> filter_;
    size_t lag_;
    size_t N_;

    // Circular buffers (using deque)
    // history_particles_[0] corresponds to oldest stored state (x_{k-L})
    // history_particles_.back() corresponds to current state (x_k)
    std::deque<std::vector<State>> history_particles_;

    // history_parents_[0] maps particles from index 1 to 0 (x_{k-L+1} -> x_{k-L})
    // history_parents_.back() maps particles from k to k-1 (x_k -> x_{k-1})
    std::deque<std::vector<size_t>> history_parents_;

    /**
     * @brief Backtrack ancestry to find indices at history_index
     *
     * @param history_idx Index in the deque (0 = oldest)
     * @return std::vector<size_t> Indices of particles at history_idx corresponding to current particles 0..N-1
     */
    std::vector<size_t> backtrack_indices(size_t history_idx) const {
        // Start with identity at current time (last in deque)
        std::vector<size_t> indices(N_);
        std::iota(indices.begin(), indices.end(), 0);

        // Iterate backwards from current time down to history_idx
        // parents deque size is particles deque size - 1.
        // particles: [0, 1, ..., M]
        // parents:   [0->1, 1->2, ..., M-1->M] NO
        // parents: [map 1 to 0, map 2 to 1, ..., map M to M-1]
        // Let's clarify:
        // history_parents_[t] stores indices of parents at time t for particles at time t+1 relative to history start?
        // Let's align with step():
        // history_parents_.push_back(parents) where parents[i] is index at k-1.
        // So history_parents_.back() maps x_k to x_{k-1}.

        // We want to map x_k indices to x_{history_idx} indices.
        // We traverse parents from back to front.

        // history_particles_ has size S.
        // history_parents_ has size S - 1.
        // indices 0..N-1 correspond to history_particles_.back() (current time)

        size_t current_layer = history_particles_.size() - 1;

        // While we are above the target layer
        size_t parent_layer_idx = history_parents_.size(); // 1 past end

        while (current_layer > history_idx) {
            parent_layer_idx--; // Move to the parent map responsible for current_layer -> current_layer - 1
            const auto& parents = history_parents_[parent_layer_idx];

            for (size_t i = 0; i < N_; ++i) {
                indices[i] = parents[indices[i]];
            }

            current_layer--;
        }

        return indices;
    }

    State compute_smoothed_mean_at_index(size_t idx) const {
        std::vector<size_t> ancestral_indices = backtrack_indices(idx);
        const auto& target_particles = history_particles_[idx];
        const auto& current_log_weights = filter_.get_log_weights();

        State mean = State::Zero();
        for (size_t i = 0; i < N_; ++i) {
            // Weight w_k^{(i)} applies to particle x_{k-L}^{A(i)}
            mean += std::exp(current_log_weights[i]) * target_particles[ancestral_indices[i]];
        }
        return mean;
    }

    StateMat compute_smoothed_cov_at_index(size_t idx) const {
        State mean = compute_smoothed_mean_at_index(idx);
        std::vector<size_t> ancestral_indices = backtrack_indices(idx);
        const auto& target_particles = history_particles_[idx];
        const auto& current_log_weights = filter_.get_log_weights();

        StateMat cov = StateMat::Zero();
        for (size_t i = 0; i < N_; ++i) {
            State diff = target_particles[ancestral_indices[i]] - mean;
            cov += std::exp(current_log_weights[i]) * (diff * diff.transpose());
        }
        return cov;
    }
};

} // namespace PKF

#endif // PKF_PARTICLE_FIXED_LAG_HPP
