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

namespace rbpf {

template<typename Types>
struct RbpfParticle {
    using NonlinearState = typename Types::NonlinearState;
    using LinearState    = typename Types::LinearState;
    using LinearCov      = typename Types::LinearCov;

    NonlinearState    x_nl;
    LinearKalmanFilter<Types> kf;
    double            log_weight;
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

    RaoBlackwellizedParticleFilter(const NonlinearModelT& nonlinear_model,
                                   const CondLinModelT& conditional_model,
                                   const RbpfConfig& config)
        : nonlinear_model_(nonlinear_model),
          conditional_model_(conditional_model),
          config_(config) {

        particles_.resize(config_.num_particles);
        rng_.seed(config_.seed);

        if (config_.fixed_lag > 0) {
            ancestry_buffer_size_ = config_.fixed_lag + 1;
            parent_indices_.resize(ancestry_buffer_size_);
            particle_history_.resize(ancestry_buffer_size_);
            for(auto& vec : parent_indices_) vec.resize(config_.num_particles);
            for(auto& vec : particle_history_) vec.resize(config_.num_particles);
        }
    }

    void initialize(const NonlinearState& x_nl0,
                    const LinearState&    x_lin0,
                    const LinearCov&      P_lin0) {
        double init_log_weight = -std::log(static_cast<double>(config_.num_particles));

        for (auto& p : particles_) {
            p.x_nl = x_nl0;
            p.kf.initialize(x_lin0, P_lin0);
            p.log_weight = init_log_weight;
        }

        if (config_.fixed_lag > 0) {
            store_history(0);
        }
    }

    void step(double t_k,
              const Observation& y_k,
              const NonlinearState& u_k) {

        Eigen::MatrixXd A(Types::Nlin, Types::Nlin);
        Eigen::MatrixXd B(Types::Nlin, Types::Nlin);
        LinearState bias;
        LinearCov Q;

        Eigen::MatrixXd H(Types::Ny, Types::Nlin);
        Observation offset;
        ObsCov R;

        for (int i = 0; i < config_.num_particles; ++i) {
            auto& p = particles_[i];

            NonlinearState x_nl_prev = p.x_nl;
            p.x_nl = nonlinear_model_.propagate(x_nl_prev, t_k, u_k, rng_);

            conditional_model_.get_dynamics(x_nl_prev, t_k, bias, A, B, Q);

            LinearState total_bias = bias;
            // Handle B*u if applicable
            if (B.cols() == u_k.rows() && B.rows() == Types::Nlin) {
                total_bias += B * u_k;
            }

            p.kf.predict(A, total_bias, Q);

            conditional_model_.get_observation(p.x_nl, t_k, offset, H, R);

            Observation y_pred = H * p.kf.x + offset;
            Observation innovation = y_k - y_pred;
            ObsCov S = H * p.kf.P * H.transpose() + R;

            double log_det = std::log(S.determinant());
            double mahalanobis = innovation.transpose() * S.ldlt().solve(innovation);
            double log_lik = -0.5 * (mahalanobis + log_det + Types::Ny * std::log(2 * M_PI));

            p.log_weight += log_lik;

            p.kf.update(y_k, H, offset, R);
        }

        normalize_weights();

        double n_eff = get_effective_sample_size();
        std::vector<int> parents(config_.num_particles);
        std::iota(parents.begin(), parents.end(), 0);

        if (n_eff < config_.resampling_threshold * config_.num_particles) {
            std::vector<double> weights(config_.num_particles);
            for(size_t i=0; i<config_.num_particles; ++i) weights[i] = std::exp(particles_[i].log_weight);

            if (config_.use_systematic_resampling) {
                parents = systematic_resampling(weights, rng_);
            } else {
                parents = stratified_resampling(weights, rng_);
            }

            std::vector<RbpfParticle<Types>> new_particles(config_.num_particles);
            for(size_t i=0; i<config_.num_particles; ++i) {
                new_particles[i] = particles_[parents[i]];
                new_particles[i].log_weight = -std::log(static_cast<double>(config_.num_particles));
            }
            particles_ = std::move(new_particles);
        }

        if (config_.fixed_lag > 0) {
            store_history(parent_indices_cnt_, parents);
            parent_indices_cnt_++;
        }
    }

    void get_filtered_mean(NonlinearState& x_nl_mean,
                           LinearState&    x_lin_mean) const {
        x_nl_mean.setZero();
        x_lin_mean.setZero();

        for (const auto& p : particles_) {
            double w = std::exp(p.log_weight);
            x_nl_mean += w * p.x_nl;
            x_lin_mean += w * p.kf.x;
        }
    }

    bool can_smooth(int lag) const {
        if (config_.fixed_lag <= 0) return false;
        return parent_indices_cnt_ > lag;
    }

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
            double w = std::exp(particles_[i].log_weight);
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

    int ancestry_buffer_size_ = 0;
    std::vector<std::vector<int>> parent_indices_;
    std::vector<std::vector<RbpfParticle<Types>>> particle_history_;
    long long parent_indices_cnt_ = 0;

    void normalize_weights() {
        double max_log_w = -std::numeric_limits<double>::infinity();
        for (const auto& p : particles_) {
            if (p.log_weight > max_log_w) max_log_w = p.log_weight;
        }
        double sum_exp = 0.0;
        for (const auto& p : particles_) {
            sum_exp += std::exp(p.log_weight - max_log_w);
        }
        double log_sum = max_log_w + std::log(sum_exp);
        for (auto& p : particles_) {
            p.log_weight -= log_sum;
        }
    }

    double get_effective_sample_size() const {
        double sum_sq = 0.0;
        for (const auto& p : particles_) {
            sum_sq += std::exp(2.0 * p.log_weight);
        }
        return 1.0 / sum_sq;
    }

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
