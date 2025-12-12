#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Dense>
#include <fstream>

#include "particle_filter.hpp"
#include <optmath/neon_kernels.hpp>
#include <optmath/vulkan_backend.hpp>

// --- System Model ---
// State: [px, py, vx, vy]^T
// Constant Velocity Model
constexpr int NX = 4;
constexpr int NY = 1;

class BearingOnlyModel : public PKF::StateSpaceModel<NX, NY> {
public:
    static constexpr float DT = 0.1f;
    static constexpr float SIGMA_A = 0.1f; // Acceleration noise
    static constexpr float SIGMA_THETA = 0.05f; // Measurement noise (radians) ~ 3 degrees
    static constexpr float OUTLIER_PROB = 0.1f;
    static constexpr float DF_STUDENT_T = 3.0f;

    Eigen::Matrix<float, NX, NX> F;
    Eigen::Matrix<float, NX, NX> Q;

    BearingOnlyModel() {
        F.setIdentity();
        F(0, 2) = DT;
        F(1, 3) = DT;

        // Q - Discrete White Noise Acceleration Model
        // G = [0.5 dt^2; 0.5 dt^2; dt; dt]
        float dt2 = DT * DT;
        float dt3 = dt2 * DT;
        float dt4 = dt2 * dt2;

        Q.setZero();
        float sa2 = SIGMA_A * SIGMA_A;

        // Block diagonal for x and y
        // px, vx block
        Q(0, 0) = dt4/4.0f * sa2; Q(0, 2) = dt3/2.0f * sa2;
        Q(2, 0) = dt3/2.0f * sa2; Q(2, 2) = dt2 * sa2;

        // py, vy block
        Q(1, 1) = dt4/4.0f * sa2; Q(1, 3) = dt3/2.0f * sa2;
        Q(3, 1) = dt3/2.0f * sa2; Q(3, 3) = dt2 * sa2;
    }

    State propagate(const State& x, float t, const Eigen::Ref<const State>& u) const override {
        (void)t;
        (void)u;
        // x_{k} = F * x_{k-1} + noise (noise added externally in PF step)
        return F * x;
    }

    State sample_process_noise(float t, std::mt19937_64& rng) const override {
        (void)t;
        // Multivariate Normal N(0, Q)
        static Eigen::Matrix<float, NX, NX> L = Q.llt().matrixL();

        std::normal_distribution<float> dist(0.0f, 1.0f);
        Eigen::VectorXf z(NX);
        for(int i=0; i<NX; ++i) z(i) = dist(rng);

        return L * z;
    }

    Observation observe(const State& x, float t) const override {
        (void)t;
        Observation y;
        y(0) = std::atan2(x(1), x(0));
        return y;
    }

    // Required by interface but not used in PF update (we use observation_loglik)
    Observation sample_observation_noise(float t, std::mt19937_64& rng) const override {
        (void)t;
        // Generate from Student-t
        std::student_t_distribution<float> dist(DF_STUDENT_T);
        Observation v;
        v(0) = dist(rng) * SIGMA_THETA;
        return v;
    }

    float observation_loglik(const Observation& y, const State& x, float t) const override {
        (void)t;
        // Mixture Model:
        // (1-e) * Student-t(nu, scale) + e * Uniform(-pi, pi)

        float predicted_theta = std::atan2(x(1), x(0));
        float error = y(0) - predicted_theta;

        // Wrap error to [-pi, pi)
        const float PI = static_cast<float>(M_PI);
        while(error > PI) error -= 2.0f * PI;
        while(error <= -PI) error += 2.0f * PI;

        float sigma = SIGMA_THETA;
        float nu = DF_STUDENT_T;

        // Student-t PDF
        float u = error / sigma;
        float t_dist_prob = 0.0f;

        // Gamma(2)/ (Gamma(1.5)*sqrt(3*pi)*sigma) -> 1 / (0.5*sqrt(pi) * sqrt(3*pi)*sigma)
        // -> 1 / (0.5 * pi * sqrt(3) * sigma)
        // Let's rely on std::tgamma for general nu
        float coeff = std::tgamma((nu + 1.0f)/2.0f) / (std::tgamma(nu/2.0f) * std::sqrt(nu * PI) * sigma);
        t_dist_prob = coeff * std::pow(1.0f + (u*u)/nu, -(nu+1.0f)/2.0f);

        // Outlier (Uniform)
        float uniform_prob = 1.0f / (2.0f * PI);

        float total_prob = (1.0f - OUTLIER_PROB) * t_dist_prob + OUTLIER_PROB * uniform_prob;

        return std::log(total_prob);
    }
};

// --- Test Implementation ---

struct SimulationResult {
    std::vector<float> time;
    std::vector<Eigen::VectorXf> truth;
    std::vector<Eigen::VectorXf> estimates;
    std::vector<float> measurements;
    float rmse_pos;
};

SimulationResult run_pkf_test() {
    BearingOnlyModel model;
    int steps = 200;
    int num_particles = 4000;

    PKF::ParticleFilter<NX, NY> pf(&model, num_particles);

    // Truth Trajectory
    // Start at (-50, 20), moving (10, 0)
    // Avoids passing through (0,0) singularity
    Eigen::VectorXf x_true(NX);
    x_true << -50.0f, 20.0f, 10.0f, 0.0f;

    std::vector<Eigen::VectorXf> truth_hist;
    std::vector<Eigen::VectorXf> est_hist;
    std::vector<float> meas_hist;
    std::vector<float> time_hist;

    // Initial Prior: Gaussian around true state with large variance
    Eigen::VectorXf x0_mean = x_true;
    // Add some error to initialization
    x0_mean(0) += 5.0f;
    x0_mean(1) -= 5.0f;
    x0_mean(2) += 1.0f;
    x0_mean(3) += 1.0f;

    Eigen::MatrixXf P0 = Eigen::MatrixXf::Identity(NX, NX);
    P0.topLeftCorner(2,2) *= 100.0f; // Large position uncertainty
    P0.bottomRightCorner(2,2) *= 25.0f; // Large velocity uncertainty

    std::mt19937_64 rng(42);

    // Initialize PF
    pf.initialize([&](std::mt19937_64& r) {
        Eigen::VectorXf x = x0_mean;
        // Sample from P0
        for(int i=0; i<NX; ++i) {
             std::normal_distribution<float> dist(0.0f, std::sqrt(P0(i,i)));
             x(i) += dist(r);
        }
        return x;
    });

    const float PI = static_cast<float>(M_PI);
    std::uniform_real_distribution<float> outlier_dist(-PI, PI);
    std::student_t_distribution<float> t_dist(BearingOnlyModel::DF_STUDENT_T);

    float rmse_sum_sq = 0.0f;

    for (int k = 0; k < steps; ++k) {
        float t = static_cast<float>(k) * BearingOnlyModel::DT;

        // 1. Propagate Truth
        if (k > 0) {
            Eigen::VectorXf u = Eigen::VectorXf::Zero(NX); // dummy
            x_true = model.propagate(x_true, t, u) + model.sample_process_noise(t, rng);
        }

        // 2. Generate Measurement
        float true_theta = std::atan2(x_true(1), x_true(0));
        float noise;

        // Mixture generation
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        if (u01(rng) < BearingOnlyModel::OUTLIER_PROB) {
            // Outlier: pure noise replacing measurement
            true_theta = outlier_dist(rng);
            noise = 0.0f;
        } else {
             noise = t_dist(rng) * BearingOnlyModel::SIGMA_THETA;
        }

        Eigen::VectorXf y(1);
        y(0) = true_theta + noise;
        // Wrap
        while(y(0) > PI) y(0) -= 2.0f * PI;
        while(y(0) <= -PI) y(0) += 2.0f * PI;

        // 3. PF Step
        Eigen::VectorXf u_dummy = Eigen::VectorXf::Zero(NX);
        pf.step(y, t, u_dummy);
        pf.resample_if_needed();

        Eigen::VectorXf est = pf.get_mean();

        // Store
        truth_hist.push_back(x_true);
        est_hist.push_back(est);
        meas_hist.push_back(y(0));
        time_hist.push_back(t);

        // Compute error (skip first few steps for convergence)
        if (k > 20) {
            float err_x = x_true(0) - est(0);
            float err_y = x_true(1) - est(1);
            rmse_sum_sq += err_x*err_x + err_y*err_y;
        }
    }

    float rmse = std::sqrt(rmse_sum_sq / static_cast<float>(steps - 20));

    SimulationResult res;
    res.time = time_hist;
    res.truth = truth_hist;
    res.estimates = est_hist;
    res.measurements = meas_hist;
    res.rmse_pos = rmse;

    return res;
}

int main() {
    std::cout << "Running PKF Bearing-Only Tracking Test..." << std::endl;
    std::cout << "Optimizations: " << (optmath::vulkan::is_available() ? "Vulkan ENABLED" : "CPU Fallback") << std::endl;

    SimulationResult res = run_pkf_test();

    std::cout << "Final Position RMSE (excluding transient): " << res.rmse_pos << " m" << std::endl;

    // Threshold is loose because bearing-only is unobservable initially and depends on geometry
    if (res.rmse_pos < 20.0f) {
        std::cout << "SUCCESS: RMSE acceptable." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: RMSE too high." << std::endl;
        return 1;
    }
}
