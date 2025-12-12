#include "rbpf/rbpf_core.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

constexpr int N_NL = 1;
constexpr int N_LIN = 4;
constexpr int N_Y = 2;

using AppTypes = rbpf::RbpfTypes<N_NL, N_LIN, N_Y>;

class CtrvNonlinearModel : public rbpf::NonlinearModel<AppTypes> {
public:
    float dt;
    float std_omega;

    CtrvNonlinearModel(float dt_val, float std_omega_val)
        : dt(dt_val), std_omega(std_omega_val) {}

    NonlinearState propagate(const NonlinearState& x_nl_prev,
                             float,
                             const NonlinearState&,
                             std::mt19937_64& rng) const override {
        std::normal_distribution<float> dist(0.0f, std_omega);
        NonlinearState x_nl_new;
        x_nl_new(0) = x_nl_prev(0) + dist(rng);
        return x_nl_new;
    }

    float log_proposal_density(const NonlinearState& x_nl_curr,
                                const NonlinearState& x_nl_prev,
                                float,
                                const NonlinearState&) const override {
        float diff = x_nl_curr(0) - x_nl_prev(0);
        float var = std_omega * std_omega;
        return -0.5f * (std::log(2.0f * static_cast<float>(M_PI) * var) + (diff * diff) / var);
    }
};

class CtrvConditionalModel : public rbpf::ConditionalLinearGaussianModel<AppTypes> {
public:
    float dt;
    float std_acc;
    float std_range;
    float std_bearing;

    CtrvConditionalModel(float dt_val, float std_a, float std_r, float std_b)
        : dt(dt_val), std_acc(std_a), std_range(std_r), std_bearing(std_b) {}

    void get_dynamics(const NonlinearState& x_nl_prev,
                      float,
                      Eigen::Ref<LinearState> bias,
                      Eigen::Ref<Eigen::MatrixXf> A,
                      Eigen::Ref<Eigen::MatrixXf> B,
                      Eigen::Ref<LinearCov> Q) const override {
        float omega = x_nl_prev(0);
        bias.setZero();
        B.setZero();

        if (std::abs(omega) < 1e-5f) {
            A.setIdentity();
            A(0, 2) = dt;
            A(1, 3) = dt;
        } else {
            float sin_w = std::sin(omega * dt);
            float cos_w = std::cos(omega * dt);

            A.setIdentity();
            A(0, 2) = sin_w / omega;
            A(0, 3) = -(1.0f - cos_w) / omega;
            A(1, 2) = (1.0f - cos_w) / omega;
            A(1, 3) = sin_w / omega;

            A(2, 2) = cos_w;
            A(2, 3) = -sin_w;
            A(3, 2) = sin_w;
            A(3, 3) = cos_w;
        }

        Q.setZero();
        float var_a = std_acc * std_acc;
        Q(2, 2) = var_a * dt;
        Q(3, 3) = var_a * dt;
    }

    void get_observation(const NonlinearState&,
                         float,
                         Eigen::Ref<Observation> offset,
                         Eigen::Ref<Eigen::MatrixXf> H,
                         Eigen::Ref<ObsCov> R) const override {
        offset.setZero();
        H.setZero();
        H(0, 0) = 1.0f;
        H(1, 1) = 1.0f;
        R.setZero();
        R(0, 0) = std_range * std_range;
        R(1, 1) = std_range * std_range;
    }
};

int main() {
    rbpf::RbpfConfig config;
    config.num_particles = 1000;
    config.resampling_threshold = 0.5f;
    config.fixed_lag = 10;
    config.seed = 42;

    float dt = 0.1f;
    CtrvNonlinearModel nl_model(dt, 0.05f);
    CtrvConditionalModel lin_model(dt, 0.5f, 2.0f, 0.0f);

    rbpf::RaoBlackwellizedParticleFilter<AppTypes, CtrvNonlinearModel, CtrvConditionalModel> rbpf(nl_model, lin_model, config);

    Eigen::VectorXf true_x_lin(4);
    true_x_lin << 0.0f, 0.0f, 10.0f, 0.0f;
    float true_omega = 0.1f;

    AppTypes::NonlinearState x_nl0; x_nl0 << 0.0f;
    AppTypes::LinearState x_lin0; x_lin0 << 0.0f, 0.0f, 10.0f, 0.0f;
    AppTypes::LinearCov P_lin0 = AppTypes::LinearCov::Identity() * 1.0f;

    rbpf.initialize(x_nl0, x_lin0, P_lin0);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> noise_obs(0.0f, 2.0f);

    std::cout << "Time,TrueX,TrueY,EstX,EstY,RMSE_Pos" << std::endl;

    float total_sq_err = 0.0f;
    int steps = 100;

    for (int k = 0; k < steps; ++k) {
        float t = k * dt;

        Eigen::MatrixXf A(4, 4);
        if (std::abs(true_omega) < 1e-5f) {
            A.setIdentity();
            A(0,2)=dt; A(1,3)=dt;
        } else {
             float sw = std::sin(true_omega * dt);
             float cw = std::cos(true_omega * dt);
             A.setIdentity();
             A(0,2)=sw/true_omega; A(0,3)=-(1.0f-cw)/true_omega;
             A(1,2)=(1.0f-cw)/true_omega; A(1,3)=sw/true_omega;
             A(2,2)=cw; A(2,3)=-sw;
             A(3,2)=sw; A(3,3)=cw;
        }
        true_x_lin = A * true_x_lin;

        AppTypes::Observation y;
        y(0) = true_x_lin(0) + noise_obs(rng);
        y(1) = true_x_lin(1) + noise_obs(rng);

        AppTypes::NonlinearState u; u.setZero();
        rbpf.step(t, y, u);

        AppTypes::NonlinearState est_nl;
        AppTypes::LinearState est_lin;
        rbpf.get_filtered_mean(est_nl, est_lin);

        float dx = est_lin(0) - true_x_lin(0);
        float dy = est_lin(1) - true_x_lin(1);
        float err_sq = dx*dx + dy*dy;
        total_sq_err += err_sq;

        std::cout << t << "," << true_x_lin(0) << "," << true_x_lin(1) << ","
                  << est_lin(0) << "," << est_lin(1) << "," << std::sqrt(err_sq) << std::endl;
    }

    std::cout << "Average RMSE: " << std::sqrt(total_sq_err / static_cast<float>(steps)) << std::endl;

    return 0;
}
