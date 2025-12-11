#ifndef PKF_LORENZ63_MODEL_HPP
#define PKF_LORENZ63_MODEL_HPP

#include "state_space_model.hpp"
#include "noise_models.hpp"

namespace PKF {

/**
 * @class Lorenz63Model
 * @brief Lorenz-63 Chaotic System with non-Gaussian observations.
 *
 * State: [x, y, z]
 * Dynamics:
 * dx/dt = sigma * (y - x)
 * dy/dt = x * (rho - z) - y
 * dz/dt = x * y - beta * z
 *
 * Observation:
 * y_k = [x, y, z] + v_k
 * v_k ~ Student-t(nu=3, Sigma=R)
 */
class Lorenz63Model : public StateSpaceModel<3, 3> {
public:
    // Lorenz Parameters
    static constexpr double SIGMA = 10.0;
    static constexpr double RHO = 28.0;
    static constexpr double BETA = 8.0 / 3.0;
    static constexpr double DT = 0.01; // Integration step

    // Noise Parameters
    static constexpr double PROCESS_NOISE_STD = 1.0; // Reduced for stability in discrete time? Or standard.
    static constexpr double OBS_NOISE_STD = 2.0;
    static constexpr double OBS_NU = 3.0; // Degrees of freedom for Student-t

    Lorenz63Model() {
        Q_chol_.setIdentity();
        Q_chol_ *= std::sqrt(DT) * PROCESS_NOISE_STD; // Simple random walk diffusion approximation

        R_.setIdentity();
        R_ *= (OBS_NOISE_STD * OBS_NOISE_STD);
    }

    State propagate(const State& x_prev, double t_k, const Eigen::Ref<const State>& u_k) const override {
        // RK4 Integration
        (void)t_k;
        (void)u_k;

        auto dynamics = [](const State& s) -> State {
            double x = s(0);
            double y = s(1);
            double z = s(2);
            State ds;
            ds(0) = SIGMA * (y - x);
            ds(1) = x * (RHO - z) - y;
            ds(2) = x * y - BETA * z;
            return ds;
        };

        State k1 = dynamics(x_prev);
        State k2 = dynamics(x_prev + 0.5 * DT * k1);
        State k3 = dynamics(x_prev + 0.5 * DT * k2);
        State k4 = dynamics(x_prev + DT * k3);

        return x_prev + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    Observation observe(const State& x_k, double t_k) const override {
        (void)t_k;
        // Direct observation of full state
        return x_k;
    }

    State sample_process_noise(double t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        // Gaussian process noise
        return Noise::gaussian_sample<3>(State::Zero(), Q_chol_ * Q_chol_.transpose(), rng);
    }

    Observation sample_observation_noise(double t_k, std::mt19937_64& rng) const override {
        (void)t_k;
        // Student-t measurement noise
        return Noise::student_t_sample<3>(Observation::Zero(), R_, OBS_NU, rng);
    }

    double observation_loglik(const Observation& y_k, const State& x_k, double t_k) const override {
        (void)t_k;
        Observation expected = observe(x_k, t_k);
        return Noise::student_t_logpdf<3>(y_k, expected, R_, OBS_NU);
    }

private:
    Eigen::Matrix<double, 3, 3> Q_chol_;
    Eigen::Matrix<double, 3, 3> R_;
};

} // namespace PKF

#endif // PKF_LORENZ63_MODEL_HPP
