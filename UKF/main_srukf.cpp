#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>

#include "SRUKF.h"
#include "SRUKFFixedLagSmoother.h"
#include "DragBallModel.h"

using namespace UKFCore;

int main() {
    std::cout << "=== SRUKF Test: Drag Ball Model ===" << std::endl;

    constexpr int NX = 4;
    constexpr int NY = 2;

    // Model
    DragBallModel model;

    // Initial state
    Eigen::Matrix<float, NX, 1> x0;
    x0 << 0.0f, 0.0f, 10.0f, 15.0f;  // [x, y, vx, vy]

    // Initial covariance
    Eigen::Matrix<float, NX, NX> P0 = Eigen::Matrix<float, NX, NX>::Identity();
    P0(0, 0) = 1.0f;
    P0(1, 1) = 1.0f;
    P0(2, 2) = 2.0f;
    P0(3, 3) = 2.0f;

    // Simulation parameters
    float dt = 0.01f;
    float T_final = 5.0f;
    int N = static_cast<int>(T_final / dt);

    // Random number generation
    std::mt19937 gen(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Ground truth trajectory
    std::vector<float> times;
    std::vector<Eigen::Matrix<float, NX, 1>> true_states;
    std::vector<Eigen::Matrix<float, NY, 1>> measurements;

    Eigen::Matrix<float, NX, 1> x_true = x0;
    Eigen::Matrix<float, NX, 1> u = Eigen::Matrix<float, NX, 1>::Zero();

    auto Q = model.Q(0.0f);
    auto R = model.R(0.0f);

    Eigen::LLT<Eigen::Matrix<float, NX, NX>> llt_Q(Q);
    auto L_Q = llt_Q.matrixL();

    Eigen::LLT<Eigen::Matrix<float, NY, NY>> llt_R(R);
    auto L_R = llt_R.matrixL();

    std::cout << "Generating ground truth trajectory..." << std::endl;

    for (int i = 0; i < N; ++i) {
        float t = i * dt;
        times.push_back(t);
        true_states.push_back(x_true);

        // Measurement
        Eigen::Matrix<float, NY, 1> y_true = model.h(x_true, t);
        Eigen::Matrix<float, NY, 1> noise_R;
        for (int j = 0; j < NY; ++j) {
            noise_R(j) = normal(gen);
        }
        Eigen::Matrix<float, NY, 1> y_meas = y_true + L_R * noise_R;
        measurements.push_back(y_meas);

        // Propagate truth
        x_true = model.f(x_true, t, u);

        // Add process noise
        Eigen::Matrix<float, NX, 1> noise_Q;
        for (int j = 0; j < NX; ++j) {
            noise_Q(j) = normal(gen);
        }
        x_true += L_Q * noise_Q;
    }

    std::cout << "Generated " << N << " time steps." << std::endl;

    // ========== Test 1: SRUKF without smoother ==========
    std::cout << "\n=== Running SRUKF (no smoother) ===" << std::endl;

    SRUKF<NX, NY> srukf(model);
    srukf.initialize(x0, P0);

    std::vector<Eigen::Matrix<float, NX, 1>> srukf_states;

    for (int i = 0; i < N; ++i) {
        srukf.predict(times[i], u);
        srukf.update(times[i], measurements[i]);
        srukf_states.push_back(srukf.getState());
    }

    // Compute RMSE
    float rmse = 0.0f;
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<float, NX, 1> error = true_states[i] - srukf_states[i];
        rmse += error.squaredNorm();
    }
    rmse = std::sqrt(rmse / N);
    std::cout << "SRUKF RMSE: " << rmse << std::endl;

    // Save results
    std::ofstream file_srukf("srukf_results.csv");
    file_srukf << "time,true_x,true_y,true_vx,true_vy,srukf_x,srukf_y,srukf_vx,srukf_vy" << std::endl;
    for (int i = 0; i < N; ++i) {
        file_srukf << times[i] << ","
                   << true_states[i](0) << "," << true_states[i](1) << ","
                   << true_states[i](2) << "," << true_states[i](3) << ","
                   << srukf_states[i](0) << "," << srukf_states[i](1) << ","
                   << srukf_states[i](2) << "," << srukf_states[i](3) << std::endl;
    }
    file_srukf.close();
    std::cout << "Saved results to srukf_results.csv" << std::endl;

    // ========== Test 2: SRUKF with smoother ==========
    std::cout << "\n=== Running SRUKF with Fixed-Lag Smoother ===" << std::endl;

    int lag = 50;
    SRUKFFixedLagSmoother<NX, NY> smoother(model, lag);
    smoother.initialize(x0, P0);

    std::vector<Eigen::Matrix<float, NX, 1>> filt_states;
    std::vector<Eigen::Matrix<float, NX, 1>> smooth_states;

    for (int i = 0; i < N; ++i) {
        smoother.step(times[i], measurements[i], u);
        filt_states.push_back(smoother.get_filtered_state());
        smooth_states.push_back(smoother.get_smoothed_state(lag));
    }

    // Compute RMSE for filtered and smoothed
    float rmse_filt = 0.0f;
    float rmse_smooth = 0.0f;
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<float, NX, 1> error_filt = true_states[i] - filt_states[i];
        rmse_filt += error_filt.squaredNorm();

        Eigen::Matrix<float, NX, 1> error_smooth = true_states[i] - smooth_states[i];
        rmse_smooth += error_smooth.squaredNorm();
    }
    rmse_filt = std::sqrt(rmse_filt / N);
    rmse_smooth = std::sqrt(rmse_smooth / N);

    std::cout << "Filtered RMSE: " << rmse_filt << std::endl;
    std::cout << "Smoothed RMSE: " << rmse_smooth << std::endl;
    std::cout << "Improvement: " << ((rmse_filt - rmse_smooth) / rmse_filt * 100.0f) << "%" << std::endl;

    // Save results
    std::ofstream file_smooth("srukf_smoother_results.csv");
    file_smooth << "time,true_x,true_y,true_vx,true_vy,"
                << "filt_x,filt_y,filt_vx,filt_vy,"
                << "smooth_x,smooth_y,smooth_vx,smooth_vy" << std::endl;
    for (int i = 0; i < N; ++i) {
        file_smooth << times[i] << ","
                    << true_states[i](0) << "," << true_states[i](1) << ","
                    << true_states[i](2) << "," << true_states[i](3) << ","
                    << filt_states[i](0) << "," << filt_states[i](1) << ","
                    << filt_states[i](2) << "," << filt_states[i](3) << ","
                    << smooth_states[i](0) << "," << smooth_states[i](1) << ","
                    << smooth_states[i](2) << "," << smooth_states[i](3) << std::endl;
    }
    file_smooth.close();
    std::cout << "Saved results to srukf_smoother_results.csv" << std::endl;

    std::cout << "\n=== SRUKF Test Complete ===" << std::endl;

    return 0;
}
