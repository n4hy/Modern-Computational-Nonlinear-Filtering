#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include "UKF.h"
#include "UnscentedFixedLagSmoother.h"
#include "DragBallModel.h"

using namespace UKFCore;
using namespace UKFModel;

// Helper to generate noisy data
void generate_data(DragBallModel& model, int steps,
                   std::vector<DragBallModel::State>& true_states,
                   std::vector<DragBallModel::Observation>& measurements) {

    DragBallModel::State x;
    x << 0, 100, 10, 0; // Initial: x=0, y=100, vx=10, vy=0

    true_states.reserve(steps);
    measurements.reserve(steps);

    std::mt19937 gen(42);
    std::normal_distribution<> d_proc(0, model.q_std); // Crude process noise approx
    std::normal_distribution<> d_meas(0, model.r_std);

    for (int k = 0; k < steps; ++k) {
        true_states.push_back(x);

        // Generate measurement for CURRENT state x
        DragBallModel::Observation y = model.h(x, k * model.dt);
        y(0) += d_meas(gen);
        y(1) += d_meas(gen);
        measurements.push_back(y);

        // Evolve state to NEXT step
        DragBallModel::State u = DragBallModel::State::Zero(); // No control
        x = model.f(x, k * model.dt, u);

        // Add process noise (simulated)
        x(2) += d_proc(gen) * std::sqrt(model.dt);
        x(3) += d_proc(gen) * std::sqrt(model.dt);
    }
}

int main() {
    // 1. Setup Model
    DragBallModel model; // defaults: dt=0.1
    int steps = 200;
    int lag = 10;

    std::cout << "Generating " << steps << " steps of simulation..." << std::endl;
    std::vector<DragBallModel::State> true_states;
    std::vector<DragBallModel::Observation> measurements;
    generate_data(model, steps, true_states, measurements);

    // 2. Setup Smoother
    // Initial estimate at k=0 (Start)
    // We use the first measurement or truth to initialize, then start the loop from k=1.

    DragBallModel::State x0 = true_states[0];
    // Add large uncertainty to initial guess
    x0(0) += 5.0; x0(1) -= 5.0; x0(2) += 2.0;

    DragBallModel::StateMat P0 = DragBallModel::StateMat::Identity();
    P0.topLeftCorner(2,2) *= 10.0;
    P0.bottomRightCorner(2,2) *= 5.0;

    UnscentedFixedLagSmoother<4, 2> smoother(model, lag);
    smoother.initialize(x0, P0);

    // 3. Run Loop
    std::cout << "Running UKF Smoother (Lag=" << lag << ")..." << std::endl;

    double rmse_filt_pos = 0.0;
    double rmse_smooth_pos = 0.0;
    int count = 0;

    // File output
    std::ofstream out_file("ukf_results.csv");
    out_file << "k,tx,ty,tvx,tvy,mx,my,fx,fy,fvx,fvy,sx,sy,svx,svy\n";

    DragBallModel::State u_dummy = DragBallModel::State::Zero();

    // Start loop from k=1
    // smoother initialized at k=0.
    // In loop k=1:
    //   step(t, measurements[1]) -> Predict 0->1, Update using meas 1.
    //   Result is estimate for k=1.

    for (int k = 1; k < steps; ++k) {
        double t = k * model.dt;

        // Step (Predict k-1 -> k, Update k)
        smoother.step(t, measurements[k], u_dummy);

        // Current Filtered State (k)
        DragBallModel::State x_filt = smoother.get_filtered_state();
        DragBallModel::State x_true = true_states[k];

        // RMSE Filtered (k)
        double err_fx = x_filt(0) - x_true(0);
        double err_fy = x_filt(1) - x_true(1);
        rmse_filt_pos += err_fx*err_fx + err_fy*err_fy;

        // Smoothed State for (k - lag)
        // If k >= lag, we have a "finalized" smoothed estimate for k-lag.
        if (k >= lag) {
            int k_delayed = k - lag;
            DragBallModel::State x_s = smoother.get_smoothed_state(lag);
            DragBallModel::State x_t_delayed = true_states[k_delayed];

            double err_sx = x_s(0) - x_t_delayed(0);
            double err_sy = x_s(1) - x_t_delayed(1);
            rmse_smooth_pos += err_sx*err_sx + err_sy*err_sy;
            count++;

            // Log Results for k-lag
            DragBallModel::Observation m_delayed = measurements[k_delayed];
            // Note: logging filtered value for k-delayed is tricky as we didn't cache it.
            // We'll leave it 0 or change the CSV format.
            // Let's log 'current filtered' for row k, and 'smoothed' for row k?
            // If we log row 'k-lag', we align everything to k-lag.

            out_file << k_delayed << ","
                     << x_t_delayed(0) << "," << x_t_delayed(1) << "," << x_t_delayed(2) << "," << x_t_delayed(3) << ","
                     << m_delayed(0) << "," << m_delayed(1) << ","
                     << "0,0,0,0," // Filtered for k-lag (missing)
                     << x_s(0) << "," << x_s(1) << "," << x_s(2) << "," << x_s(3) << "\n";
        }
    }

    out_file.close();

    rmse_filt_pos = std::sqrt(rmse_filt_pos / (steps - 1));
    rmse_smooth_pos = std::sqrt(rmse_smooth_pos / count);

    std::cout << "Filtered Position RMSE (Current): " << rmse_filt_pos << std::endl;
    std::cout << "Smoothed Position RMSE (Lagged):  " << rmse_smooth_pos << std::endl;
    std::cout << "Results written to ukf_results.csv" << std::endl;

    if (rmse_smooth_pos < rmse_filt_pos) {
        std::cout << "SUCCESS: Smoothing reduced error." << std::endl;
    } else {
        std::cout << "WARNING: Smoothing did not reduce error (might be short lag or noise tuning)." << std::endl;
    }

    return 0;
}
