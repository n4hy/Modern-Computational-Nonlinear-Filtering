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

        // Generate measurement
        DragBallModel::Observation y = model.h(x, k * model.dt);
        y(0) += d_meas(gen);
        y(1) += d_meas(gen);
        measurements.push_back(y);

        // Evolve state
        // Add random wind gust to velocity?
        DragBallModel::State u = DragBallModel::State::Zero(); // No control
        x = model.f(x, k * model.dt, u);

        // Add process noise (simulated)
        x(2) += d_proc(gen) * std::sqrt(model.dt);
        x(3) += d_proc(gen) * std::sqrt(model.dt);
    }
}

int main() {
    // 1. Setup Model
    DragBallModel model; // defaults: dt=0.1, drag=0.001
    int steps = 200;
    int lag = 10;

    std::cout << "Generating " << steps << " steps of simulation..." << std::endl;
    std::vector<DragBallModel::State> true_states;
    std::vector<DragBallModel::Observation> measurements;
    generate_data(model, steps, true_states, measurements);

    // 2. Setup Smoother
    // Initial estimate
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

    for (int k = 0; k < steps; ++k) {
        double t = k * model.dt;

        // Step
        smoother.step(t, measurements[k], u_dummy);

        // Get Estimates
        DragBallModel::State x_filt = smoother.get_filtered_state();
        DragBallModel::State x_smooth = smoother.get_smoothed_state(std::min(k, lag));
        // Note: For quantitative comparison at time k-L, we should compare:
        // x_smooth from lag L (which corresponds to time k-L) vs true_state[k-L].
        // But for visualization, we can just dump what we have.
        // Let's compute RMSE for the *current* time k (Filtered)
        // and for the time k-lag (Smoothed).

        DragBallModel::State x_true = true_states[k];

        double err_fx = x_filt(0) - x_true(0);
        double err_fy = x_filt(1) - x_true(1);
        rmse_filt_pos += err_fx*err_fx + err_fy*err_fy;

        // Smoothed error: Look back L steps
        if (k >= lag) {
            DragBallModel::State x_s_lagged = smoother.get_smoothed_state(lag); // Estimate for k-lag
            DragBallModel::State x_true_lagged = true_states[k - lag];

            double err_sx = x_s_lagged(0) - x_true_lagged(0);
            double err_sy = x_s_lagged(1) - x_true_lagged(1);
            rmse_smooth_pos += err_sx*err_sx + err_sy*err_sy;
            count++;

             // Log for csv (using the smoothed estimate for k-lag)
             // To make lines align in CSV, let's just log "current filtered" and "smoothed at current time if available"?
             // Actually, usually we plot the entire trajectory.
             // At step k, we have a smoothed estimate for k-lag.
        }

        // For CSV, let's log the *current* filtered and the *most refined* smoothed value for the current time?
        // No, at time k, the best smoothed value for k is just x_filt (lag 0).
        // The value for k will improve as we step forward.
        // So a CSV generated on the fly is tricky for a smoother.
        // We typically dump the final smoothed trajectory after the run.
        // But since this is a "Fixed Lag" online algorithm, at step k we finalize the estimate for k-L.
        // So let's log the value for k-L.

        if (k >= lag) {
             DragBallModel::State x_s = smoother.get_smoothed_state(lag);
             DragBallModel::State x_t = true_states[k - lag];
             DragBallModel::Observation m_t = measurements[k - lag];
             // Filtered value for k-L was computed L steps ago. We don't have it easily here unless we cached it.
             // But we can log the current filtered value x_filt (at k) and x_true (at k).
             // And separate columns for "Finalized Smoothed at k-L".

             out_file << (k-lag) << ","
                      << x_t(0) << "," << x_t(1) << "," << x_t(2) << "," << x_t(3) << ","
                      << m_t(0) << "," << m_t(1) << ","
                      << "0,0,0,0," // We skip logging filtered for k-L here to save complexity
                      << x_s(0) << "," << x_s(1) << "," << x_s(2) << "," << x_s(3) << "\n";
        }
    }

    out_file.close();

    rmse_filt_pos = std::sqrt(rmse_filt_pos / steps); // This is RMSE of filtered at all steps
    rmse_smooth_pos = std::sqrt(rmse_smooth_pos / count); // This is RMSE of smoothed at lags

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
