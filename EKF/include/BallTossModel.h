#ifndef BALL_TOSS_MODEL_H
#define BALL_TOSS_MODEL_H

#include "SystemModel.h"

/**
 * 3D Ball Toss System.
 * State: [x, y, z, vx, vy, vz]
 * Dynamics: Constant velocity (x,y), Constant acceleration (z)
 * Observations: [x, y, z]
 */
class BallTossModel : public SystemModel {
public:
    BallTossModel(double dt, double q_std, double r_std) 
        : dt_(dt) {
        
        // Initialize matrices
        // F is constant for this linear-ish system (linear dynamics actually)
        // x_k+1 = x_k + v_k*dt + 0.5*a*dt^2 (for z)
        // v_k+1 = v_k + a*dt (for z)
        
        // F:
        // 1 0 0 dt 0  0
        // 0 1 0 0  dt 0
        // 0 0 1 0  0  dt
        // 0 0 0 1  0  0
        // 0 0 0 0  1  0
        // 0 0 0 0  0  1
        
        F_ = Eigen::MatrixXd::Identity(6, 6);
        F_(0, 3) = dt;
        F_(1, 4) = dt;
        F_(2, 5) = dt;

        // H: Observe positions
        // 1 0 0 0 0 0
        // 0 1 0 0 0 0
        // 0 0 1 0 0 0
        H_ = Eigen::MatrixXd::Zero(3, 6);
        H_(0, 0) = 1;
        H_(1, 1) = 1;
        H_(2, 2) = 1;

        // Q: Discrete process noise
        // Assume noise enters via acceleration
        // G = [0.5dt^2; dt] for each dim
        // Simplified: Diagonal Q
        Q_ = Eigen::MatrixXd::Identity(6, 6) * (q_std * q_std);

        // R: Measurement noise
        R_ = Eigen::MatrixXd::Identity(3, 3) * (r_std * r_std);
    }

    Eigen::VectorXd f(const Eigen::VectorXd& x) const override {
        // Linear dynamics + Gravity
        // x_next = F*x + control(gravity)
        Eigen::VectorXd x_next = F_ * x;
        
        // Add gravity to z position and z velocity
        // z_pos += -0.5 * g * dt^2
        // z_vel += -g * dt
        double g = 9.81;
        x_next(2) -= 0.5 * g * dt_ * dt_;
        x_next(5) -= g * dt_;
        
        return x_next;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& x) const override {
        return H_ * x;
    }

    Eigen::MatrixXd F(const Eigen::VectorXd& x) const override {
        return F_;
    }

    Eigen::MatrixXd H(const Eigen::VectorXd& x) const override {
        return H_;
    }

    Eigen::MatrixXd Q() const override { return Q_; }
    Eigen::MatrixXd R() const override { return R_; }
    
    int getStateDim() const override { return 6; }
    int getObsDim() const override { return 3; }

private:
    double dt_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};

#endif // BALL_TOSS_MODEL_H
