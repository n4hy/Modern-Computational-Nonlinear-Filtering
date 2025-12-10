#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;

void save_to_csv(const std::string& filename,
                 double dt,
                 const std::vector<VectorXd>& true_hist,
                 const std::vector<VectorXd>& filt_hist,
                 const std::vector<VectorXd>& smooth_hist,
                 const std::vector<MatrixXd>& filt_cov_hist,
                 const std::vector<MatrixXd>& smooth_cov_hist) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    int n = true_hist[0].size();

    file << "t";
    for(int i=0; i<n; ++i) file << ",x_true_" << i;
    for(int i=0; i<n; ++i) file << ",x_filt_" << i;
    for(int i=0; i<n; ++i) file << ",P_filt_" << i;
    for(int i=0; i<n; ++i) file << ",x_smooth_" << i;
    for(int i=0; i<n; ++i) file << ",P_smooth_" << i;
    file << "\n";

    size_t rows = std::min(true_hist.size(), smooth_hist.size());

    for (size_t i = 0; i < rows; ++i) {
        double t = i * dt;
        file << t;

        for(int j=0; j<n; ++j) file << "," << true_hist[i](j);

        if (i < filt_hist.size()) {
            for(int j=0; j<n; ++j) file << "," << filt_hist[i](j);
            if (i < filt_cov_hist.size()) {
                for(int j=0; j<n; ++j) file << "," << filt_cov_hist[i](j, j);
            } else {
                for(int j=0; j<n; ++j) file << ",0";
            }
        } else {
            for(int j=0; j<n; ++j) file << ",0";
            for(int j=0; j<n; ++j) file << ",0";
        }

        if (i < smooth_hist.size()) {
            for(int j=0; j<n; ++j) file << "," << smooth_hist[i](j);
            if (i < smooth_cov_hist.size()) {
                for(int j=0; j<n; ++j) file << "," << smooth_cov_hist[i](j, j);
            } else {
                for(int j=0; j<n; ++j) file << ",0";
            }
        } else {
            for(int j=0; j<n; ++j) file << ",0";
            for(int j=0; j<n; ++j) file << ",0";
        }

        file << "\n";
    }
    file.close();
    std::cout << "Data exported to " << filename << std::endl;
}

#endif // FILE_UTILS_H
