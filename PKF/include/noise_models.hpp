#ifndef PKF_NOISE_MODELS_HPP
#define PKF_NOISE_MODELS_HPP

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <numbers>

namespace PKF {

namespace Noise {

    // Helper for Log Gamma function if std::lgamma is not enough or for clarity
    inline double log_gamma(double x) {
        return std::lgamma(x);
    }

    /**
     * @brief Multivariate Student-t Log-PDF
     *
     * log p(x) = log_const - 0.5 * (nu + p) * log(1 + (1/nu) * (x - mu)^T * Sigma^{-1} * (x - mu))
     *
     * @tparam Dim Dimension of the vector
     * @param x Sample vector
     * @param mu Mean vector
     * @param cov Scale matrix (Sigma)
     * @param nu Degrees of freedom
     * @return double Log probability density
     */
    template<int Dim>
    double student_t_logpdf(const Eigen::Matrix<double, Dim, 1>& x,
                            const Eigen::Matrix<double, Dim, 1>& mu,
                            const Eigen::Matrix<double, Dim, Dim>& cov,
                            double nu) {

        using Vector = Eigen::Matrix<double, Dim, 1>;

        double p = static_cast<double>(Dim);
        Vector diff = x - mu;

        // Use LLT to compute inverse and determinant
        // If cov is diagonal, we could optimize, but this is general.
        Eigen::LLT<Eigen::Matrix<double, Dim, Dim>> llt(cov);

        // Mahalanobis distance squared: (x-mu)^T * Sigma^{-1} * (x-mu)
        // We solve Sigma * y = diff, so y = Sigma^{-1} * diff.
        // Then diff.dot(y)
        Vector y = llt.solve(diff);
        double mahalanobis_sq = diff.dot(y);

        // Log determinant of Sigma
        // LLT L matrix determinant is product of diagonal elements.
        // det(Sigma) = det(L) * det(L^T) = det(L)^2
        double log_det_cov = 0.0;
        Eigen::Matrix<double, Dim, Dim> L = llt.matrixL();
        for (int i = 0; i < Dim; ++i) {
            log_det_cov += std::log(L(i, i));
        }
        log_det_cov *= 2.0;

        double log_numerator = std::lgamma((nu + p) / 2.0);
        double log_denominator = std::lgamma(nu / 2.0) + (p / 2.0) * std::log(nu * std::numbers::pi) + 0.5 * log_det_cov;

        double log_term = -((nu + p) / 2.0) * std::log(1.0 + mahalanobis_sq / nu);

        return log_numerator - log_denominator + log_term;
    }

    /**
     * @brief Sample from Multivariate Student-t distribution
     *
     * x = mu + sqrt(nu / u) * z
     * where u ~ ChiSq(nu) and z ~ N(0, Sigma)
     *
     * @tparam Dim Dimension
     * @param mu Mean vector
     * @param cov Scale matrix (Sigma)
     * @param nu Degrees of freedom
     * @param rng Random number generator
     * @return Eigen::Matrix<double, Dim, 1> Sample
     */
    template<int Dim>
    Eigen::Matrix<double, Dim, 1> student_t_sample(const Eigen::Matrix<double, Dim, 1>& mu,
                                                   const Eigen::Matrix<double, Dim, Dim>& cov,
                                                   double nu,
                                                   std::mt19937_64& rng) {

        // Sample u ~ ChiSquared(nu)
        // ChiSquared(nu) is Gamma(nu/2, 2)
        std::chi_squared_distribution<double> chi_dist(nu);
        double u = chi_dist(rng);

        // Sample z ~ N(0, Sigma)
        // z = L * standard_normal
        Eigen::LLT<Eigen::Matrix<double, Dim, Dim>> llt(cov);
        Eigen::Matrix<double, Dim, Dim> L = llt.matrixL();

        std::normal_distribution<double> norm_dist(0.0, 1.0);
        Eigen::Matrix<double, Dim, 1> standard_normal_vec;
        for (int i = 0; i < Dim; ++i) {
            standard_normal_vec(i) = norm_dist(rng);
        }

        Eigen::Matrix<double, Dim, 1> z = L * standard_normal_vec;

        // Combine
        // x = mu + z * sqrt(nu / u)
        return mu + z * std::sqrt(nu / u);
    }

    /**
     * @brief Multivariate Gaussian Log-PDF (for completeness/reference)
     */
    template<int Dim>
    double gaussian_logpdf(const Eigen::Matrix<double, Dim, 1>& x,
                           const Eigen::Matrix<double, Dim, 1>& mu,
                           const Eigen::Matrix<double, Dim, Dim>& cov) {
        double p = static_cast<double>(Dim);
        Eigen::Matrix<double, Dim, 1> diff = x - mu;

        Eigen::LLT<Eigen::Matrix<double, Dim, Dim>> llt(cov);
        Eigen::Matrix<double, Dim, 1> y = llt.solve(diff);
        double mahalanobis_sq = diff.dot(y);

        double log_det_cov = 0.0;
        Eigen::Matrix<double, Dim, Dim> L = llt.matrixL();
        for (int i = 0; i < Dim; ++i) {
            log_det_cov += std::log(L(i, i));
        }
        log_det_cov *= 2.0;

        return -0.5 * (p * std::log(2 * std::numbers::pi) + log_det_cov + mahalanobis_sq);
    }

    /**
     * @brief Sample from Multivariate Gaussian
     */
    template<int Dim>
    Eigen::Matrix<double, Dim, 1> gaussian_sample(const Eigen::Matrix<double, Dim, 1>& mu,
                                                  const Eigen::Matrix<double, Dim, Dim>& cov,
                                                  std::mt19937_64& rng) {
        Eigen::LLT<Eigen::Matrix<double, Dim, Dim>> llt(cov);
        Eigen::Matrix<double, Dim, Dim> L = llt.matrixL();

        std::normal_distribution<double> norm_dist(0.0, 1.0);
        Eigen::Matrix<double, Dim, 1> standard_normal_vec;
        for (int i = 0; i < Dim; ++i) {
            standard_normal_vec(i) = norm_dist(rng);
        }

        return mu + L * standard_normal_vec;
    }

} // namespace Noise
} // namespace PKF

#endif // PKF_NOISE_MODELS_HPP
