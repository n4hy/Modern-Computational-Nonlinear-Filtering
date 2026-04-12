#ifndef PKF_NOISE_MODELS_HPP
#define PKF_NOISE_MODELS_HPP

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <numbers>

namespace PKF {

namespace Noise {

    /** Thin wrapper around std::lgamma for readability in distribution formulas. */
    inline float log_gamma(float x) {
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
     * @return float Log probability density
     */
    template<int Dim>
    float student_t_logpdf(const Eigen::Matrix<float, Dim, 1>& x,
                            const Eigen::Matrix<float, Dim, 1>& mu,
                            const Eigen::Matrix<float, Dim, Dim>& cov,
                            float nu) {

        using Vector = Eigen::Matrix<float, Dim, 1>;

        float p = static_cast<float>(Dim);
        Vector diff = x - mu;

        // Use LLT to compute inverse and determinant
        Eigen::LLT<Eigen::Matrix<float, Dim, Dim>> llt(cov);
        if (llt.info() != Eigen::Success)
            return -std::numeric_limits<float>::infinity();

        // Mahalanobis distance squared: (x-mu)^T * Sigma^{-1} * (x-mu)
        Vector y = llt.solve(diff);
        float mahalanobis_sq = diff.dot(y);

        // Log determinant of Sigma
        // LLT L matrix determinant is product of diagonal elements.
        // det(Sigma) = det(L) * det(L^T) = det(L)^2
        float log_det_cov = 0.0f;
        Eigen::Matrix<float, Dim, Dim> L = llt.matrixL();
        for (int i = 0; i < Dim; ++i) {
            log_det_cov += std::log(L(i, i));
        }
        log_det_cov *= 2.0f;

        float log_numerator = std::lgamma((nu + p) / 2.0f);
        float log_denominator = std::lgamma(nu / 2.0f) + (p / 2.0f) * std::log(nu * std::numbers::pi_v<float>) + 0.5f * log_det_cov;

        float log_term = -((nu + p) / 2.0f) * std::log(1.0f + mahalanobis_sq / nu);

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
     * @return Eigen::Matrix<float, Dim, 1> Sample
     */
    template<int Dim>
    Eigen::Matrix<float, Dim, 1> student_t_sample(const Eigen::Matrix<float, Dim, 1>& mu,
                                                   const Eigen::Matrix<float, Dim, Dim>& cov,
                                                   float nu,
                                                   std::mt19937_64& rng) {

        // Sample u ~ ChiSquared(nu)
        // ChiSquared(nu) is Gamma(nu/2, 2)
        std::chi_squared_distribution<float> chi_dist(nu);
        float u = chi_dist(rng);

        // Sample z ~ N(0, Sigma)
        // z = L * standard_normal
        Eigen::LLT<Eigen::Matrix<float, Dim, Dim>> llt(cov);
        if (llt.info() != Eigen::Success)
            return mu;  // Fallback: return mean if decomposition fails

        Eigen::Matrix<float, Dim, Dim> L = llt.matrixL();

        std::normal_distribution<float> norm_dist(0.0f, 1.0f);
        Eigen::Matrix<float, Dim, 1> standard_normal_vec;
        for (int i = 0; i < Dim; ++i) {
            standard_normal_vec(i) = norm_dist(rng);
        }

        Eigen::Matrix<float, Dim, 1> z = L * standard_normal_vec;

        // Combine
        // x = mu + z * sqrt(nu / u)
        return mu + z * std::sqrt(nu / u);
    }

    /**
     * @brief Multivariate Gaussian Log-PDF (for completeness/reference)
     */
    template<int Dim>
    float gaussian_logpdf(const Eigen::Matrix<float, Dim, 1>& x,
                           const Eigen::Matrix<float, Dim, 1>& mu,
                           const Eigen::Matrix<float, Dim, Dim>& cov) {
        float p = static_cast<float>(Dim);
        Eigen::Matrix<float, Dim, 1> diff = x - mu;

        Eigen::LLT<Eigen::Matrix<float, Dim, Dim>> llt(cov);
        if (llt.info() != Eigen::Success)
            return -std::numeric_limits<float>::infinity();

        Eigen::Matrix<float, Dim, 1> y = llt.solve(diff);
        float mahalanobis_sq = diff.dot(y);

        float log_det_cov = 0.0f;
        Eigen::Matrix<float, Dim, Dim> L = llt.matrixL();
        for (int i = 0; i < Dim; ++i) {
            log_det_cov += std::log(L(i, i));
        }
        log_det_cov *= 2.0f;

        return -0.5f * (p * std::log(2.0f * std::numbers::pi_v<float>) + log_det_cov + mahalanobis_sq);
    }

    /**
     * @brief Sample from Multivariate Gaussian
     */
    template<int Dim>
    Eigen::Matrix<float, Dim, 1> gaussian_sample(const Eigen::Matrix<float, Dim, 1>& mu,
                                                  const Eigen::Matrix<float, Dim, Dim>& cov,
                                                  std::mt19937_64& rng) {
        Eigen::LLT<Eigen::Matrix<float, Dim, Dim>> llt(cov);
        if (llt.info() != Eigen::Success)
            return mu;  // Fallback: return mean if decomposition fails

        Eigen::Matrix<float, Dim, Dim> L = llt.matrixL();

        std::normal_distribution<float> norm_dist(0.0f, 1.0f);
        Eigen::Matrix<float, Dim, 1> standard_normal_vec;
        for (int i = 0; i < Dim; ++i) {
            standard_normal_vec(i) = norm_dist(rng);
        }

        return mu + L * standard_normal_vec;
    }

} // namespace Noise
} // namespace PKF

#endif // PKF_NOISE_MODELS_HPP
