#include "PCA.h"
#include <Eigen/Eigenvalues>

PCA::PCA(int num_components) : num_components(num_components) {}

void PCA::fit(const std::vector<std::vector<double>>& data) {
    int n_samples = data.size();
    int n_features = data[0].size();

    Eigen::MatrixXd X(n_samples, n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
    }

    mean = X.colwise().mean();
    Eigen::MatrixXd centered = X.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(n_samples - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(cov);
    components = eigen_solver.eigenvectors().rightCols(num_components);
}

std::vector<std::vector<double>> PCA::transform(const std::vector<std::vector<double>>& data) {
    int n_samples = data.size();
    int n_features = data[0].size();

    Eigen::MatrixXd X(n_samples, n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
    }

    Eigen::MatrixXd centered = X.rowwise() - mean.transpose();
    Eigen::MatrixXd transformed = centered * components;

    std::vector<std::vector<double>> result(n_samples, std::vector<double>(num_components));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < num_components; ++j) {
            result[i][j] = transformed(i, j);
        }
    }

    return result;
}
