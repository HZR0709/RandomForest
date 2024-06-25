#ifndef PCA_H
#define PCA_H

#include <vector>
#include <Eigen/Dense>

class PCA {
public:
    PCA(int num_components);
    void fit(const std::vector<std::vector<double>>& data);
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& data);

private:
    int num_components;
    Eigen::MatrixXd components;
    Eigen::VectorXd mean;
};

#endif // PCA_H
