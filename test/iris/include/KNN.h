#ifndef KNN_H
#define KNN_H

#include <vector>

class KNN {
public:
    KNN(int k);
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
    int predict(const std::vector<double>& sample) const;

private:
    int k;
    std::vector<std::vector<double>> train_data;
    std::vector<int> train_labels;
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;
};

#endif // KNN_H
