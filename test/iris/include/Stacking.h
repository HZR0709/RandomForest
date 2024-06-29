#ifndef STACKING_H
#define STACKING_H

#include <vector>
#include "RandomForest.h"
#include "KNN.h"

class Stacking {
public:
    Stacking(int num_trees, int max_depth, int k);
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
    int predict(const std::vector<double>& sample) const;

private:
    RandomForest rf;
    KNN knn;
    std::vector<double> meta_features(const std::vector<double>& sample) const;
};

#endif // STACKING_H
