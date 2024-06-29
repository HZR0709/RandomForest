#include "Stacking.h"
#include <vector>

Stacking::Stacking(int num_trees, int max_depth, int k) : rf(num_trees, max_depth), knn(k) {}

void Stacking::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    rf.train(data, labels);
    knn.train(data, labels);
}

int Stacking::predict(const std::vector<double>& sample) const {
    std::vector<double> meta_feat = meta_features(sample);
    // 简单投票机制
    return meta_feat[0] > meta_feat[1] ? 0 : 1;
}

std::vector<double> Stacking::meta_features(const std::vector<double>& sample) const {
    std::vector<double> features(2);
    features[0] = rf.predict(sample);
    features[1] = knn.predict(sample);
    return features;
}
