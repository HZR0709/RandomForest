#include "Stacking.h"
#include <vector>

// 构造函数：初始化RandomForest和KNN模型
Stacking::Stacking(int num_trees, int max_depth, int k) : rf(num_trees, max_depth), knn(k) {}

// 训练方法：使用数据和标签训练RF和KNN模型
void Stacking::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    rf.train(data, labels); // 训练RandomForest模型
    knn.train(data, labels);// 训练KNN模型
}

// 预测方法：对给定样本进行预测
int Stacking::predict(const std::vector<double>& sample) const {
    std::vector<double> meta_feat = meta_features(sample);  // 获取元特征
    // 简单投票机制，返回投票结果
    return meta_feat[0] > meta_feat[1] ? 0 : 1;
}

// 获取元特征的方法：返回RF和KNN的预测结果
std::vector<double> Stacking::meta_features(const std::vector<double>& sample) const {
    std::vector<double> features(2);
    features[0] = rf.predict(sample);   // RandomForest预测结果
    features[1] = knn.predict(sample);  // KNN预测结果
    return features;
}
