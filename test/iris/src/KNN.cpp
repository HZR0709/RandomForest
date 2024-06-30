#include "KNN.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// 构造函数：初始化k值
KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    train_data = data;
    train_labels = labels;
}

// 训练方法：保存训练数据和标签
int KNN::predict(const std::vector<double>& sample) const {

    // 计算样本与所有训练数据的距离
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < train_data.size(); ++i) {
        double dist = distance(sample, train_data[i]);
        distances.push_back(std::make_pair(dist, train_labels[i]));
    }

    // 按距离排序
    std::sort(distances.begin(), distances.end());

    // 选取最近的k个邻居
    std::vector<int> k_nearest_labels;
    for (int i = 0; i < k; ++i) {
        k_nearest_labels.push_back(distances[i].second);
    }

    // 统计每个标签的出现次数
    std::unordered_map<int, int> label_counts;
    for (int label : k_nearest_labels) {
        label_counts[label]++;
    }

    // 返回出现次数最多的标签
    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;
}

// 计算两个样本之间的欧几里得距离
double KNN::distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}
