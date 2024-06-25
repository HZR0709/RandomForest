#include "KNN.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    train_data = data;
    train_labels = labels;
}

int KNN::predict(const std::vector<double>& sample) const {
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < train_data.size(); ++i) {
        double dist = distance(sample, train_data[i]);
        distances.push_back(std::make_pair(dist, train_labels[i]));
    }
    std::sort(distances.begin(), distances.end());

    std::vector<int> k_nearest_labels;
    for (int i = 0; i < k; ++i) {
        k_nearest_labels.push_back(distances[i].second);
    }

    std::unordered_map<int, int> label_counts;
    for (int label : k_nearest_labels) {
        label_counts[label]++;
    }

    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;
}

double KNN::distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}
