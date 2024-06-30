#include "Stacking.h"
#include <vector>

// ���캯������ʼ��RandomForest��KNNģ��
Stacking::Stacking(int num_trees, int max_depth, int k) : rf(num_trees, max_depth), knn(k) {}

// ѵ��������ʹ�����ݺͱ�ǩѵ��RF��KNNģ��
void Stacking::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    rf.train(data, labels); // ѵ��RandomForestģ��
    knn.train(data, labels);// ѵ��KNNģ��
}

// Ԥ�ⷽ�����Ը�����������Ԥ��
int Stacking::predict(const std::vector<double>& sample) const {
    std::vector<double> meta_feat = meta_features(sample);  // ��ȡԪ����
    // ��ͶƱ���ƣ�����ͶƱ���
    return meta_feat[0] > meta_feat[1] ? 0 : 1;
}

// ��ȡԪ�����ķ���������RF��KNN��Ԥ����
std::vector<double> Stacking::meta_features(const std::vector<double>& sample) const {
    std::vector<double> features(2);
    features[0] = rf.predict(sample);   // RandomForestԤ����
    features[1] = knn.predict(sample);  // KNNԤ����
    return features;
}
