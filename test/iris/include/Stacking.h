#ifndef STACKING_H
#define STACKING_H

#include <vector>
#include "RandomForest.h"
#include "KNN.h"

// Stacking�ࣺʵ�ֶѵ�����ѧϰģ�ͣ���RandomForest��KNN���ʹ��
class Stacking {
public:
    // ���캯������ʼ��RandomForest��KNNģ��
    Stacking(int num_trees, int max_depth, int k);      

    // ѵ��������ʹ�ø��������ݺͱ�ǩѵ��RF��KNNģ��
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // Ԥ�ⷽ�����Ը�����������Ԥ�⣬����Ԥ���ǩ
    int predict(const std::vector<double>& sample) const;

private:
    RandomForest rf;    // RandomForestģ��
    KNN knn;            // KNNģ��

    // ��ȡԪ�����ķ���������RF��KNN��Ԥ����
    std::vector<double> meta_features(const std::vector<double>& sample) const;
};

#endif // STACKING_H
