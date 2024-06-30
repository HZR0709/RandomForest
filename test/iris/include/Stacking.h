#ifndef STACKING_H
#define STACKING_H

#include <vector>
#include "RandomForest.h"
#include "KNN.h"

// Stacking类：实现堆叠集成学习模型，将RandomForest和KNN结合使用
class Stacking {
public:
    // 构造函数，初始化RandomForest和KNN模型
    Stacking(int num_trees, int max_depth, int k);      

    // 训练方法，使用给定的数据和标签训练RF和KNN模型
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // 预测方法，对给定样本进行预测，返回预测标签
    int predict(const std::vector<double>& sample) const;

private:
    RandomForest rf;    // RandomForest模型
    KNN knn;            // KNN模型

    // 获取元特征的方法，返回RF和KNN的预测结果
    std::vector<double> meta_features(const std::vector<double>& sample) const;
};

#endif // STACKING_H
