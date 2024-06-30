#ifndef KNN_H
#define KNN_H

#include <vector>

// KNN类：实现K近邻算法
class KNN {
public:
    // 构造函数，初始化k值
    KNN(int k);

    // 训练方法，使用给定的数据和标签进行训练
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // 预测方法，对给定样本进行预测，返回预测标签
    int predict(const std::vector<double>& sample) const;

private:
    int k;  // k值，表示最近邻的数量
    std::vector<std::vector<double>> train_data;    // 训练数据
    std::vector<int> train_labels;                  // 训练标签

    // 计算样本之间的欧几里得距离
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;
};

#endif // KNN_H
