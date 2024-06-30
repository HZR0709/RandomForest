#ifndef KNN_H
#define KNN_H

#include <vector>

// KNN�ࣺʵ��K�����㷨
class KNN {
public:
    // ���캯������ʼ��kֵ
    KNN(int k);

    // ѵ��������ʹ�ø��������ݺͱ�ǩ����ѵ��
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // Ԥ�ⷽ�����Ը�����������Ԥ�⣬����Ԥ���ǩ
    int predict(const std::vector<double>& sample) const;

private:
    int k;  // kֵ����ʾ����ڵ�����
    std::vector<std::vector<double>> train_data;    // ѵ������
    std::vector<int> train_labels;                  // ѵ����ǩ

    // ��������֮���ŷ����þ���
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;
};

#endif // KNN_H
