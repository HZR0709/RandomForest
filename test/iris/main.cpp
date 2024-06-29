#include "RandomForest.h"
#include "PCA.h"
#include "DataAugmentation.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// ����ǩ���ַ���ӳ�䵽����
int label_to_int(const std::string& label) {
    if (label == "Iris-setosa") return 0;
    if (label == "Iris-versicolor") return 1;
    if (label == "Iris-virginica") return 2;
    throw std::invalid_argument("Unknown label: " + label);
}

// ��������
std::vector<std::vector<double>> load_data(const std::string& filename, std::vector<int>& labels) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line, cell;

    while (std::getline(file, line)) {
        std::vector<double> sample;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ',')) {
            if (cell.find_first_not_of("0123456789.-") == std::string::npos) {
                sample.push_back(std::stod(cell));
            }
            else {
                labels.push_back(label_to_int(cell));
            }
        }
        if (!sample.empty()) {
            data.push_back(sample);
        }
    }

    return data;
}

int main() {
     // �������ݼ�
    std::vector<int> labels;
    std::vector<std::vector<double>> data = load_data("D:/���ݼ�/iris/iris.data", labels);

    // ������ݼ��Ƿ�Ϊ��
    if (data.empty() || labels.empty()) {
        std::cerr << "���ݼ�����ʧ�ܻ�Ϊ��" << std::endl;
        return -1;
    }
    std::cout << "Loaded " << data.size() << " samples with " << data[0].size() << " features each." << std::endl;

    // ������ݺͱ�ǩ�Ƿ�ƥ��
    if (data.size() != labels.size()) {
        std::cerr << "Mismatch between data samples and labels." << std::endl;
        return -1;
    }

    // ��ӡһЩ���������ͱ�ǩ
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
        std::cout << "Sample " << i << ": ";
        for (double feature : data[i]) {
            std::cout << feature << " ";
        }
        std::cout << "Label: " << labels[i] << std::endl;
    }

    // ������ǿ
    DataAugmentation::add_random_noise(data, 0.1);
    DataAugmentation::scale(data, 1.2);
    DataAugmentation::shift(data, 0.5);
    DataAugmentation::rotate(data, 15.0); // ��ת15��
    DataAugmentation::crop(data, 0.1); // ����ü�10%

    // ʹ��PCA���н�ά
    PCA pca(2); // ���轵��2ά
    pca.fit(data);
    std::vector<std::vector<double>> transformed_data = pca.transform(data);

    // ʹ��RandomForest����ѵ��������
    RandomForest::grid_search(transformed_data, labels);
    RandomForest forest(10, 10, 2, 1); // ʹ����ѳ�����
    forest.cross_validate(transformed_data, labels, 5);
    forest.train(transformed_data, labels);

    // Ԥ�ⲿ��
    std::vector<double> sample = { 4.9, 3.0, 1.4, 0.2 }; // Ԥ������������4����������ԭʼ����һ��
    std::vector<std::vector<double>> sample_data = { sample };
    std::vector<std::vector<double>> transformed_sample = pca.transform(sample_data);

    // ȷ�� transformed_sample �ǿղ���ά����ȷ
    if (transformed_sample.empty() || transformed_sample[0].size() != 2) {
        std::cerr << "PCA ת��ʧ��" << std::endl;
        return -1;
    }

    int prediction = forest.predict(transformed_sample[0]);

    std::cout << "Prediction: " << prediction << std::endl;

    double acc = forest.accuracy(transformed_data, labels);
    std::cout << "Accuracy: " << acc << std::endl;

    // ���������������
    forest.print_confusion_matrix(transformed_data, labels);
    forest.calculate_metrics(transformed_data, labels);

    return 0;
}

