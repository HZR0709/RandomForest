#include "RandomForest.h"
#include "PCA.h"
#include "DataAugmentation.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// 将标签从字符串映射到整数
int label_to_int(const std::string& label) {
    if (label == "Iris-setosa") return 0;
    if (label == "Iris-versicolor") return 1;
    if (label == "Iris-virginica") return 2;
    throw std::invalid_argument("Unknown label: " + label);
}

// 加载数据
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
     // 加载数据集
    std::vector<int> labels;
    std::vector<std::vector<double>> data = load_data("D:/数据集/iris/iris.data", labels);

    // 检查数据集是否为空
    if (data.empty() || labels.empty()) {
        std::cerr << "数据集加载失败或为空" << std::endl;
        return -1;
    }
    std::cout << "Loaded " << data.size() << " samples with " << data[0].size() << " features each." << std::endl;

    // 检查数据和标签是否匹配
    if (data.size() != labels.size()) {
        std::cerr << "Mismatch between data samples and labels." << std::endl;
        return -1;
    }

    // 打印一些数据样本和标签
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
        std::cout << "Sample " << i << ": ";
        for (double feature : data[i]) {
            std::cout << feature << " ";
        }
        std::cout << "Label: " << labels[i] << std::endl;
    }

    // 数据增强
    DataAugmentation::add_random_noise(data, 0.1);
    DataAugmentation::scale(data, 1.2);
    DataAugmentation::shift(data, 0.5);
    DataAugmentation::rotate(data, 15.0); // 旋转15度
    DataAugmentation::crop(data, 0.1); // 随机裁剪10%

    // 使用PCA进行降维
    PCA pca(2); // 假设降到2维
    pca.fit(data);
    std::vector<std::vector<double>> transformed_data = pca.transform(data);

    // 使用RandomForest进行训练和评估
    RandomForest::grid_search(transformed_data, labels);
    RandomForest forest(10, 10, 2, 1); // 使用最佳超参数
    forest.cross_validate(transformed_data, labels, 5);
    forest.train(transformed_data, labels);

    // 预测部分
    std::vector<double> sample = { 4.9, 3.0, 1.4, 0.2 }; // 预测样本必须有4个特征，与原始数据一致
    std::vector<std::vector<double>> sample_data = { sample };
    std::vector<std::vector<double>> transformed_sample = pca.transform(sample_data);

    // 确保 transformed_sample 非空并且维度正确
    if (transformed_sample.empty() || transformed_sample[0].size() != 2) {
        std::cerr << "PCA 转换失败" << std::endl;
        return -1;
    }

    int prediction = forest.predict(transformed_sample[0]);

    std::cout << "Prediction: " << prediction << std::endl;

    double acc = forest.accuracy(transformed_data, labels);
    std::cout << "Accuracy: " << acc << std::endl;

    // 混淆矩阵计算和输出
    forest.print_confusion_matrix(transformed_data, labels);
    forest.calculate_metrics(transformed_data, labels);

    return 0;
}

