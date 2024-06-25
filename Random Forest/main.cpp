#include "RandomForest.h"
#include "PCA.h"
#include <iostream>
#include <vector>

void test_decision_tree() {
    DecisionTree tree(5);
    std::vector<std::vector<double>> data = {
        {2.3, 4.5, 1.2},
        {1.3, 3.4, 2.1},
        {3.1, 5.6, 1.5},
        {2.8, 4.8, 1.4},
        {1.5, 3.2, 2.0},
        {3.2, 4.5, 1.7},
        {1.2, 3.1, 1.9},
        {2.6, 4.0, 1.3},
        {3.0, 5.0, 1.6},
        {2.0, 3.5, 2.2},
    };
    std::vector<int> labels = { 0, 1, 0, 0, 1, 0, 1, 0, 0, 1 };

    tree.train(data, labels);
    std::cout << "Decision Tree Test:" << std::endl;
    std::cout << "Expected: 0, Predicted: " << tree.predict(data[0]) << std::endl;
    std::cout << "Expected: 1, Predicted: " << tree.predict(data[1]) << std::endl;
}

void test_random_forest() {
    RandomForest forest(10, 5);
    std::vector<std::vector<double>> data = {
        {2.3, 4.5, 1.2},
        {1.3, 3.4, 2.1},
        {3.1, 5.6, 1.5},
        {2.8, 4.8, 1.4},
        {1.5, 3.2, 2.0},
        {3.2, 4.5, 1.7},
        {1.2, 3.1, 1.9},
        {2.6, 4.0, 1.3},
        {3.0, 5.0, 1.6},
        {2.0, 3.5, 2.2},
    };
    std::vector<int> labels = { 0, 1, 0, 0, 1, 0, 1, 0, 0, 1 };

    forest.train(data, labels);
    std::cout << "Random Forest Test:" << std::endl;
    std::cout << "Expected: 0, Predicted: " << forest.predict(data[0]) << std::endl;
    std::cout << "Expected: 1, Predicted: " << forest.predict(data[1]) << std::endl;
}


int main() {
    // 假设我们有一个更大的数据集
    std::vector<std::vector<double>> data = {
        {2.3, 4.5, 1.2}, {1.3, 3.4, 2.1}, {3.1, 5.6, 1.5}, {2.8, 4.8, 1.4}, {1.5, 3.2, 2.0},
        {3.2, 4.5, 1.7}, {1.2, 3.1, 1.9}, {2.6, 4.0, 1.3}, {3.0, 5.0, 1.6}, {2.0, 3.5, 2.2},
        {3.3, 5.1, 1.9}, {1.9, 3.3, 2.2}, {3.2, 5.5, 1.4}, {2.4, 4.7, 1.5}, {1.6, 3.4, 2.3},
        {3.0, 4.8, 1.8}, {1.1, 3.0, 1.7}, {2.9, 4.1, 1.6}, {2.8, 5.3, 1.2}, {2.1, 3.7, 2.1},
        {2.4, 4.6, 1.9}, {1.7, 3.5, 2.3}, {3.1, 5.2, 1.6}, {2.5, 4.9, 1.5}, {1.6, 3.3, 2.1},
        {3.3, 5.4, 1.8}, {1.2, 3.0, 1.9}, {2.7, 4.3, 1.4}, {3.0, 5.1, 1.7}, {2.2, 3.6, 2.0},
        // 更多数据
    };
    std::vector<int> labels = {
        0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
        // 更多标签
    };

    // 使用PCA进行降维
    PCA pca(2); // 假设降到2维
    pca.fit(data);
    std::vector<std::vector<double>> transformed_data = pca.transform(data);

    // 使用RandomForest进行训练和评估
    RandomForest::grid_search(transformed_data, labels);
    RandomForest forest(10, 10, 2, 1); // 使用最佳超参数
    forest.cross_validate(transformed_data, labels, 5);
    forest.train(transformed_data, labels);

    std::vector<double> sample = { 2.1, 4.3, 1.8 };
    std::vector<std::vector<double>> sample_data = { sample };
    std::vector<std::vector<double>> transformed_sample = pca.transform(sample_data);
    int prediction = forest.predict(transformed_sample[0]);

    std::cout << "Prediction: " << prediction << std::endl;

    double acc = forest.accuracy(transformed_data, labels);
    std::cout << "Accuracy: " << acc << std::endl;

    forest.print_confusion_matrix(transformed_data, labels);
    forest.calculate_metrics(transformed_data, labels);

    return 0;
}
