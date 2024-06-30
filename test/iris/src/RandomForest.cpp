#include "RandomForest.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <random>
#include <fstream>
#include <string>

// 构造函数：初始化决策树参数和根节点
DecisionTree::DecisionTree(int max_depth, int min_samples_split, int min_samples_leaf)
    : max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf), root(nullptr) {}

// 训练决策树
void DecisionTree::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    root = build_tree(data, labels, 0); // 从根节点开始构建树
}

// 预测样本的类别
int DecisionTree::predict(const std::vector<double>& sample) const {
    return predict_sample(root, sample);    // 从根节点开始预测
}

// 递归构建决策树
TreeNode* DecisionTree::build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth) {
    // 终止条件：达到最大深度、节点纯净或样本数小于最小拆分样本数
    if (depth >= max_depth || is_pure(labels) || data.size() < min_samples_split) {
        TreeNode* leaf = new TreeNode();            // 创建叶节点
        leaf->prediction = majority_vote(labels);   // 叶节点的预测值为多数投票结果
        return leaf;
    }

    int best_feature;
    double best_threshold;
    find_best_split(data, labels, best_feature, best_threshold);    // 找到最佳拆分点

    TreeNode* node = new TreeNode();
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    split_data(data, labels, best_feature, best_threshold, left_data, left_labels, right_data, right_labels);// 数据拆分

    // 检查是否满足最小叶节点样本数
    if (left_data.size() < min_samples_leaf || right_data.size() < min_samples_leaf) {
        TreeNode* leaf = new TreeNode();            // 创建叶节点
        leaf->prediction = majority_vote(labels);   // 叶节点的预测值为多数投票结果
        return leaf;
    }

    // 递归构建左右子树
    node->left = build_tree(left_data, left_labels, depth + 1);
    node->right = build_tree(right_data, right_labels, depth + 1);

    return node;
}

// 根据特征和阈值拆分数据
void DecisionTree::split_data(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold,
    std::vector<std::vector<double>>& left_data, std::vector<int>& left_labels,
    std::vector<std::vector<double>>& right_data, std::vector<int>& right_labels) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][feature] < threshold) {     // 小于阈值的数据进入左子树
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        }
        else {  // 大于等于阈值的数据进入右子树
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }
}

// 找到最佳拆分点
void DecisionTree::find_best_split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
    int& best_feature, double& best_threshold) {
    double best_gain = -1.0;            // 初始化最佳信息增益
    int num_features = data[0].size();

    for (int feature = 0; feature < num_features; ++feature) {  // 遍历所有特征
        std::vector<double> values;
        for (const auto& sample : data) {
            values.push_back(sample[feature]);  // 收集特征值
        }

        std::sort(values.begin(), values.end());    // 排序特征值
        for (size_t i = 1; i < values.size(); ++i) {
            double threshold = (values[i - 1] + values[i]) / 2; // 计算阈值
            double gain = calculate_gain(data, labels, feature, threshold);// 计算信息增益

            if (gain > best_gain) { // 更新最佳信息增益和对应的特征和阈值
                best_gain = gain;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
}

// 计算信息增益
double DecisionTree::calculate_gain(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold) {
    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    split_data(data, labels, feature, threshold, left_data, left_labels, right_data, right_labels);// 拆分数据

    double p_left = static_cast<double>(left_labels.size()) / labels.size();
    double p_right = static_cast<double>(right_labels.size()) / labels.size();

    // 信息增益 = 当前熵 - (左子树熵 * 左子树概率 + 右子树熵 * 右子树概率)
    return entropy(labels) - p_left * entropy(left_labels) - p_right * entropy(right_labels);
}

// 计算熵
double DecisionTree::entropy(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    double entropy = 0.0;
    for (const auto& pair : label_counts) {
        double p = static_cast<double>(pair.second) / labels.size();
        entropy -= p * std::log2(p);    // 熵的公式
    }
    return entropy;
}

// 判断节点是否纯净（所有标签是否相同）
bool DecisionTree::is_pure(const std::vector<int>& labels) const {
    return std::all_of(labels.begin(), labels.end(), [&labels](int label) { return label == labels[0]; });
}

// 多数投票
int DecisionTree::majority_vote(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;  // 返回出现次数最多的标签
}

// 预测样本的类别（递归）
int DecisionTree::predict_sample(TreeNode* node, const std::vector<double>& sample) const {
    if (!node->left && !node->right) {  // 叶节点
        return node->prediction;
    }

    if (sample[node->feature_index] < node->threshold) {    // 根据阈值选择左子树或右子树
        return predict_sample(node->left, sample);
    }
    else {
        return predict_sample(node->right, sample);
    }
}

// 保存决策树
void DecisionTree::save(std::ofstream& ofs) const {
    save_node(root, ofs);
}

// 保存节点（递归）
void DecisionTree::save_node(TreeNode* node, std::ofstream& ofs) const {
    if (!node) {    // 空节点
        ofs << "null\n";
        return;
    }
    ofs << node->feature_index << " " << node->threshold << " " << node->prediction << "\n";
    save_node(node->left, ofs);     // 保存左子节点
    save_node(node->right, ofs);    // 保存右子节点
}

// 加载决策树
void DecisionTree::load(std::ifstream& ifs) {
    root = load_node(ifs);  // 递归加载节点
}

// 加载节点（递归）
TreeNode* DecisionTree::load_node(std::ifstream& ifs) {
    std::string token;
    ifs >> token;
    if (token == "null") {  // 空节点
        return nullptr;
    }
    TreeNode* node = new TreeNode();
    node->feature_index = std::stoi(token);     // 读取节点信息
    ifs >> node->threshold >> node->prediction;
    node->left = load_node(ifs);    // 加载左子节点
    node->right = load_node(ifs);   // 加载右子节点
    return node;
}

// 随机森林构造函数
RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, int min_samples_leaf)
    : num_trees(num_trees), max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf) {
    std::srand(std::time(nullptr)); // 初始化随机种子
}

// 训练随机森林
void RandomForest::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    trees.clear();
    for (int i = 0; i < num_trees; ++i) {
        std::vector<std::vector<double>> bootstrap_data;
        std::vector<int> bootstrap_labels;
        bootstrap(data, labels, bootstrap_data, bootstrap_labels);  // 生成自助样本

        DecisionTree tree(max_depth, min_samples_split, min_samples_leaf);
        tree.train(bootstrap_data, bootstrap_labels);   // 训练决策树
        trees.push_back(tree);
    }
}

// 预测样本的类别
int RandomForest::predict(const std::vector<double>& sample) const {
    std::vector<int> predictions;
    for (const auto& tree : trees) {
        predictions.push_back(tree.predict(sample));    // 每棵树的预测结果
    }

    return majority_vote(predictions);  // 多数投票确定最终分类
}

// 计算模型的准确率
double RandomForest::accuracy(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const {
    int correct = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (predict(data[i]) == labels[i]) {    // 判断预测是否正确
            correct++;
        }
    }
    return static_cast<double>(correct) / data.size();
}

// 打印混淆矩阵
void RandomForest::print_confusion_matrix(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const {
    std::unordered_map<int, std::unordered_map<int, int>> matrix;
    int num_classes = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int actual = labels[i];
        int predicted = predict(data[i]);
        matrix[actual][predicted]++;    // 更新混淆矩阵
        num_classes = std::max(num_classes, std::max(actual, predicted) + 1);
    }

    std::cout << "Confusion Matrix:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            std::cout << matrix[i][j] << " ";   // 输出混淆矩阵
        }
        std::cout << std::endl;
    }
}

// 计算模型的性能指标（精确率、召回率、F1分数）
void RandomForest::calculate_metrics(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const {
    std::unordered_map<int, int> true_positives, false_positives, false_negatives;
    for (size_t i = 0; i < data.size(); ++i) {
        int actual = labels[i];
        int predicted = predict(data[i]);
        if (predicted == actual) {
            true_positives[actual]++;
        }
        else {
            false_positives[predicted]++;
            false_negatives[actual]++;
        }
    }

    for (const auto& tp : true_positives) {
        int label = tp.first;
        int tp_count = tp.second;
        int fp_count = false_positives[label];
        int fn_count = false_negatives[label];
        double precision = tp_count / static_cast<double>(tp_count + fp_count);
        double recall = tp_count / static_cast<double>(tp_count + fn_count);
        double f1_score = 2 * (precision * recall) / (precision + recall);
        std::cout << "Label: " << label << " Precision: " << precision << " Recall: " << recall << " F1-Score: " << f1_score << std::endl;
    }
}

// 交叉验证
void RandomForest::cross_validate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int k) {
    // 打乱数据索引
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    // 计算每个fold的大小
    int fold_size = data.size() / k;
    double total_accuracy = 0.0;

    // 划分训练集和测试集
    for (int i = 0; i < k; ++i) {
        std::vector<std::vector<double>> train_data, test_data;
        std::vector<int> train_labels, test_labels;

        for (int j = 0; j < data.size(); ++j) {
            if (j >= i * fold_size && j < (i + 1) * fold_size) {
                test_data.push_back(data[indices[j]]);
                test_labels.push_back(labels[indices[j]]);
            }
            else {
                train_data.push_back(data[indices[j]]);
                train_labels.push_back(labels[indices[j]]);
            }
        }
        // 训练模型
        train(train_data, train_labels);

        // 评估模型
        double acc = accuracy(test_data, test_labels);
        total_accuracy += acc;

        std::cout << "Fold " << i + 1 << " Accuracy: " << acc << std::endl;
    }

    std::cout << "Average Accuracy: " << total_accuracy / k << std::endl;
}

// 网格搜索
void RandomForest::grid_search(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    std::vector<int> tree_counts = { 10, 20, 30 };
    std::vector<int> max_depths = { 10, 15, 20 };
    std::vector<int> min_samples_split = { 2, 5, 10 };
    std::vector<int> min_samples_leaf = { 1, 2, 4 };

    double best_accuracy = 0.0;
    int best_tree_count = 0;
    int best_max_depth = 0;
    int best_min_samples_split = 0;
    int best_min_samples_leaf = 0;

    for (int tree_count : tree_counts) {
        for (int max_depth : max_depths) {
            for (int min_split : min_samples_split) {
                for (int min_leaf : min_samples_leaf) {
                    RandomForest forest(tree_count, max_depth, min_split, min_leaf);
                    std::vector<int> indices(data.size());
                    std::iota(indices.begin(), indices.end(), 0);
                    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

                    int k = 5;
                    int fold_size = data.size() / k;
                    double total_accuracy = 0.0;

                    for (int i = 0; i < k; ++i) {
                        std::vector<std::vector<double>> train_data, test_data;
                        std::vector<int> train_labels, test_labels;

                        for (int j = 0; j < data.size(); ++j) {
                            if (j >= i * fold_size && j < (i + 1) * fold_size) {
                                test_data.push_back(data[indices[j]]);
                                test_labels.push_back(labels[indices[j]]);
                            }
                            else {
                                train_data.push_back(data[indices[j]]);
                                train_labels.push_back(labels[indices[j]]);
                            }
                        }

                        forest.train(train_data, train_labels);
                        double acc = forest.accuracy(test_data, test_labels);
                        total_accuracy += acc;
                    }

                    double avg_accuracy = total_accuracy / k;
                    if (avg_accuracy > best_accuracy) {
                        best_accuracy = avg_accuracy;
                        best_tree_count = tree_count;
                        best_max_depth = max_depth;
                        best_min_samples_split = min_split;
                        best_min_samples_leaf = min_leaf;
                    }
                }
            }
        }
    }

    std::cout << "Best Accuracy: " << best_accuracy << " with Trees: " << best_tree_count
        << ", Max Depth: " << best_max_depth
        << ", Min Samples Split: " << best_min_samples_split
        << ", Min Samples Leaf: " << best_min_samples_leaf << std::endl;
}

// 自助法生成样本
void RandomForest::bootstrap(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
    std::vector<std::vector<double>>& bootstrap_data, std::vector<int>& bootstrap_labels) const {
    size_t n = data.size();
    for (size_t i = 0; i < n; ++i) {
        size_t index = std::rand() % n;     // 随机采样
        bootstrap_data.push_back(data[index]);
        bootstrap_labels.push_back(labels[index]);
    }
}

// 多数投票
int RandomForest::majority_vote(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;  // 返回出现次数最多的标签
}

// 保存模型
void RandomForest::save_model(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file to save model");
    }
    ofs << num_trees << " " << max_depth << " " << min_samples_split << " " << min_samples_leaf << "\n";
    for (const auto& tree : trees) {
        tree.save(ofs); // 保存每棵树
    }
}

// 加载模型
void RandomForest::load_model(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file to load model");
    }
    ifs >> num_trees >> max_depth >> min_samples_split >> min_samples_leaf; // 读取模型参数
    trees.resize(num_trees, DecisionTree(max_depth, min_samples_split, min_samples_leaf));
    for (auto& tree : trees) {
        tree.load(ifs);     // 加载每棵树
    }
}