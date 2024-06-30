#include "RandomForest.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <random>
#include <fstream>
#include <string>

// ���캯������ʼ�������������͸��ڵ�
DecisionTree::DecisionTree(int max_depth, int min_samples_split, int min_samples_leaf)
    : max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf), root(nullptr) {}

// ѵ��������
void DecisionTree::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    root = build_tree(data, labels, 0); // �Ӹ��ڵ㿪ʼ������
}

// Ԥ�����������
int DecisionTree::predict(const std::vector<double>& sample) const {
    return predict_sample(root, sample);    // �Ӹ��ڵ㿪ʼԤ��
}

// �ݹ鹹��������
TreeNode* DecisionTree::build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth) {
    // ��ֹ�������ﵽ�����ȡ��ڵ㴿����������С����С���������
    if (depth >= max_depth || is_pure(labels) || data.size() < min_samples_split) {
        TreeNode* leaf = new TreeNode();            // ����Ҷ�ڵ�
        leaf->prediction = majority_vote(labels);   // Ҷ�ڵ��Ԥ��ֵΪ����ͶƱ���
        return leaf;
    }

    int best_feature;
    double best_threshold;
    find_best_split(data, labels, best_feature, best_threshold);    // �ҵ���Ѳ�ֵ�

    TreeNode* node = new TreeNode();
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    split_data(data, labels, best_feature, best_threshold, left_data, left_labels, right_data, right_labels);// ���ݲ��

    // ����Ƿ�������СҶ�ڵ�������
    if (left_data.size() < min_samples_leaf || right_data.size() < min_samples_leaf) {
        TreeNode* leaf = new TreeNode();            // ����Ҷ�ڵ�
        leaf->prediction = majority_vote(labels);   // Ҷ�ڵ��Ԥ��ֵΪ����ͶƱ���
        return leaf;
    }

    // �ݹ鹹����������
    node->left = build_tree(left_data, left_labels, depth + 1);
    node->right = build_tree(right_data, right_labels, depth + 1);

    return node;
}

// ������������ֵ�������
void DecisionTree::split_data(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold,
    std::vector<std::vector<double>>& left_data, std::vector<int>& left_labels,
    std::vector<std::vector<double>>& right_data, std::vector<int>& right_labels) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][feature] < threshold) {     // С����ֵ�����ݽ���������
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        }
        else {  // ���ڵ�����ֵ�����ݽ���������
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }
}

// �ҵ���Ѳ�ֵ�
void DecisionTree::find_best_split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
    int& best_feature, double& best_threshold) {
    double best_gain = -1.0;            // ��ʼ�������Ϣ����
    int num_features = data[0].size();

    for (int feature = 0; feature < num_features; ++feature) {  // ������������
        std::vector<double> values;
        for (const auto& sample : data) {
            values.push_back(sample[feature]);  // �ռ�����ֵ
        }

        std::sort(values.begin(), values.end());    // ��������ֵ
        for (size_t i = 1; i < values.size(); ++i) {
            double threshold = (values[i - 1] + values[i]) / 2; // ������ֵ
            double gain = calculate_gain(data, labels, feature, threshold);// ������Ϣ����

            if (gain > best_gain) { // ���������Ϣ����Ͷ�Ӧ����������ֵ
                best_gain = gain;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
}

// ������Ϣ����
double DecisionTree::calculate_gain(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold) {
    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    split_data(data, labels, feature, threshold, left_data, left_labels, right_data, right_labels);// �������

    double p_left = static_cast<double>(left_labels.size()) / labels.size();
    double p_right = static_cast<double>(right_labels.size()) / labels.size();

    // ��Ϣ���� = ��ǰ�� - (�������� * ���������� + �������� * ����������)
    return entropy(labels) - p_left * entropy(left_labels) - p_right * entropy(right_labels);
}

// ������
double DecisionTree::entropy(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    double entropy = 0.0;
    for (const auto& pair : label_counts) {
        double p = static_cast<double>(pair.second) / labels.size();
        entropy -= p * std::log2(p);    // �صĹ�ʽ
    }
    return entropy;
}

// �жϽڵ��Ƿ񴿾������б�ǩ�Ƿ���ͬ��
bool DecisionTree::is_pure(const std::vector<int>& labels) const {
    return std::all_of(labels.begin(), labels.end(), [&labels](int label) { return label == labels[0]; });
}

// ����ͶƱ
int DecisionTree::majority_vote(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;  // ���س��ִ������ı�ǩ
}

// Ԥ����������𣨵ݹ飩
int DecisionTree::predict_sample(TreeNode* node, const std::vector<double>& sample) const {
    if (!node->left && !node->right) {  // Ҷ�ڵ�
        return node->prediction;
    }

    if (sample[node->feature_index] < node->threshold) {    // ������ֵѡ����������������
        return predict_sample(node->left, sample);
    }
    else {
        return predict_sample(node->right, sample);
    }
}

// ���������
void DecisionTree::save(std::ofstream& ofs) const {
    save_node(root, ofs);
}

// ����ڵ㣨�ݹ飩
void DecisionTree::save_node(TreeNode* node, std::ofstream& ofs) const {
    if (!node) {    // �սڵ�
        ofs << "null\n";
        return;
    }
    ofs << node->feature_index << " " << node->threshold << " " << node->prediction << "\n";
    save_node(node->left, ofs);     // �������ӽڵ�
    save_node(node->right, ofs);    // �������ӽڵ�
}

// ���ؾ�����
void DecisionTree::load(std::ifstream& ifs) {
    root = load_node(ifs);  // �ݹ���ؽڵ�
}

// ���ؽڵ㣨�ݹ飩
TreeNode* DecisionTree::load_node(std::ifstream& ifs) {
    std::string token;
    ifs >> token;
    if (token == "null") {  // �սڵ�
        return nullptr;
    }
    TreeNode* node = new TreeNode();
    node->feature_index = std::stoi(token);     // ��ȡ�ڵ���Ϣ
    ifs >> node->threshold >> node->prediction;
    node->left = load_node(ifs);    // �������ӽڵ�
    node->right = load_node(ifs);   // �������ӽڵ�
    return node;
}

// ���ɭ�ֹ��캯��
RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, int min_samples_leaf)
    : num_trees(num_trees), max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf) {
    std::srand(std::time(nullptr)); // ��ʼ���������
}

// ѵ�����ɭ��
void RandomForest::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    trees.clear();
    for (int i = 0; i < num_trees; ++i) {
        std::vector<std::vector<double>> bootstrap_data;
        std::vector<int> bootstrap_labels;
        bootstrap(data, labels, bootstrap_data, bootstrap_labels);  // ������������

        DecisionTree tree(max_depth, min_samples_split, min_samples_leaf);
        tree.train(bootstrap_data, bootstrap_labels);   // ѵ��������
        trees.push_back(tree);
    }
}

// Ԥ�����������
int RandomForest::predict(const std::vector<double>& sample) const {
    std::vector<int> predictions;
    for (const auto& tree : trees) {
        predictions.push_back(tree.predict(sample));    // ÿ������Ԥ����
    }

    return majority_vote(predictions);  // ����ͶƱȷ�����շ���
}

// ����ģ�͵�׼ȷ��
double RandomForest::accuracy(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const {
    int correct = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (predict(data[i]) == labels[i]) {    // �ж�Ԥ���Ƿ���ȷ
            correct++;
        }
    }
    return static_cast<double>(correct) / data.size();
}

// ��ӡ��������
void RandomForest::print_confusion_matrix(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const {
    std::unordered_map<int, std::unordered_map<int, int>> matrix;
    int num_classes = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int actual = labels[i];
        int predicted = predict(data[i]);
        matrix[actual][predicted]++;    // ���»�������
        num_classes = std::max(num_classes, std::max(actual, predicted) + 1);
    }

    std::cout << "Confusion Matrix:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            std::cout << matrix[i][j] << " ";   // �����������
        }
        std::cout << std::endl;
    }
}

// ����ģ�͵�����ָ�꣨��ȷ�ʡ��ٻ��ʡ�F1������
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

// ������֤
void RandomForest::cross_validate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int k) {
    // ������������
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    // ����ÿ��fold�Ĵ�С
    int fold_size = data.size() / k;
    double total_accuracy = 0.0;

    // ����ѵ�����Ͳ��Լ�
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
        // ѵ��ģ��
        train(train_data, train_labels);

        // ����ģ��
        double acc = accuracy(test_data, test_labels);
        total_accuracy += acc;

        std::cout << "Fold " << i + 1 << " Accuracy: " << acc << std::endl;
    }

    std::cout << "Average Accuracy: " << total_accuracy / k << std::endl;
}

// ��������
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

// ��������������
void RandomForest::bootstrap(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
    std::vector<std::vector<double>>& bootstrap_data, std::vector<int>& bootstrap_labels) const {
    size_t n = data.size();
    for (size_t i = 0; i < n; ++i) {
        size_t index = std::rand() % n;     // �������
        bootstrap_data.push_back(data[index]);
        bootstrap_labels.push_back(labels[index]);
    }
}

// ����ͶƱ
int RandomForest::majority_vote(const std::vector<int>& labels) const {
    std::unordered_map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        })->first;  // ���س��ִ������ı�ǩ
}

// ����ģ��
void RandomForest::save_model(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file to save model");
    }
    ofs << num_trees << " " << max_depth << " " << min_samples_split << " " << min_samples_leaf << "\n";
    for (const auto& tree : trees) {
        tree.save(ofs); // ����ÿ����
    }
}

// ����ģ��
void RandomForest::load_model(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file to load model");
    }
    ifs >> num_trees >> max_depth >> min_samples_split >> min_samples_leaf; // ��ȡģ�Ͳ���
    trees.resize(num_trees, DecisionTree(max_depth, min_samples_split, min_samples_leaf));
    for (auto& tree : trees) {
        tree.load(ifs);     // ����ÿ����
    }
}