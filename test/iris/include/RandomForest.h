#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <unordered_map>
#include <fstream>

struct TreeNode {
    int feature_index;
    double threshold;
    TreeNode* left;
    TreeNode* right;
    int prediction;

    TreeNode() : feature_index(-1), threshold(0.0), left(nullptr), right(nullptr), prediction(-1) {}
};

class DecisionTree {
public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2, int min_samples_leaf = 1);      // 构造函数：初始化决策树参数和根节点
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);   // 训练决策树
    int predict(const std::vector<double>& sample) const;                                       // 预测样本的类别

    void save(std::ofstream& ofs) const;    // 保存决策树
    void load(std::ifstream& ifs);          // 加载决策树

private:
    TreeNode* root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    TreeNode* build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth);                      // 递归构建决策树
    void split_data(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold,        // 根据特征和阈值拆分数据
        std::vector<std::vector<double>>& left_data, std::vector<int>& left_labels,                 
        std::vector<std::vector<double>>& right_data, std::vector<int>& right_labels);
    void find_best_split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,                                  // 找到最佳拆分点
        int& best_feature, double& best_threshold);
    double calculate_gain(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold); // 计算信息增益
    double entropy(const std::vector<int>& labels) const;                           // 计算熵
    bool is_pure(const std::vector<int>& labels) const;                             // 判断节点是否纯净（所有标签是否相同）
    int majority_vote(const std::vector<int>& labels) const;                        // 多数投票
    int predict_sample(TreeNode* node, const std::vector<double>& sample) const;    // 预测样本的类别（递归）

    void save_node(TreeNode* node, std::ofstream& ofs) const;   // 保存节点（递归）
    TreeNode* load_node(std::ifstream& ifs);                    // 加载节点（递归）
};

class RandomForest {
public:
    RandomForest(int num_trees = 20, int max_depth = 15, int min_samples_split = 2, int min_samples_leaf = 1);      // 随机森林构造函数
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);                       // 训练随机森林
    int predict(const std::vector<double>& sample) const;                                                           // 预测样本的类别
    double accuracy(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;            // 计算模型的准确率
    void print_confusion_matrix(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;// 打印混淆矩阵
    void calculate_metrics(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;     // 计算模型的性能指标（精确率、召回率、F1分数）
    void cross_validate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int k = 5);   // 交叉验证
    static void grid_search(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);          // 网格搜索

    void save_model(const std::string& filename) const; // 保存模型
    void load_model(const std::string& filename);       // 加载模型

private:
    int num_trees;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    std::vector<DecisionTree> trees;
    void bootstrap(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,        // 自助法生成样本
        std::vector<std::vector<double>>& bootstrap_data, std::vector<int>& bootstrap_labels) const;
    int majority_vote(const std::vector<int>& labels) const;                                            // 多数投票
};

#endif // RANDOM_FOREST_H
