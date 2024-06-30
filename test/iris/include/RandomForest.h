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
    DecisionTree(int max_depth = 10, int min_samples_split = 2, int min_samples_leaf = 1);      // ���캯������ʼ�������������͸��ڵ�
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);   // ѵ��������
    int predict(const std::vector<double>& sample) const;                                       // Ԥ�����������

    void save(std::ofstream& ofs) const;    // ���������
    void load(std::ifstream& ifs);          // ���ؾ�����

private:
    TreeNode* root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    TreeNode* build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth);                      // �ݹ鹹��������
    void split_data(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold,        // ������������ֵ�������
        std::vector<std::vector<double>>& left_data, std::vector<int>& left_labels,                 
        std::vector<std::vector<double>>& right_data, std::vector<int>& right_labels);
    void find_best_split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,                                  // �ҵ���Ѳ�ֵ�
        int& best_feature, double& best_threshold);
    double calculate_gain(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int feature, double threshold); // ������Ϣ����
    double entropy(const std::vector<int>& labels) const;                           // ������
    bool is_pure(const std::vector<int>& labels) const;                             // �жϽڵ��Ƿ񴿾������б�ǩ�Ƿ���ͬ��
    int majority_vote(const std::vector<int>& labels) const;                        // ����ͶƱ
    int predict_sample(TreeNode* node, const std::vector<double>& sample) const;    // Ԥ����������𣨵ݹ飩

    void save_node(TreeNode* node, std::ofstream& ofs) const;   // ����ڵ㣨�ݹ飩
    TreeNode* load_node(std::ifstream& ifs);                    // ���ؽڵ㣨�ݹ飩
};

class RandomForest {
public:
    RandomForest(int num_trees = 20, int max_depth = 15, int min_samples_split = 2, int min_samples_leaf = 1);      // ���ɭ�ֹ��캯��
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);                       // ѵ�����ɭ��
    int predict(const std::vector<double>& sample) const;                                                           // Ԥ�����������
    double accuracy(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;            // ����ģ�͵�׼ȷ��
    void print_confusion_matrix(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;// ��ӡ��������
    void calculate_metrics(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) const;     // ����ģ�͵�����ָ�꣨��ȷ�ʡ��ٻ��ʡ�F1������
    void cross_validate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int k = 5);   // ������֤
    static void grid_search(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);          // ��������

    void save_model(const std::string& filename) const; // ����ģ��
    void load_model(const std::string& filename);       // ����ģ��

private:
    int num_trees;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    std::vector<DecisionTree> trees;
    void bootstrap(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,        // ��������������
        std::vector<std::vector<double>>& bootstrap_data, std::vector<int>& bootstrap_labels) const;
    int majority_vote(const std::vector<int>& labels) const;                                            // ����ͶƱ
};

#endif // RANDOM_FOREST_H
