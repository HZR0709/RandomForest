# RandomForest（随机森林）
 从决策树开始，然后构建一个随机森林。

到目前为止，完成了以下几个步骤来构建和优化随机森林和KNN堆叠模型。

**1. 初始实现决策树和随机森林**

实现了决策树（Decision Tree）类，包括训练（train）、预测（predict）和评估函数。
实现了随机森林（Random Forest）类，包括训练（train）、预测（predict）、准确率计算（accuracy）、混淆矩阵打印（print_confusion_matrix）和评估函数（calculate_metrics）。

**2. 数据集和交叉验证**

加载了一个自拟的数据集，用于模型的训练和评估。
实现了交叉验证（cross_validate）功能，通过K折交叉验证来评估模型的性能。

**3. 实现KNN算法**

实现了一个简单的KNN类，包括训练（train）和预测（predict）函数。
通过计算欧几里得距离来确定最近邻样本。

**4. 构建堆叠模型**

实现了堆叠模型（Stacking）类，结合了随机森林和KNN，通过简单投票机制进行预测。
实现了堆叠模型的训练和预测函数。

**5. 特征选择和降维**

实现了主成分分析（PCA）类，用于数据降维。
在主程序中应用PCA，先对数据进行降维，再使用降维后的数据进行模型训练和评估。

**6. 增大数据集和数据增强**

增加了数据集的规模，以更全面地评估模型的泛化能力。
介绍了数据增强技术，如添加随机噪声和数据变换。

**7. 超参数优化**

实现了随机森林的超参数优化（grid_search）功能，包括树的数量、最大深度、最小样本分裂和最小样本叶子数等参数的网格搜索。
在主程序中进行了超参数优化，并使用最佳参数进行模型训练和评估。

**8. 模型的保存和加载**

使用C++标准库中的文件操作和自定义的序列化方法来保存和加载模型。

# 模型测试
为了预测模型的性能，我使用了UCI机器学习库中的`iris`数据集，该数据集仅包含150个实例，便于测试。

我将数据集分为两个部分，使用其中一部分数据来训练模型，加载训练好的模型，对另一部分数据进行预测分类，预测准确率达到了100%。

当然，这并不能完全代表模型的性能，因为我使用的数据集很小，数据特征只有4个。 

**训练和预测结果：**
```
Loaded 150 samples with 4 features each.
Sample 0: 5.1 3.5 1.4 0.2 Label: 0
Sample 1: 4.9 3 1.4 0.2 Label: 0
Sample 2: 4.7 3.2 1.3 0.2 Label: 0
Sample 3: 4.6 3.1 1.5 0.2 Label: 0
Sample 4: 5 3.6 1.4 0.2 Label: 0
Best Accuracy: 0.973333 with Trees: 10, Max Depth: 10, Min Samples Split: 10, Min Samples Leaf: 2
Fold 1 Accuracy: 0.966667
Fold 2 Accuracy: 0.933333
Fold 3 Accuracy: 0.966667
Fold 4 Accuracy: 0.933333
Fold 5 Accuracy: 0.9
Average Accuracy: 0.94
Prediction for sample 6.3, 3.4, 5.6, 2.4: Iris-virginica
Prediction for sample 6.4, 3.1, 5.5, 1.8: Iris-virginica
Prediction for sample 6, 3, 4.8, 1.8: Iris-virginica
Prediction for sample 6.9, 3.1, 5.4, 2.1: Iris-virginica
Prediction for sample 6.7, 3.1, 5.6, 2.4: Iris-virginica
Prediction for sample 5.7, 2.8, 4.1, 1.3: Iris-versicolor
Prediction for sample 5.1, 2.5, 3, 1.1: Iris-versicolor
Prediction for sample 6.2, 2.9, 4.3, 1.3: Iris-versicolor
Prediction for sample 5.7, 2.9, 4.2, 1.3: Iris-versicolor
Prediction for sample 5.7, 3, 4.2, 1.2: Iris-versicolor
Prediction for sample 5, 3.3, 1.4, 0.2: Iris-setosa
Prediction for sample 5.3, 3.7, 1.5, 0.2: Iris-setosa
Prediction for sample 4.6, 3.2, 1.4, 0.2: Iris-setosa
Prediction for sample 5.1, 3.8, 1.6, 0.2: Iris-setosa
Prediction for sample 4.8, 3, 1.4, 0.3: Iris-setosa
Accuracy: 1
Confusion Matrix:
5 0 0
0 5 0
0 0 5
Label: 2 Precision: 1 Recall: 1 F1-Score: 1
Label: 1 Precision: 1 Recall: 1 F1-Score: 1
Label: 0 Precision: 1 Recall: 1 F1-Score: 1
```
