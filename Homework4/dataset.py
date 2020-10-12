from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
为其他算法实现文件提供数据，所有的算法都从此py文件中获取数据
训练集和测试集比例7:3
45条测试数据，105条训练数据
"""

# 导入鸢尾花数据集
iris = datasets.load_iris()

# 数据集划分, 按照7：3的比例（45个测试用例）划分训练集和测试集

# 特征
iris_feature = iris.data
iris_feat_name = iris.feature_names
# 标签
iris_label = iris.target
iris_target_name = iris.target_names

# 划分
feat_train, feat_test, label_train, label_test = train_test_split(iris_feature, iris_label, test_size=0.3,
                                                                  random_state=42)
