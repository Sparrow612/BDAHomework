# 使用matplotlib可视化模型评估结果（降维取前两个维度）

import matplotlib.pyplot as plt
from graphviz import Digraph

"""
绘制散点图，按顺时针方向依次为
1. 测试集的标签分布
2. 测试集的预测标签分布
3. 训练集的标签分布
4. 训练集的预测标签分布
"""


def get_model_graph(feat_train, label_train, label_train_prediction, feat_test, label_test, label_test_prediction):
    plt.figure()

    plt.subplot(221)
    plt.title('Train-Actual')
    plt.scatter(feat_train[:, 0], feat_train[:, 1], c=label_train, edgecolors='k', s=50)
    plt.subplot(222)
    plt.title('Train-Prediction')
    plt.scatter(feat_train[:, 0], feat_train[:, 1], c=label_train_prediction, edgecolors='k', s=50)
    plt.subplot(223)
    plt.title('Test-Actual')
    plt.scatter(feat_test[:, 0], feat_test[:, 1], c=label_test, edgecolors='k', s=50)
    plt.subplot(224)
    plt.title('Test-Prediction')
    plt.scatter(feat_test[:, 0], feat_test[:, 1], c=label_test_prediction, edgecolors='k', s=50)

    plt.show()


def plot_model(tree, name):
    def _sub_plot(g, tree, inc):
        nonlocal root
        first_label = list(tree.keys())[0]
        ts = tree[first_label]
        for i in ts.keys():
            if isinstance(tree[first_label][i], dict):
                root = str(int(root) + 1)
                g.node(root, list(tree[first_label][i].keys())[0])
                g.edge(inc, root, str(i))
                _sub_plot(g, tree[first_label][i], root)
            else:
                root = str(int(root) + 1)
                g.node(root, tree[first_label][i])
                g.edge(inc, root, str(i))

    root = '0'
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    g.view()
