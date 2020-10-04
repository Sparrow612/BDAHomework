import dataset
import math
from visualization import plot_model


# 对可能的取值出现的次数进行统计
def unique_counts(src):
    result = {}
    for r in src:
        if r not in result: result[r] = 0
        result[r] += 1
    return result


# 信息熵
def entropy(rows):
    results = unique_counts(rows)
    # 开始计算熵的值
    ent = 0.0
    for r in results.keys():
        p = results[r] / len(rows)
        ent -= p * math.log(p, 2)
    return ent


# 计算基尼指数
def gini(rows):
    total = len(rows)
    counts = unique_counts(rows)
    impure = 0.0
    for k1 in counts.keys():
        p1 = float(counts[k1]) / total
        impure += p1 * (1 - p1)
    return impure


def divide_set(feats, labels, col, value):
    def split(row):
        return row[col] >= value

    ts, fs = [], []
    for i in range(len(feats)):
        item = (feats[i], labels[i])
        if split(feats[i]):
            ts.append(item)
        else:
            fs.append(item)
    return ts, fs


class TreeNode:
    def __init__(self, criteria=None, result=None, tn=None, fn=None):
        self.criteria = criteria  # 决策依据，(col, value), 分别代表列索引和阈值
        self.result = result  # 保存的是针对当前分支的结果，只有叶节点才有这个值
        self.tn = tn  # true node，对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fn = fn  # false node，对应于结果为false时，树上相对于当前节点的子树上的节点

    def go_down(self, data):
        v = data[self.criteria[0]]
        if v >= self.criteria[1]:
            return self.tn
        return self.fn


def build_tree(feats, labels):
    def get_feat_and_label(src):
        feats, labels = [], []
        for s in src:
            feats.append(s[0])
            labels.append(s[1])
        return feats, labels

    if len(feats) == 0: return None

    curr_score = entropy(labels)

    max_gain = 0.0
    best_criteria = None
    results = None

    n = len(feats[0])  # 维数
    for col in range(n):
        values = set([row[col] for row in feats])  # 可能取值的集合
        for value in values:
            ts, fs = divide_set(feats, labels, col, value)
            p = len(ts) / len(feats)
            gain = curr_score - (p * entropy([t[1] for t in ts]) + (1 - p) * entropy([f[1] for f in fs]))
            if gain > max_gain and len(ts) and len(fs):
                # 需要剔除无关因子
                max_gain = gain
                best_criteria = (col, value)
                results = (ts, fs)

    if max_gain:
        feats, labels = get_feat_and_label(results[0])
        tn = build_tree(feats, labels)
        feats, labels = get_feat_and_label(results[1])
        fn = build_tree(feats, labels)
        return TreeNode(criteria=best_criteria, tn=tn, fn=fn)
    else:  # 叶节点
        return TreeNode(result=labels[0])


class DecisionTreeClassifier:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def fit(self):
        self.tree = build_tree(self.features, self.labels)

    def predict(self, feats):
        def classify(tree, feat):
            if tree.result is not None:
                return tree.result
            tree = tree.go_down(feat)
            return classify(tree, feat)

        tree = self.tree
        res = []
        for row in feats:
            res.append(classify(tree, row))
        return res

    def score(self, feats_set, label_set):
        n = len(label_set)
        predict_set = self.predict(feats_set)
        return len([index for index in range(n) if predict_set[index] == label_set[index]]) / n

    def print_tree(self):
        # 是否是叶节点
        def _print(root, indent='-', dict_tree={}, direct='root'):
            if root.result is not None:
                dict_tree = {direct: str(root.result) + ':' + dataset.iris_target_name[root.result]}
            else:
                left = _print(root.tn, indent=indent + '-', direct='yes')
                left_copy = left.copy()
                right = _print(root.fn, indent=indent + '-', direct='no')
                right_copy = right.copy()
                left_copy.update(right_copy)
                cnt = 'dimension:' + str(root.criteria[0]) + ' ' + dataset.iris_feat_name[
                    root.criteria[0]] + '\nvalue>=:' + str(root.criteria[1]) + '?'
                if indent != '-':
                    dict_tree = {direct: {cnt: left_copy}}
                else:  # 根
                    dict_tree = {cnt: left_copy}
            return dict_tree

        return _print(self.tree)


dt_model = DecisionTreeClassifier(dataset.feat_train, dataset.label_train)  # 决策树分类器
dt_model.fit()  # 训练决策树模型
predict_y = dt_model.predict(dataset.feat_test)
score = dt_model.score(dataset.feat_test, dataset.label_test)
plot_model(dt_model.print_tree(), 'tree_graph.gv')

print('模型预测结果', predict_y)
print('预期结果', dataset.label_test)
print('模型准确率: {:.3f}'.format(score))
