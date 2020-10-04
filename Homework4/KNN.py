"""
KNN算法比较好实现，这里选择手动实现
将分类器封装在类中，直接用
"""
import dataset
import visualization as vl


class KNNClassifier:
    def __init__(self, feature, labels, k):
        self.feature = feature
        self.labels = labels
        self.k = k

    # 计算两个样本的欧式距离，先求各自对应维的平方和，再开根号
    @classmethod
    def get_distance(cls, features_1, features_2):
        n = len(features_1)
        sum_of_squares = 0
        for i in range(n):
            sum_of_squares += (features_1[i] - features_2[i]) ** 2
        return sum_of_squares ** 0.5

    def _vote(self, feature_line):
        dists = {}
        for index in range(len(self.feature)):
            dist = self.get_distance(self.feature[index], feature_line)
            dists[index] = dist
        # 距离由近到远排序
        dists_queue = sorted(dists.items(), key=lambda x: x[1])

        votes = {}  # 5个近邻投票，票数最高的类就是预测结果
        for i in range(self.k):
            index = dists_queue[i][0]
            t = self.labels[index]
            if t not in votes:
                votes[t] = 1
            else:
                votes[t] += 1
        votes = sorted(votes.items(), key=lambda x: -x[1])
        # 取票数最高者
        return votes[0][0]

    def predict(self, feature):
        res = []
        for feature_line in feature:
            res.append(self._vote(feature_line))
        return res

    # 根据正确率评估效果
    def score(self, feature, labels):
        predict_set = self.predict(feature)
        return len([index for index in range(len(labels)) if predict_set[index] == labels[index]]) / len(labels)


knn = KNNClassifier(dataset.feat_train, dataset.label_train, 5)

print("训练集: {:.3f}".format(knn.score(dataset.feat_train, dataset.label_train)))
print("测试集: {:.3f}".format(knn.score(dataset.feat_test, dataset.label_test)))

label_train_prediction = knn.predict(dataset.feat_train)
label_test_prediction = knn.predict(dataset.feat_test)

vl.get_model_graph(dataset.feat_test, dataset.label_test, label_test_prediction, dataset.feat_train,
                   dataset.label_train, label_train_prediction)
