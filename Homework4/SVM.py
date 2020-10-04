from sklearn import svm
import dataset


"""
难度大，这里选择调包了
"""
# 训练SVM分类器
svm_classifier = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01)
svm_classifier.fit(dataset.feat_train, dataset.label_train)

# 查看分类器分类效果
print("训练集:", svm_classifier.score(dataset.feat_train, dataset.label_train))
print("测试集:", svm_classifier.score(dataset.feat_test, dataset.label_test))