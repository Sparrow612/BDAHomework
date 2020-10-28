import tensorflow as tf
import dataset
import matplotlib.pyplot as plt

# 训练集和测试集统一从dataset.py文件中获取
x_train = dataset.feat_train
y_train = dataset.label_train
x_test = dataset.feat_test
y_test = dataset.label_test

# 类型转化，方便后续的矩阵乘法运算
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 输入特征和标签值一一对应
# 把数据集分批次，每个批次15组数据
train_items = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(15)  # 105个训练数据，共7组
test_items = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(15)  # 45个测试数据，共3组


class BPNNClassifier:

    def __init__(self, lr=0.1, epoch=200):
        self.lr = lr  # 学习率
        self.epoch = epoch  # 演化次数，这里的200次已经能达到很好的效果
        self.train_loss_results = []
        self.test_correct_rates = []
        self.loss_sum = 0
        # 生成神经网络的参数，4个输入特征，故输入层为4个输入节点；因为3分类，故输出层为3个神经元
        # 用tf.Variable()标记参数可训练
        self.w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.25))
        self.b = tf.Variable(tf.random.truncated_normal([3], stddev=0.25))

    def train(self, train_items, test_items):

        steps = len(train_items)  # 一轮的训练数据有steps组

        for e in range(self.epoch):

            for step, (x_train, y_train) in enumerate(train_items):
                with tf.GradientTape() as tape:

                    y = tf.matmul(x_train, self.w) + self.b
                    y = tf.nn.softmax(y)
                    y_ = tf.one_hot(y_train, depth=3)
                    loss = tf.reduce_mean(tf.square(y_ - y))
                    self.loss_sum += loss.numpy()

                # 计算loss对各个参数的梯度
                grads = tape.gradient(loss, [self.w, self.b])
                # 实现梯度更新 w = w - lr * w_grad    b = b - lr * b_grad
                self.w.assign_sub(self.lr * grads[0])
                self.b.assign_sub(self.lr * grads[1])

            # 每个epoch，打印loss信息
            print("Epoch {}\nloss: {}".format(e, self.loss_sum / steps))
            self.train_loss_results.append(self.loss_sum / steps)  # 每次用steps组数据训练，loss取steps次loss的均值
            self.loss_sum = 0  # loss_sum归零，为记录下一个epoch的loss做准备

            total_correct, total_number = 0, 0  # 计算本轮正确率

            # 使用本轮训练后的参数预测，查看正确率
            for x_test, y_test in test_items:
                y = tf.matmul(x_test, self.w) + self.b
                y = tf.nn.softmax(y)
                pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
                pred = tf.cast(pred, dtype=y_test.dtype)
                correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                total_correct += int(correct)
                total_number += x_test.shape[0]
            score = total_correct / total_number
            self.test_correct_rates.append(score)
            print('accuracy: {}'.format(score))
            print('===========================================')

    def visualize(self):
        plt.figure()
        # 绘制loss曲线
        plt.subplot(121)
        plt.title('Loss')
        plt.plot(self.train_loss_results, label='Training loss')
        plt.legend()
        plt.ylim(ymax=0.25)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # 绘制正确率曲线
        plt.subplot(122)
        plt.title('Accuracy')
        plt.plot(self.test_correct_rates, label='accuracy', color='r')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        # 大功告成
        plt.show()


bpc = BPNNClassifier()
bpc.train(train_items, test_items)
print(bpc.w)
print(bpc.b)
bpc.visualize()
