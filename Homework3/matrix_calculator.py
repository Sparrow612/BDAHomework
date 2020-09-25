import numpy as np
from Homework3.matrix_reader import MatrixReader

img_path = 'res/matrix.png'

arr = MatrixReader(img_path).latex_to_matrix()

matrix = np.array(arr)
n = matrix.shape[1]
vector = np.array([[0.25, 0.25, 0.25, 0.25]]).T


def transfer(times=10):
    global vector
    for i in range(times):
        print('迭代', i + 1, '==========')
        vector = np.dot(matrix, vector)
        print(vector)


def transfer_with_heartbeat(times=10, beta=0.2):
    global vector
    tail = np.array([[1, 1, 1, 1]]).T
    tail = np.multiply(tail, beta / n)
    for i in range(times):
        print('心跳转移迭代', i + 1, '==========')
        vector = np.add(np.multiply(
            np.dot(matrix, vector), 1 - beta), tail)
        print(vector)


# transfer()
transfer_with_heartbeat()
