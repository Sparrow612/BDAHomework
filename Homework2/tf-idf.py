# TF-IDF算法代码实现
from collections import defaultdict
from Homework2 import pre_processor
import math
import sys


def TF_IDF(word_pool, essay_id):
    pairs = defaultdict(int)
    tf_idf = []
    for word in word_pool[essay_id]:
        pairs[word] += 1
    for word, freq in pairs.items():
        n = 0
        for i in range(len(word_pool)):
            if word in word_pool[i]: n += 1
        idf = math.log(len(word_pool) / n + 1)
        tf_idf.append((word, freq * idf))
    tf_idf.sort(key=lambda x: -x[1])
    return tf_idf


def drive_TF_IDF():
    word_pool = pre_processor.pre_process()
    tf_idfs = []
    for i in range(len(word_pool)):
        tf_idfs.append(TF_IDF(word_pool, i))
    essays = pre_processor.get_essay()
    standard_out = sys.stdout
    sys.stdout = open('result/TF_IDF.txt', 'w')
    for i in range(len(essays)):
        print(essays[i][:-4])
        pairs = tf_idfs[i]
        for j in range(min(len(pairs), 20)):
            print(pairs[j][0] + '\t' + str(pairs[j][1]))
        print()
    sys.stdout.close()
    sys.stdout = standard_out


if __name__ == '__main__':
    drive_TF_IDF()
