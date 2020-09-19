# TF-IDF算法代码实现
from collections import defaultdict
from Homework2 import pre_processor
import math
import sys


def TF_IDF(word_pool, essay_id):
    # corpus = [' '.join(words) for words in pre_processor.pre_process()]
    # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # tfidf = transformer.fit_transform(
    #     vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(weight)):
    #     print("这里输出第", i, "篇论文关键词tf-idf权重")
    #     pairs = []
    #     for j in range(len(word)):
    #         if weight[i][j] > 0: pairs.append((word[j], weight[i][j]))
    #     pairs.sort(key=lambda x: -x[1])
    #     for pair in pairs:
    #         print(pair[0], pair[1])
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
            print(pairs[j][0], pairs[j][1])
        print()
    sys.stdout.close()
    sys.stdout = standard_out


if __name__ == '__main__':
    drive_TF_IDF()
