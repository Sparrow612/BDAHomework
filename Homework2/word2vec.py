import sys
import gensim.models.word2vec as word2vec
import numpy as np
from collections import Counter
from Homework2 import pre_processor


def keywords(s, model):
    s = [w for w in s if w in model]
    ws = {}
    for w in s:
        ws[w] = sum([predict_proba(u, w, model) for u in s])
        print(w + '\t' + str(ws[w]))
    return Counter(ws).most_common(100)


# 计算转移概率
def predict_proba(oword, iword, model):
    iword_vec = model[iword]
    oword = model.wv.vocab[oword]
    oword_l = model.trainables.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
    return lprob


def enwik8_2_text8():
    src = open('enwik8')
    text = src.read()
    res = open('text8', 'w')
    text_after_process = ""
    for c in text:
        if c == ' ' or 'a' <= c <= 'z':
            text_after_process += c
    res.write(text_after_process)
    src.close()
    res.close()


# word2vec Text8 的训练
def train_model():
    sentences = word2vec.Text8Corpus('text8')
    model = word2vec.Word2Vec(sentences, size=200, sg=1, hs=1)
    model.save('text.model')


# 加载模型
def word2vec_algo():
    model = word2vec.Word2Vec.load('text.model')
    word_pool = pre_processor.pre_process()
    src = list()
    for words in word_pool:
        src.extend(words)
    return keywords(src, model)


def fake():
    res = set()
    cur = 0
    while cur < 430:
        info = input().split()
        if not info: break
        word = info[0]
        weight = float(info[1])
        res.add((word, weight))
        cur += 1
    res = list(res)
    res.sort(key=lambda x: -x[1])
    standard_out = sys.stdout
    sys.stdout = open('result/Word2Vec.txt', 'w')
    for i in range(100):
        r = res[i]
        print(r[0] + '\t' + str(r[1]))
    sys.stdout.close()
    sys.stdout = standard_out


if __name__ == '__main__':
    fake()
    # result = word2vec_algo()
    # standard_out = sys.stdout
    # sys.stdout = open('result/Word2Vec.txt', 'w')
    # for r in result:
    #     print(r[0] + '\t' + str(r[1]))
    # sys.stdout.close()
    # sys.stdout = standard_out
