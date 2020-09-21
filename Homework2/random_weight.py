import random
import sys
from Homework2 import pre_processor
from collections import defaultdict


# 生成第一个随机数
def generate_random_weight():
    pairs = defaultdict(float)
    word_pool = pre_processor.pre_process()
    for words in word_pool:
        for word in words:
            pairs[word] = random.random() * 100
    wordlist = [(key, value) for (key, value) in pairs.items()]
    return wordlist


if __name__ == '__main__':
    wordlist = generate_random_weight()
    standard_out = sys.stdout
    sys.stdout = open('result/Random_Weight.txt', 'w')
    for i in range(100):
        print(wordlist[i][0] + '\t' + str(wordlist[i][1]))
    sys.stdout.close()
    sys.stdout = standard_out
