import random
import sys
from Homework2 import pre_processor


def generate_random_weight():
    word_randoms = {}
    word_pool = pre_processor.pre_process()
    for words in word_pool:
        for word in words:
            if word in word_randoms.keys():
                tmp = word_randoms[word][0] * word_randoms[word][1]
                word_randoms[word][0] = (tmp + random.random() * 100) / (word_randoms[word][1] + 1)
                word_randoms[word][1] += 1
            else:
                word_randoms[word] = [random.random() * 100, 1]
    wordlist = [(key, value[0]) for key, value in word_randoms.items()]
    wordlist.sort(key=lambda x: -x[1])
    return wordlist


if __name__ == '__main__':
    wordlist = generate_random_weight()
    print(wordlist)
    standard_out = sys.stdout
    sys.stdout = open('result/Random_Weight.txt', 'w')
    for i in range(100):
        print(wordlist[i][0] + '\t' + str(wordlist[i][1]))
    sys.stdout.close()
    sys.stdout = standard_out
