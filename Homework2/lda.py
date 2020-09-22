from gensim import corpora, models
from Homework2 import pre_processor
import sys


def LDA_model(words_list):
    dictionary = corpora.Dictionary(words_list)
    corpus = [dictionary.doc2bow(words) for words in words_list]
    return models.ldamodel.LdaModel(corpus=corpus, num_topics=1, id2word=dictionary, passes=10)


if __name__ == '__main__':
    lda_model = LDA_model(pre_processor.pre_process())
    result = lda_model.show_topic(0, 100)
    result.sort(key=lambda x: -x[1])
    standard_out = sys.stdout
    sys.stdout = open('result/LDA_dic.txt', 'w')
    for r in result:
        print(r[0] + '\t' + str(r[1]))
    sys.stdout.close()
    sys.stdout = standard_out
