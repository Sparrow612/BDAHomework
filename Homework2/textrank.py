import sys
from textrank4zh import TextRank4Keyword
from Homework2 import pre_processor


def keywords_extraction(text):
    tr = TextRank4Keyword()
    tr.analyze(text)
    keywords = tr.get_keywords(num=100)
    return keywords


if __name__ == '__main__':
    word_pool = pre_processor.pre_process()
    text = ""
    for words in word_pool:
        text += ' '.join(words)
    res = keywords_extraction(text)
    res.sort(key=lambda x: -x['weight'])
    standard_out = sys.stdout
    sys.stdout = open('result/TextRank.txt', 'w')
    for pair in res:
        print(pair['word']+'\t'+str(pair['weight']))
    sys.stdout.close()
    sys.stdout = standard_out
