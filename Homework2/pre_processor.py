import os
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')

lemmas = WordNetLemmatizer()

root_dir = './essay'

special_words = ['author', 'refer', 'introduct', 'however']
stoplist = stopwords.words("english")
word_pool = list()


def get_essay():
    essays = os.listdir(root_dir)
    essays.remove('.DS_Store')
    return essays


def lemma(word: str) -> str:
    upo_list = ['a', 'v', 'n', 'r']
    return min([lemmas.lemmatize(word, upo) for upo in upo_list], key=lambda x: len(x))


def pre_process():
    essays = get_essay()
    for essay in essays:
        path = os.path.join(root_dir, essay)
        file = open(path, 'rb')
        content = file.read()
        content = content.decode(encoding="utf-8", errors="replace").lower()
        file.close()
        words = content.split()
        res = list()
        for w in words:
            w = list(w)
            w = ''.join([c for c in w if c not in string.punctuation])
            w = lemma(w)
            if len(w) == 1 and 'a' <= w <= 'z': continue
            if not bool(re.match(r'[a-z]+$', w)): continue
            if w in special_words or w in stoplist: continue
            res.append(w)
        word_pool.append(res)
    return word_pool


if __name__ == '__main__':
    w = 'ahang123'
    print(lemma("gone"))
