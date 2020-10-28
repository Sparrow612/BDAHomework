import pandas as pd
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split


def find(src, tar):
    for key in src:
        if src[key] == tar:
            return key
    return -1


def get_sim(list_x, list_y):
    res = 0
    d_x = 0
    d_y = 0
    for val1, val2 in zip(list_x, list_y):
        res += val1 * val2
        d_x += val1 ** 2
        d_y += val2 ** 2
    return res / (math.sqrt(d_x * d_y))


class UserCF:
    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path, index_col=None)
        self.train_ratings, self.test_ratings = train_test_split(self.ratings, test_size=0.2)

        print("电影总数:" + str(len(set(self.ratings['movieId'].values.tolist()))))
        print("用户总数:" + str(len(set(self.ratings['userId'].values.tolist()))))
        print("训练集电影数:" + str(len(set(self.train_ratings['movieId'].values.tolist()))))
        print("训练集用户数:" + str(len(set(self.train_ratings['userId'].values.tolist()))))
        print("测试集电影数:" + str(len(set(self.test_ratings['movieId'].values.tolist()))))
        print("测试集用户数:" + str(len(set(self.test_ratings['userId'].values.tolist()))))

        self.train_ratings_pivotDF = pd.pivot_table(self.train_ratings[['userId', 'movieId', 'rating']],
                                                    columns=['movieId'],
                                                    index=['userId'], values='rating', fill_value=0)
        # 如果没有看过这个电影（没有留下评分），默认评分为0

        self.movies = dict(enumerate(list(self.train_ratings_pivotDF.columns)))
        self.users = dict(enumerate(list(self.train_ratings_pivotDF.index)))
        self.ratings = self.train_ratings_pivotDF.values.tolist()

        self.n = len(self.ratings)  # n users
        self.user_sim_matrix = np.zeros((self.n, self.n), dtype=np.float32)

        self.user_most_sim = dict()
        self.m = 10  # get m most similar users, recommend 10 movies

        self.user_interest = np.zeros((self.n, len(self.ratings[0])), dtype=np.float32)  # 用户对电影可能的兴趣值
        self.user_rec = dict()

    def build_user_sim(self):
        for u in range(self.n - 1):
            for v in range(u + 1, self.n):
                self.user_sim_matrix[u, v] = get_sim(self.ratings[u], self.ratings[v])
                self.user_sim_matrix[v, u] = self.user_sim_matrix[u, v]

    def get_most_sim(self):
        for i in range(self.n):
            self.user_most_sim[i] = sorted(enumerate(list(self.user_sim_matrix[i])), key=lambda x: -x[1])[:self.m]

    def get_rec_vals(self):
        for i in range(self.n):
            for j in range(len(self.ratings[i])):
                if self.ratings[i][j] == 0:
                    v = 0
                    for u, sim in self.user_most_sim[i]:
                        v += self.ratings[u][j] * sim
                    self.user_interest[i, j] = v

    def init(self):
        self.build_user_sim()
        self.get_most_sim()
        self.get_rec_vals()

    def recommend(self):
        self.init()

        for i in range(self.n):
            self.user_rec[i] = sorted(enumerate(list(self.user_interest[i])), key=lambda x: -x[1])[:self.m]

        rec_list = []
        for key, val in self.user_rec.items():
            uid = self.users[key]
            for m, v in val:
                mid = self.movies[m]
                rec_list.append([uid, mid])

        with open('movie.csv', 'w+') as f:
            f.write('userId' + ',' + 'movieId\n')
            for uid, mid in rec_list:
                f.write(str(uid) + ',' + str(mid) + '\n')
        f.close()

        return np.array(rec_list)

    def eval(self):
        rec_list = self.recommend()

        tp = 0
        rec = 0
        test_ratings_pviotDF = pd.pivot_table(self.test_ratings[['userId', 'movieId', 'rating']], columns=['movieId'],
                                              index=['userId'], values='rating', fill_value=0)

        test_users = dict(enumerate(list(test_ratings_pviotDF.index)))
        test_movies = dict(enumerate(list(test_ratings_pviotDF.columns)))
        test_ratings = test_ratings_pviotDF.values.tolist()

        good_movies = 0

        for movs in test_ratings:
            for m in movs:
                if m: good_movies += 1

        for uid, mid in rec_list:
            u = find(test_users, uid)
            v = find(test_movies, mid)
            if u == -1: continue
            rec += 1
            if v == -1: continue
            if test_ratings[u][v]:
                tp += 1
        return {'precision': tp / rec, 'recall': tp / good_movies}


p = os.path.join('ml-latest-small', 'ratings.csv')
usercf = UserCF(p)
print(usercf.eval())
