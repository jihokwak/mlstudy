import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
sent = {"휴일인 오늘 도 서쪽을 중심으로 폭염이 이어졌는데요, 내일 은 반가운 비 소식이 있습니다.",
        "폭염을 피해서 휴일에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니다."}

tfidf_vect = TfidfVectorizer()
tfidf_matrix = tfidf_vect.fit_transform(sent)
tfidf_matrix.shape
tfidf_matrix[0:1]

#코사인 유사도
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(tfidf_matrix[0,:], tfidf_matrix[1,:])

#L1정규화 + 유클리디언 유사도(=L2거리)
from sklearn.metrics.pairwise import euclidean_distances
l1_normalize = lambda v : v/np.sum(v)
tfidf_norm_l1 = l1_normalize(tfidf_matrix)
euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])

#L1정규화 + 맨하탄 유사도(=L1거리)
from sklearn.metrics.pairwise import manhattan_distances
l1_normalize = lambda v : v/np.sum(v)
tfidf_norm_l1 = l1_normalize(tfidf_matrix)
manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
