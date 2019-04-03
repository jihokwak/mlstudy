import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

movies = pd.read_csv("data/tmdb-5000-movie-dataset/tmdb_5000_movies.csv")

movies.shape

pd.set_option('max_colwidth', 100)
movies[['genres', 'keywords']][:1]

from ast import literal_eval
movies.genres = movies.genres.apply(literal_eval)
movies.keywords = movies.keywords.apply(literal_eval)

movies.genres = movies.genres.apply(lambda x : [y['name'] for y in x])
movies.keywords = movies.keywords.apply(lambda x : [y['name'] for y in x])

from sklearn.feature_extraction.text import CountVectorizer

movies['genres_literal'] = movies.genres.apply(" ".join)
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies.genres_literal)
genre_mat.shape

from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat)
genre_sim.shape

genre_sim_sorted_idx = genre_sim.argsort()[:, ::-1]

def find_sim_movie(df, sorted_idx, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_idx = title_movie.index.values
    similar_indices = sorted_idx[title_idx, :(top_n)]
    print(similar_indices)
    similar_indices = similar_indices.reshape(-1)
    return df.iloc[similar_indices]

find_sim_movie(movies, genre_sim_sorted_idx, "The Godfather", 10).title

movies[['title', 'vote_average', 'vote_count']].sort_values('vote_average', ascending=False)[:10]

#IMDB 가중 평점 공식 : (v/(v+m) * R + (m/(v+m)) * C

percentile = 0.6
m = movies.vote_count.quantile(percentile)
C = movies.vote_average.mean()

def weighted_vote_average(record) :
    v = record['vote_count']
    R = record['vote_average']
    return ((v/(v+m)) * R) + ((m/(v+m)) * C)

movies['weighted_vote'] = movies.apply(weighted_vote_average, axis=1)

movies[['title','vote_average','weighted_vote','vote_count']].sort_values(by="weighted_vote", ascending=False)[:10]

def find_sim_movie(df, sorted_idx, title_nm, top_n=10):
    title_movie = df[df.title == title_nm]
    title_idx = title_movie.index.values

    similar_indices = sorted_idx[title_idx, :(top_n*2)]
    similar_indices = similar_indices.reshape(-1)
    similar_indices = similar_indices[similar_indices != title_idx]

    return df.iloc[similar_indices].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie(movies, genre_sim_sorted_idx, "The Godfather", 10)
similar_movies[['title','vote_average','weighted_vote']]