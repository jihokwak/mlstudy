import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


movies = pd.read_csv('data/ml-latest-small/movies.csv')
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
ratings = pd.merge(ratings, movies, on="movieId", how="left")
ratings_matrix = ratings.pivot_table("rating","userId", "title", fill_value=0)

item_sim = cosine_similarity(ratings_matrix.T)
item_sim_df =pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

#개인화
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def predict_rating(rating_arr, item_sim_arr) :
    ratings_pred = rating_arr.dot(item_sim_arr) / np.abs(item_sim_arr).sum(axis=1)
    return ratings_pred

rating_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
rating_pred_matrix = pd.DataFrame(rating_pred, index=ratings_matrix.index, columns= ratings_matrix.columns)

def predict_rating_topsim(ratings_arr, item_sim_arr, n=20) :
    pred = np.zeros(ratings_arr.shape)
    for col in range(ratings_arr.shape[1]):
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row,:][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))
    return pred

ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
ratings_pred_matrix = pd.DataFrame(ratings_pred, index=ratings_matrix.index, columns= ratings_matrix.columns)
get_mse(ratings_pred, ratings_matrix.values)

#추천
user_rating_id = ratings_matrix.loc[9,:]
user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:10]

def get_unseen_movies(ratings_matrix, userId) :
    user_rating = ratings_matrix.loc[userId, :]
    already_seen = user_rating[user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [movie for movie in movies_list if movie not in already_seen]
    return unseen_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

unseen_list = get_unseen_movies(ratings_matrix, 9)

recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies = pd.DataFrame(recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
recomm_movies

