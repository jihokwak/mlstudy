from surprise.model_selection import cross_validate, GridSearchCV
from surprise import Dataset, SVD, Reader
import pandas as pd


ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

algo = SVD(random_state=0)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

param_grid = {'n_epochs':[20,40,60], 'n_factors':[50,100,200]}
gs = GridSearchCV(SVD, param_grid=param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
gs.best_params
#{'rmse': {'n_epochs': 20, 'n_factors': 50}, 'mae': {'n_epochs': 20, 'n_factors': 50}}

from surprise.dataset import DatasetAutoFolds

def build_model(filename) :
    reader = Reader(line_format='user item rating timestamp', sep=",", rating_scale=(0.5,5))
    data_folds = DatasetAutoFolds(ratings_file=filename, reader=reader)
    trainset = data_folds.build_full_trainset()
    algo = SVD(n_epochs=20, n_factors=50, random_state=0)
    return algo.fit(trainset)

def recomm_movie_by_surprise(algo, movies, userId, top_n=10):

    unseen_movies = [movie for movie in movies.movieId if movie not in ratings.loc[ratings.userId == 9, 'movieId']]

    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]
    predictions.sort(key=lambda pred : pred.est, reverse=True)
    top_predictions = predictions[:top_n]

    top_movie_ids = [ int(pred.iid) for pred in top_predictions ]
    top_movie_rating = [ pred.est for pred in top_predictions ]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)].title

    return pd.DataFrame({"영화ID":top_movie_ids,"평점":top_movie_rating, "영화제목":top_movie_titles}).reset_index(drop=True)

algo = build_model("data/ml-latest-small/ratings_noh.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")
recomm_movie_by_surprise(algo, movies, 2)
