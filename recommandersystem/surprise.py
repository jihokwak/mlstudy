from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)
algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)
type(predictions)
len(predictions)
predictions[:5]

[(pred.uid, pred.iid, pred.r_ui) for pred in predictions[:3]]

uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)

accuracy.rmse(predictions)


#파일 로딩
import pandas as pd
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
ratings.to_csv("data/ml-latest-small/ratings_noh.csv", index=False, header=False)

from surprise import Reader

reader = Reader(line_format="user item rating timestamp", sep=",", rating_scale=(0.5,5))
data = Dataset.load_from_file("data/ml-latest-small/ratings_noh.csv", reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

#판다스 로딩
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5,5.0))

data = Dataset.load_from_df(ratings[['userId','movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

