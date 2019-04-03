#pip install JPype1-0.5.7-cp27-none-win_amd64.whl
#https://konlpy-ko.readthedocs.io/ko/v0.5.1/install/
import pandas as pd
import re

train_df = pd.read_csv('data/nsmc/ratings_train.txt', sep='\t')
train_df = train_df.fillna(' ')
train_df.document = train_df.document.apply(lambda x : re.sub(r'\d+', ' ', x))

test_df = pd.read_csv('data/nsmc/ratings_test.txt', sep="\t")
test_df = test_df.fillna(' ')
test_df.document = test_df.document.apply(lambda x : re.sub('\d+', ' ', x))

from konlpy.tag import Mecab
mecab = Mecab()

def tw_tokenizer(text) :
    tokens_ko = mecab.morphs(text)
    return tokens_ko

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])

logreg = LogisticRegression(random_state=0)
params = {'C':[1,3.5,4.5,5.5,10]}
gscv = GridSearchCV(logreg, param_grid=params, cv=3, scoring='accuracy', verbose=2)
gscv.fit(tfidf_matrix_train, train_df['label'])
gscv.best_params_
gscv.best_score_

from sklearn.metrics import accuracy_score

tfidf_matrix_test = tfidf_vect.transform(test_df['document'])
preds = gscv.predict(tfidf_matrix_test)
accuracy_score(test_df['label'], preds)
