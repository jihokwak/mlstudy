from nltk import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
text_sample = "Beto O’Rourke, the youthful Texan who gained a national following with his long-shot election battle against U.S. Senator Ted Cruz last year, told a Texas TV station on Wednesday he will seek the 2020 Democratic presidential nomination."
sentences = sent_tokenize(text_sample)
len(sentences)

sentence = sentences[0]
word = word_tokenize(sentence)
word

#샘플문장
text_sample = "Beto O’Rourke, the youthful Texan who gained a national following with his long-shot election battle against U.S. Senator Ted Cruz last year, told a Texas TV station on Wednesday he will seek the 2020 Democratic presidential nomination."
#토큰화
def tokenize_text(text) :
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens
word_tokens = tokenize_text(text_sample)
#스톱워드제거
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens :
    filtered_words=[]
    for word in sentence :
        word = word.lower()
        if word not in stopwords :
            filtered_words.append(word)
    all_tokens.append(filtered_words)

#스테밍 & 레마타이징
from nltk import LancasterStemmer
stemmer = LancasterStemmer()
stemmer.stem('happiest')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
lemma.lemmatize('happiest', 'a')

#BOW(출현횟수에 기반하여 문맥해석이 되지 않음)
import numpy as np
data = np.array([3,1,2])
row_pos = np.array([0,0,1])
col_pos = np.array([0,2,1])
from scipy import sparse
sparse_coo = sparse.coo_matrix((data,(row_pos, col_pos)))
sparse_coo.toarray()

#예제1
from sklearn.datasets import fetch_20newsgroups
news_data = fetch_20newsgroups(subset="all", random_state=156)
type(news_data)
print(news_data.keys())
import pandas as pd

print("target 클래스의 값과 분포도 \n", pd.Series(news_data.target).value_counts().sort_index())
print("target 클래스의 이름들 \n", news_data.target_names)

train_news = fetch_20newsgroups(subset='train', remove=('headers', 'footers','quotes'), random_state=156)
test_news = fetch_20newsgroups(subset='test', remove=('headers', 'footers','quotes'), random_state=156)

X_train = train_news.data
y_train = train_news.target
X_test = test_news.data
y_test = test_news.target

#카운트기반 예측모델
from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
cnt_vect.fit(X_train, y_train)
X_train_cnt_vect = cnt_vect.transform(X_train)
X_test_cnt_vect = cnt_vect.transform(X_test)

X_train_cnt_vect.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr_clf = LogisticRegression()
lr_clf.fit(X_train_cnt_vect, y_train)
pred = lr_clf.predict(X_test_cnt_vect)
accuracy_score(y_test, pred)

#TF-IDF 기반 예측모델
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

lr_clf = LogisticRegression()
lr_clf.fit(X_train_tfidf_vect, y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
accuracy_score(y_test, pred)

#TFIDF 하이퍼파라미터 변경
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)
lr_clf = LogisticRegression()
lr_clf.fit(X_train_tfidf_vect, y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
accuracy_score(y_test, pred)

#TFIDF 하이퍼파라미터 최적화(GSCV)
from sklearn.model_selection import GridSearchCV
params = {'C':[0.01,0.1,1,5,10]}
grid_cv_lr = GridSearchCV(lr_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_lr.fit(X_train_tfidf_vect, y_train)

grid_cv_lr.best_params_ # {'C': 10}
pred = grid_cv_lr.predict(X_test_tfidf_vect)
accuracy_score(y_test, pred)

#파이프라인 사용
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),
                     ('logreg', LogisticRegression(C=10))
                     ])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
accuracy_score(y_test, pred)

#파이프라인과 GSCV 동시사용
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english')),
    ('logreg', LogisticRegression())
])

params = {
    'tfidf_vect__ngram_range':[(1,1),(1,2),(1,3)],
    'tfidf_vect__max_df':[100,300,700],
    'logreg__C':[1,5,10]
}

gscv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=2)
gscv_pipe.fit(X_train, y_train)
gscv_pipe.best_params_ # {'logreg__C': 10, 'tfidf_vect__max_df': 700, 'tfidf_vect__ngram_range': (1, 2)}
gscv_pipe.best_score_ # 0.755524129397207
pred = gscv_pipe.predict(X_test)
accuracy_score(y_test, pred) # 0.7019383961763144

#스테밍 & Tdidf벡터라이징&로지스틱회귀모델
import re
X_train = [re.sub(r'[\W\d]', ' ', x) for x in X_train]
X_test = [re.sub(r'[\W\d]', ' ', x) for x in X_test]
X_train = [x.lower() for x in X_train]
X_test = [x.lower() for x in X_test]

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
for idx, text in enumerate(X_train) :
    X_train[idx] = " ".join([lemma.lemmatize(word, "v") for word in word_tokenize(text)])
for idx, text in enumerate(X_test) :
    X_test[idx] = " ".join([lemma.lemmatize(word, "v") for word in word_tokenize(text)])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score
class CsrConverter(BaseEstimator, TransformerMixin) :

    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return sparse.csr_matrix(X)


class DenseMatrixConverter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()

pipeline = Pipeline([
    ("tfidf_vect", TfidfVectorizer(stop_words='english', max_df=700, ngram_range=(1,2))),
    #('csr', CsrConverter()),
    ('dense', DenseMatrixConverter()),
    #("logreg", LogisticRegression(C=10))
    #('svc', SVC(random_state=0))
    ('nb_clf', GaussianNB(priors=None))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
accuracy_score(y_test, pred)
