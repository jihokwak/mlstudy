import os
import re

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import utils

data_set = tf.keras.utils.get_file(
    fname='imdb.tar.gz',
    origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    extract=True
)

data_set

def directory_data(directory) :
    data = {}
    data['review'] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path),'r', encoding='utf-8') as file :
            data['review'].append(file.read())

    return pd.DataFrame.from_dict(data)

def data(directory) :
    pos_df = directory_data(os.path.join(directory, "pos"))
    neg_df = directory_data(os.path.join(directory, "neg"))
    pos_df["sentiment"] = 1
    neg_df["sentiment"] = 0
    return pd.concat([pos_df, neg_df])

train_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "train"))
test_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "test"))

train_df.head()
train_df.shape

reviews = train_df.review.tolist()

tokenized_reviews = [r.split() for r in reviews]
review_len_by_token = [len(t) for t in tokenized_reviews]
review_len_by_syllable = [len(s.replace(' ','')) for s in reviews]

import matplotlib.pyplot as plt
plt.hist(review_len_by_token, bins=50, color='r', alpha=0.5, label='word')
plt.hist(review_len_by_syllable, bins=50, color='b', alpha=0.5, label='alphabet')
plt.yscale("log", nonposy="clip")
plt.title("Reivew Length Histogram")
plt.ylabel("Number of Reviews")
plt.xlabel("Reivew Length")

pd.Series(review_len_by_token).describe()

plt.boxplot(review_len_by_token, labels=['Eojeol'], showmeans=True)

plt.figure(figsize=(12,5))
plt.boxplot(review_len_by_syllable, labels=['syllable'], showmeans=True)

from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="black", width=800, height= 600).generate(' '.join(train_df.review))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

import seaborn as sns

sentiment = train_df.sentiment.value_counts()
sentiment
fig, ax = plt.subplots(ncols=1)
fig.set_size_inches(6,3)
sns.countplot(train_df.sentiment)