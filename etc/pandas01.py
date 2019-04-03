import pandas as pd
import numpy as np
train = pd.read_csv("data/titanic_train.csv")
train.shape
train.head()
train_list = train.values.tolist()

train.head()
train['Fare_revised'] =train['Fare'] *1000
train.drop('Fare_revised', axis=1, inplace=True)
train.head()

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
train.index.values
train.head()
train.info()

train.loc[train['Age']>30,:]
train_sorted = train.sort_values(by=["Age"], ascending=False)
train_sorted.groupby("Pclass").agg([np.sum, np.mean])

agg_format = {'Age':np.max, 'Pclass':np.sum, 'Fare':np.mean}
train_sorted.groupby("Pclass").agg(agg_format)
train.isna().sum()

train.apply(lambda x : x.count())