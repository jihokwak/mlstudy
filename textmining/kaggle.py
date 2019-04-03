from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

mercari_df = pd.read_csv("data/mercari/train.tsv", sep="\t")
mercari_df.info()
mercari_df.isna().sum()

y_train_df = mercari_df.price
plt.figure(figsize=(6,4))
sns.distplot(y_train_df, kde=False)

y_train_df = np.log1p(y_train_df)
sns.distplot(y_train_df, kde=False)

mercari_df.price = np.log1p(mercari_df.price)

mercari_df.shipping.value_counts()
sns.distplot(mercari_df.item_condition_id, bins=10)

boolean_cond = mercari_df.item_description == 'No description yet'
mercari_df[boolean_cond]['item_description'].count()

def split_cat(category_name) :
    try :
        return category_name.split("/")
    except:
        return ['Other Null','Other Null','Other Null']

mercari_df['cat_dae'], mercari_df['cat_jung'], mercari_df['cat_so'] = \
    zip(*mercari_df.category_name.apply(split_cat))

mercari_df.cat_dae.value_counts()
mercari_df.cat_jung.nunique()
mercari_df.cat_so.nunique()

mercari_df.category_name = mercari_df.category_name.fillna('Other Null')
mercari_df.brand_name = mercari_df.brand_name.fillna('Other Null')
mercari_df.item_description = mercari_df.item_description.fillna('Other Null')

mercari_df.isna().sum()

mercari_df.brand_name.nunique()
mercari_df.brand_name.value_counts()[:5]

mercari_df.name.nunique()
mercari_df.name[:10]

mercari_df.info()

pd.set_option('max_colwidth', 200)
mercari_df.item_description.str.len().mean()
mercari_df.item_description[:2]

cnt_vec = CountVectorizer()
X_name = cnt_vec.fit_transform(mercari_df.name)

tfidf_descp = TfidfVectorizer(max_features=50000, ngram_range=(1,3), stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df.item_description)
X_name.shape
X_descp.shape

from sklearn.preprocessing import LabelBinarizer

lb_brand_name = LabelBinarizer(sparse_output=True)
X_brand = lb_brand_name.fit_transform(mercari_df.brand_name)
lb_item_cond_id = LabelBinarizer(sparse_output=True)
X_item_cond_id = lb_item_cond_id.fit_transform(mercari_df.item_condition_id)
lb_shipping = LabelBinarizer(sparse_output=True)
X_shipping = lb_shipping.fit_transform(mercari_df.shipping)

lb_cat_dae = LabelBinarizer(sparse_output=True)
X_cat_dae = lb_cat_dae.fit_transform(mercari_df.cat_dae)
lb_cat_jung = LabelBinarizer(sparse_output=True)
X_cat_jung = lb_cat_jung.fit_transform(mercari_df.cat_jung)
lb_cat_so = LabelBinarizer(sparse_output=True)
X_cat_so = lb_cat_so.fit_transform(mercari_df.cat_so)

from scipy.sparse import hstack
import gc

def rmsle(y, y_pred) :
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_pred),2)))

def evaluate_org_price(y_test, preds):
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result

def model_train_predict(model, matrix_list):
    X=hstack(matrix_list).tocsr()
    X_train, X_test, y_train, y_test = train_test_split(X, mercari_df['price'], test_size=0.2, random_state=156)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    del X, X_train, X_test, y_train
    gc.collect()
    return preds, y_test

#릿지 회귀
linear_model = Ridge(solver = "lsqr", fit_intercept=False)

sparse_matrix_list = (X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(linear_model, sparse_matrix_list)
evaluate_org_price(y_test, linear_preds)

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(linear_model, sparse_matrix_list)
evaluate_org_price(y_test, linear_preds)

#LightGBM 회귀모델
from lightgbm import LGBMRegressor
sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.5, num_leaves=123, random_state=156, n_jobs=-1)
lgbm_preds, y_test = model_train_predict(lgbm_model, sparse_matrix_list)
evaluate_org_price(y_test, lgbm_preds)

geometry_mean = lambda x,y : np.sqrt(x*y)
geo_esemble_preds = np.array([geometry_mean(x,y) for x,y in zip(linear_preds, lgbm_preds)])
evaluate_org_price(y_test, geo_esemble_preds)

arith_esamble_preds = np.array([np.mean([x,y]) for x,y in zip(linear_preds, lgbm_preds)])
evaluate_org_price(y_test, arith_esamble_preds)

weight_esamble_preds = lgbm_preds*0.45 + linear_preds*0.55
evaluate_org_price(y_test, weight_esamble_preds)