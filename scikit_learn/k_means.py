import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
datasets = load_iris()
train_input, test_input,train_label, test_label = train_test_split(datasets.data, datasets.target, test_size=0.2, random_state=0)

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
k_means.fit(train_input)

k_means.labels_

predict_cluster =k_means.predict(test_input)
np_arr = np.array(predict_cluster)

np_arr[np_arr ==0],np_arr[np_arr ==1],np_arr[np_arr ==2] = 3,4,5
np_arr[np_arr ==3],np_arr[np_arr ==4],np_arr[np_arr ==5] = 1,0,2
predict_label = np_arr.tolist()
print('test accuracy : {:.2f}'.format(np.mean(predict_label == test_label)))

