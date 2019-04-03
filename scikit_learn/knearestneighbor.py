from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
datasets = load_iris()
train_input, test_input,train_label, test_label = train_test_split(datasets.data, datasets.target, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_input, train_label)

import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
knn.predict(new_input)

predict_label = knn.predict(test_input)
print("test accuracy {:.2f}".format(np.mean(predict_label == test_label)))
