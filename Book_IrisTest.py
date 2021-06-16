from numpy import ceil
from pandas.io.pytables import ClosedFileError


#%%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn

iris_ds = load_iris()

print("Iris data: {}".format(iris_ds.data[:5]))
print("Iris target: {}".format(iris_ds['target']))

X_train, X_test, y_train, y_test = train_test_split(
    iris_ds.data, iris_ds['target'], random_state=0
)

iris_dataframe = pd.DataFrame(X_train, columns=iris_ds.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, 
figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8,
cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

#X_new = np.array([[5, 2.9, 1, 0.2]])
#print(X_new.shape)

#prediction = knn.predict(X_new)
#print("Прогноз: {}".format(prediction))
#print("Спрогнозированная метка: {}".format(iris_ds.target_names[prediction]))

y_predicted = knn.predict(X_test)
print("Прогнозы для тестового набора данных\n {}".format(y_predicted))

print("Качество предсказательной модели: {:.2f}".format(np.mean(y_predicted == y_test)))

print("Качество предсказательной модели: {:.2f}".format(knn.score(X_test, y_test)))
# %%
