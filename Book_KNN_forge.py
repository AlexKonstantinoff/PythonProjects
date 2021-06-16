#%%
from matplotlib import pyplot as plt
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
import mglearn
import numpy as np
import pandas as pd

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

#print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))

#print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3,  figsize=(10, 3))

for n, a in zip([2, 4, 11], axes):
    clf = KNeighborsClassifier(n_neighbors=n).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=a, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=a)
    a.set_title("Количество соседей:{}".format(n))
    a.set_xlabel("Признак 0")
    a.set_ylabel("Признак 1")
axes[0].legend(loc=3)


# %%
