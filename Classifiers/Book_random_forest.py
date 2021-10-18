#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mglearn
from matplotlib import pyplot as plt

c = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, random_state=0)
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.3f}'.format(forest.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.3f}'.format(forest.score(X_test, y_test)))

def feature_importances(model):
    n_features = c.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), c.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")

feature_importances(forest)

#X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#forest = RandomForestClassifier(n_estimators=10, random_state=2)
#forest.fit(X_train, y_train)

#fig, axes= plt.subplots(2, 3, figsize=(200, 100))
#for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#    ax.set_title("Дерево {}".format(i))
#    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

#mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)

#axes[-1, -1].set_title("Случайный лес")
#mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
#%%