#%%
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

c = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, random_state=0)

gbrt = GradientBoostingClassifier(max_depth=1, random_state=0)
gbrt.fit(X_train, y_train)

print('Точность на обучающем наборе: {:.3f}'.format(gbrt.score(X_train, y_train)))
print('Точность на тестовом наборе: {:.3f}'.format(gbrt.score(X_test, y_test)))

def feature_importances(model):
    n_features = c.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), c.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")

feature_importances(gbrt)
# %%
