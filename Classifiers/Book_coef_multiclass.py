#%%
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import numpy as np
import mglearn

X, y = make_blobs(random_state=67, n_samples=500)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

#plt.xlabel("Признак 0")
#plt.ylabel("Признак 1")
#plt.legend(["Класс 0", "Класс 1", "Класс 2"])

linearSVM = LinearSVC().fit(X, y)
#print("Форма коэффициента: ", linearSVM.coef_.shape)
#print("Форма константы: ", linearSVM.intercept_.shape)

mglearn.plots.plot_2d_classification(linearSVM, X, fill=True, alpha=.7)

for coef, intercept, color in zip(linearSVM.coef_, linearSVM.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.legend(['Класс 0', 'Класс 1', 'Класс 2', 'Линия класса 0', 'Линия класса 1', 'Линия класса 2'], 
loc=(1.01, 0.3))

# %%
