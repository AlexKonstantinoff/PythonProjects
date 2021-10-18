#%%
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import mglearn
from Book_ridge import ridge

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
lasso00001 = Lasso(alpha=0.0001, max_iter=10000).fit(X_train, y_train)
lasso1 = Lasso(max_iter=10000).fit(X_train, y_train)

print("Правильность на обучающем наборе (elastic): {:.2f}".format(lasso.score(X_train, y_train)))
print("Правильность на тестовом наборе (elastic): {:.2f}".format(lasso.score(X_test, y_test)))
print("Количество использованных признаков (elastic): {}".format(np.sum(lasso.coef_ != 0)))

#elastic = ElasticNet(alpha=0.0008, max_iter=100000).fit(X_train, y_train)

#print("Правильность на обучающем наборе (elastic): {:.2f}".format(elastic.score(X_train, y_train)))
#print("Правильность на тестовом наборе (elastic): {:.2f}".format(elastic.score(X_test, y_test)))
#print("Количество использованных признаков (elastic): {}".format(np.sum(elastic.coef_ != 0)))

plt.plot(lasso.coef_, 's', label='Лассо alpha=0.01')
plt.plot(lasso00001.coef_, '^', label='Лассо alpha=0.0001')
plt.plot(lasso1.coef_, 'v', label='Лассо alpha=1')
plt.plot(ridge.coef_, 'o', label='Гребневая регрессия с alpha=0.1')

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-40, 40)
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
# %%
