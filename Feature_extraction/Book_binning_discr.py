#%%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split=3, max_leaf_nodes=100).fit(X, y)
plt.plot(line, reg.predict(line), label = 'Дерево решений')

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label='Линейная регрессия')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Выход регрессии')
plt.xlabel('Входной признак')
plt.legend(loc='best')

# %%
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)

#print('Категории: {}'.format(bins))
which_bin = np.digitize(X, bins=bins)

#print('\nТочки данных:\n', X[:3])
#print('\nКатегории для точек данных:\n', which_bin[:3])

encoder = OneHotEncoder(sparse=False)
X_binned = encoder.fit_transform(which_bin)

#print('Двоичное распределение точек данных по категориям:\n',X_binned[:3])
#print('Форма массива X_binned:', X_binned.shape)

line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)

plt.plot(
    line, reg.predict(line_binned), label='Линейная регрессия после биннинга')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(
    line, reg.predict(line_binned), label='Дерево решений после биннинга')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel('Выход регрессии')
plt.xlabel('Входной признак')
# %%
