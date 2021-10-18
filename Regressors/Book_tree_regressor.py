#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import data
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

prices = pd.read_csv("ram_price.csv")

train_data = prices[prices.date < 2000]
test_data = prices[prices.date >= 2000]

X_train = train_data.date[:, np.newaxis]
y_train = np.log(train_data.price)

tree = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = prices.date[:, np.newaxis]
tree_pred = tree.predict(X_all)
lr_pred = linear_reg.predict(X_all)

price_tree = np.exp(tree_pred)
price_lr = np.exp(lr_pred)

plt.semilogy(train_data.date, train_data.price, label="Обучающие данные")
plt.semilogy(test_data.date, test_data.price, label="Тестовые данные")
plt.semilogy(prices.date, price_tree, label="Прогнозы дерева")
plt.semilogy(prices.date, price_lr, label="Прогнозы линейной регрессии")
plt.legend()

#plt.semilogy(prices.date, prices.price)
plt.xlabel("Год")
plt.ylabel("Цена (в долларах) за Мбайт")
# %%
