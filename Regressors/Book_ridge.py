#%%
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mglearn as mg

X, y = mg.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=0)

ridge = Ridge(alpha=0.1).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge1 = Ridge().fit(X_train, y_train)

lr = LinearRegression().fit(X_train, y_train)

print("Правильность на обучающем наборе: {:.3f}".format(ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(ridge.score(X_test, y_test)))

#plt.plot(ridge.coef_, 's', label="Ridge a=0.1")
#plt.plot(ridge1.coef_, '^', label="Ridge a=1")
#plt.plot(ridge10.coef_, 'v', label="Ridge a=10")
#plt.plot(lr.coef_, 'o', label="Linear")

#plt.xlabel("Индекс коэффициента")
#plt.ylabel("Оценка коэффициента")
#plt.hlines(0, 0, len(lr.coef_))
#plt.ylim(-40, 40)
#plt.legend()

#mg.plots.plot_ridge_n_samples()
# %%
