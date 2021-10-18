#%%
import mglearn as mg
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mg.plots.plot_linear_regression_wave()

X, y = mg.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.2f}'.format(lr.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.2f}'.format(lr.score(X_test, y_test)))
# %%
