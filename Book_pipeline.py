#%%

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import mglearn

c = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    c.data, c.target, random_state=0)

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

svm = SVC()
svm.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
print('Правильность на тестовом наборе: {:.2f}'.format(
    svm.score(X_test_scaled, y_test)))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print('Наилучшее значение правильности при перекрёстной проверке: {:.2f}'.format(
    grid.best_score_))
print('Наилучшее значение правильности на тестовой выборке: {:.2f}'.format(
    grid.score(X_test_scaled, y_test)))
print('Наилучшие параметры классификатора: ', grid.best_params_)

#mglearn.plots.plot_improper_processing()

pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
pipe.fit(X_train, y_train)

#
#   Сначала будет вызван метод fit() объекта scaler, в результате чего обучающие
#   данные будут преобразованы MinMaxScaler-ом, а затем будет построена модель SVM
#   на основе масштабированных данных
#

print('Правильность на тестовом наборе при применении конвейера: {:.2f}'.format(
    pipe.score(X_test, y_test)))

new_param_gr = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=new_param_gr, cv=5)
grid.fit(X_train, y_train)
print('Наилучшее значение правильности при перекрёстной проверке: {:.2f}'.format(
    grid.best_score_))
print('Наилучшее значение правильности на тестовой выборке: {:.2f}'.format(
    grid.score(X_test, y_test)))
print('Наилучшие параметры классификатора: ', grid.best_params_)

mglearn.plots.plot_proper_processing()


# %%
#
#   Для лучшей иллюстрации утечки информации из тестового набора данных при обычной
#   перекрестной проверке используется модель гребневой регрессии со случайными данными
#

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)

print('Форма массива X_selected: {}'.format(X_selected.shape))
print('Правильность перекрёстной проверки (cv только для регрессии): {:.2f}'.format(
    np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))))

#
#   В данном случае действительные результаты могут быть получены только при
#   использовании конвейера, т.к. при этом отбор признаков осуществляется внутри цикла
#   перекрёстной проверки
#

pipe = Pipeline([('select', SelectPercentile(score_func=f_regression, percentile=5)),
    ('ridge', Ridge())])
print('Правильность перекрёстной проверки (конвейер): {:.2f}'.format(
    np.mean(cross_val_score(pipe, X, y, cv=5))))

# %%

#
#   Иллюстрация работы методов Pipeline в виде циклов
#

def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        X_transformed = estimator.fit_transform(X_transformed, y)
    self.steps[-1][1].fit(X_transformed, y)
    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        X_transformed = step[1].transform(X_transformed)
    return self.steps[-1][1].predict(X_transformed)