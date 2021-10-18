#%%

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

pipe_long = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC(C=100))])

pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

print('Этапы конвейера:\n{}'.format(pipe_short.steps))

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print('Этапы конвейера:\n{}'.format(pipe.steps))

c = load_breast_cancer()

pipe.fit(c.data)
components = pipe.named_steps['pca'].components_
#print('Форма components: {}'.format(components.shape))

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, random_state=4)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

#print('Лучшая модель, найденная GridSearchCV:\n{}'.format(grid.best_estimator_))
#print('Этап логистической регрессии:\n{}'.format(
#    grid.best_estimator_.named_steps['logisticregression']))
#print('Коэффициенты логистической регрессии:\n{}'.format(
#    grid.best_estimator_.named_steps['logisticregression'].coef_))

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, random_state=0)

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),
    vmin=0, cmap='viridis')
plt.xlabel('ridge__alpha')
plt.ylabel('polynomialfeatures__degree')
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),
    param_grid['polynomialfeatures__degree'])
plt.colorbar()

#print('Наилучшие параметры: {}'.format(grid.best_params_))
#print('Правильность на тестовом наборе: {:.2f}'.format(
#    grid.score(X_test, y_test)))

param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
#print('Правильность без полиномиального преобразования: {:.2f}'.format(
#    grid.score(X_test, y_test)))

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
    'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
    'preprocessing':[None], 'classifier__max_features': [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(
    c.data, c.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print('Наилучшие параметры:\n{}\n'.format(grid.best_params_))
print('Наилучшее значение правильности перекрёстной проверки: {:.2f}'.format(
    grid.best_score_))
print('Правильность на тестовом наборе: {:.2f}'.format(
    grid.score(X_test, y_test)))
# %%
