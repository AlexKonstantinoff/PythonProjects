#%%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("Форма решающей функции: {}".format(gbrt.decision_function(X_test).shape))
print("Решающая функция:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
print("Argmax решающей функции:\n{}".format(np.argmax(
    gbrt.decision_function(X_test), axis=1)))
print("Прогнозы:\n{}".format(gbrt.predict(X_test)))

print('Спрогнозированные вероятности:\n{}'.format(gbrt.predict_proba(X_test)[:6]))
print('Суммы: {}'.format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
print('Argmax спрогнозированных вероятностей:\n{}'.format(
    np.argmax(gbrt.predict_proba(X_test), axis=1)))

logreg = LogisticRegression()

named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print('Уникальные классы в обучающем наборе: {}'.format(logreg.classes_))
print('Прогнозы: {}'.format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print('Argmax решающей функции: {}'.format(argmax_dec_func[:10]))
print('Argmax, объединённый с классами: {}'.format(
    logreg.classes_[argmax_dec_func][:10]))
print('Predict_proba:'.format(logreg.predict_proba(X_test)[:6]))
# %%
