#%%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mglearn
import numpy as np

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

y_named = np.array(["blue", "red"])[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test =\
train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

print("Форма массива X_test: {}".format(X_test.shape))
print("Форма решающей функции: {}".format(gbrt.decision_function(X_test).shape))

print("Решающая функция:\n{}".format(gbrt.decision_function(X_test)[:6]))
print("Решающая функция с порогом отсечения:\n{}".format(
    gbrt.decision_function(X_test) > 0))
print("Прогнозы:\n{}".format(gbrt.predict(X_test)))

greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)

pred = gbrt.classes_[greater_zero]

print("Pred идентичен прогнозам: {}".format(np.all(pred == gbrt.predict(X_test))))

decision_f = gbrt.decision_function(X_test)
print("Решающая функция минимум: {:.2f}; максимум: {:.2f}".format(
    np.min(decision_f), np.max(decision_f)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, 
#fill=True, cm=mglearn.cm2)
#scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], 
#alpha=.4, cm=mglearn.ReBl)

#for ax in axes:
#    mglearn.discrete_scatter(
#        X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
#    mglearn.discrete_scatter(
#        X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
#    ax.set_xlabel("Характеристика 0")
#    ax.set_ylabel("Характеристика 1")
#    cbar = plt.colorbar(scores_image, ax=axes.tolist())
#    axes[0].legend(["Тест класс 0", "Тест класс 1", "Обучение класс 0",
#    "Обучение класс 1"], ncol=4, loc=(.1, 1.1))

print("Форма вероятностей: {}".format(gbrt.predict_proba(X_test).shape))
print("Спрогнозированные вероятности:\n{}".format(
    gbrt.predict_proba(X_test[:6])))

mglearn.tools.plot_2d_separator(
    gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel('Характеристика 0')
    ax.set_ylabel('Характеристика 1')
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Тест класс 0", "Тест класс 1", 
"Обуч класс 0", "Обуч класс 1"], ncol=4, loc=(.1, 1.1))
# %%
