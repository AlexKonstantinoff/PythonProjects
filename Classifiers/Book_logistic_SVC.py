#%%
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import mglearn

X, y = mglearn.datasets.make_forge()

#mglearn.plots.plot_linear_svc_regularization()

can = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(can.data, can.target, stratify=can.target, random_state=42)

logreg = LogisticRegression(C=1000, max_iter=5000).fit(X_train, y_train)
logreg1 = LogisticRegression(max_iter=5000).fit(X_train, y_train)
logreg0001 = LogisticRegression(C = 0.001, max_iter=5000).fit(X_train, y_train)

print('Правильность на обучающем наборе: {:.3f}'.format(logreg.score(X_train, y_train)))
print('Правильность на тестовом наборе: {:.3f}'.format(logreg.score(X_test, y_test)))

#plt.plot(logreg.coef_.T, 'o', label='C=1000')
#plt.plot(logreg1.coef_.T, '^', label='C=1')
#plt.plot(logreg0001.coef_.T, 'v', label='C=0.001')
#plt.xticks(range(can.data.shape[1]), can.feature_names, rotation=90)
#plt.hlines(0, 0, can.data.shape[1])
#plt.ylim(-5, 5)
#plt.xlabel("Индекс коэффициента")
#plt.ylabel("Оценка коэффициента")
#plt.legend()

for C, marker in zip([100, 1, 0.001], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, solver='liblinear', penalty='l1', max_iter=1000).fit(X_train, y_train)
    print("Правильность при обучении для логитрегрессии l1 с C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Правильность при тесте для логитрегрессии l1 с C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))

    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    plt.xticks(range(can.data.shape[1]), can.feature_names, rotation=90)
    plt.hlines(0, 0, can.data.shape[1])
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")

plt.ylim(-5, 5)
plt.legend(loc=3)

#fig, axes = plt.subplots(1, 2, figsize=(10, 3))

#for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#    clf = model.fit(X, y)
#    mglearn.plots.plot_2d_separator(clf, X, eps=0.5, ax=ax, alpha=.7)
#    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{}".format(clf.__class__.__name__))
#    ax.set_xlabel("Признак 0")
#    ax.set_ylabel("Признак 1")
#axes[0].legend()
# %%
