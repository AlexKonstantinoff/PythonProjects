#%%
import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

c = load_breast_cancer()

#X, y = mglearn.tools.make_handcrafted_dataset()

#svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, random_state=0)

svc = SVC(C=10, gamma=0.0001)
svc.fit(X_train, y_train)

print("Правильность на обучающем наборе: {:.2f}".format(svc.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label='min')
plt.plot(X_train.max(axis=0), '^', label='max')
plt.legend(loc=4)
plt.xlabel('Индекс признака')
plt.ylabel('Величина признака')
plt.yscale('log')

#mglearn.plots.plot_2d_separator(svm, X, eps=.5)
#mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

#sv = svm.support_vectors_
#sv_labels = svm.dual_coef_.ravel() > 0
#mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
#plt.xlabel("Признак 0")
#plt.ylabel("Признак 1")

#fig, axes = plt.subplots(3, 3, figsize=(15, 10))

#for ax, C in zip(axes, [-1, 0, 3]):
#    for a, gamma in zip(ax, range(-1, 2)):
#        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

#axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"], 
#ncol=4, loc=(.9, 1.2))
# %%
