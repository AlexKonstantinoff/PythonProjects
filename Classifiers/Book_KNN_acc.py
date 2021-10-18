#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

c = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    c.data, c.target, stratify=c.target, random_state=66
)

training_acc = []
test_acc = []
neighbors_settings = range(1, 20)

for n in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)
    training_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_acc, label='Точность на обучающем наборе')
plt.plot(neighbors_settings, test_acc, label='Точность на тестовом наборе')
plt.xlabel("Количество соседей")
plt.ylabel("Точность")
plt.legend()
# %%
