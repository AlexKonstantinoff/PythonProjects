#%%
from matplotlib import pyplot as plt
import mglearn
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans

#mglearn.plots.plot_kmeans_algorithm()
#mglearn.plots.plot_kmeans_boundaries()

X, y = make_blobs(random_state=1, n_samples=80)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

#print('Принадлежность к кластерам:\n{}'.format(kmeans.labels_))
#print('Предсказанные значения:\n{}'.format(kmeans.predict(X)))
np.set_printoptions(precision=3)

#sprint('Срез массива X:\n{}'.format(X[:, 1]))

#mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
#mglearn.discrete_scatter(
#    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
#    markers='^', markeredgewidth=2)

#fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#kmeans = KMeans(n_clusters=2)
#kmeans.fit(X)
#assignments = kmeans.labels_

#mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

#kmeans = KMeans(n_clusters=5)
#kmeans.fit(X)
#assignments = kmeans.labels_

#mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

#X_v, y_v = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
#y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_v)

#mglearn.discrete_scatter(X_v[:, 0], X_v[:, 1], y_pred)
#plt.legend(['Кластер 1', 'Кластер 2', 'Кластер 3'], loc='best')
#plt.xlabel('Признак 1')
#plt.ylabel('Признак 2')

#X, y = make_blobs(random_state=170, n_samples=600)
#rng = np.random.RandomState(74)

#transformation = rng.normal(size=(2, 2))
#X = np.dot(X, transformation)

#kmeans = KMeans(n_clusters=3)
#kmeans.fit(X)
#y_pred = kmeans.predict(X)

#plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#marker='^', c=[0, 1, 2], s=100, linewidth=2, cmap=mglearn.cm3)
#plt.xlabel('Признак 1')
#plt.ylabel('Признак 2')

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10)
y_pred = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^',
s=60, c=range(kmeans.n_clusters), linewidth=2)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

distance_features = kmeans.transform(X)
print('Форма характеристик-расстояний: {}'.format(distance_features.shape))
print('Характеристики-расстояния:\n{}'.format(distance_features))
# %%
