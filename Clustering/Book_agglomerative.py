#%%
import mglearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

#X, y = make_blobs(random_state=170)

#agg = AgglomerativeClustering(cluster_n_clusters=3)
#assignment = agg.fit_predict(X)

#mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
#plt.xlabel('Признак 1')
#plt.ylabel('Признак 2')

#mglearn.plots.plot_agglomerative_algorithm()
#mglearn.plots.plot_agglomerative()

X, y = make_blobs(random_state=0, n_samples=12)
linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, 'два кластера', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, 'три кластера', va='center', fontdict={'size': 15})
plt.xlabel('Индекс наблюдения')
plt.ylabel('Кластерное расстояние')
# %%
