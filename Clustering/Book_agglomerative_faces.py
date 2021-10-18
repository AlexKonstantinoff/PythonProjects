#%%
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from Book_KMeans_faces import X_pca, labels_km, people, image_shape, X_people, y_people
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)

print("Размеры кластеров для агломеративной кластеризации: {}".format(
    np.bincount(labels_agg)))

print('ARI: {:.2f}'.format(adjusted_rand_score(labels_agg, labels_km)))

#linkage_array = ward(X_pca)
#plt.figure(figsize=(20, 5))
#dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
#plt.xlabel("Индекс примера")
#plt.ylabel("Кластерное расстояние")

#n_clusters = 10
#for cluster in range(n_clusters):
#    mask = labels_agg == cluster
#    fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()},
#    figsize=(15, 8))
#    axes[0].set_ylabel(np.sum(mask))
#    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
#        labels_agg[mask], axes):
#        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#        ax.set_title(people.target_names[label].split()[-1],
#        fontdict={'fontsize': 9})

agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print("Размеры кластеров для агломеративной кластеризации: {}".format(
    np.bincount(labels_agg)))

n_clusters = 40
for cluster in [0, 2, 17, 23, 25, 29, 31, 36, 38, 39]:
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 20, subplot_kw={'xticks': (), 'yticks': ()},
    figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel('#{}: {}'.format(cluster, cluster_size))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
    labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
        fontdict={'fontsize': 9})
        for i in range(cluster_size, 15):
            axes[i].set_visible(False)
# %%
