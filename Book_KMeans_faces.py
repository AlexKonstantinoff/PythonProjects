#%%
import mglearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import mglearn
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

people = fetch_lfw_people(resize=0.7, min_faces_per_person=20)
image_shape = people.images[0].shape

counts = np.bincount(people.target)
mask = np.zeros(people.target.shape, dtype=np.bool8)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255

pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)

km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)

print("Размеры кластеров k-средние: {}".format(np.bincount(labels_km)))

fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()}, 
figsize=(12, 4))

for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)

mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, 
y_people, people.target_names)
# %%
