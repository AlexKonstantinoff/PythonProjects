#%%

from Book_faces import X_people, image_shape, y_people, people
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)

print('Уникальные метки: {}'.format(np.unique(labels)))
print('Количество точек на кластер: {}'.format(np.bincount(labels + 1)))

#Получено небольшое количество шумовых точек (метка -1). Они пригодны для просмотра

noise = X_people[labels==-1]
#fig, axes = plt.subplots(4, 8, subplot_kw={'xticks': (), 'yticks': ()},
#figsize=(16, 6))

#for image, ax in zip(noise, axes.ravel()):
#    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)

# Наиболее интересное разбиение на кластеры получено при eps=7

dbscan = DBSCAN(min_samples=3, eps=7)
labels=dbscan.fit_predict(X_pca)

for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images = np.sum(mask)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
    subplot_kw={'xticks': (), 'yticks': ()})

print('Размеры кластеров: {}'.format(np.bincount(labels + 1)))

for image, label, ax in zip(X_people[mask], y_people[mask], axes):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
    ax.set_title(people.target_names[label].split()[-1])
# %%
