#%%
import numpy as np
import mglearn
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

#fig, axes = plt.subplots(2, 5, figsize=(15, 8),
#subplot_kw={'xticks':(), 'yticks':()})

#for target, image, ax in zip(people.target, people.images, axes.ravel()):
#    ax.imshow(image)
#    ax.set_title(people.target_names[target])

print("Форма массива изображений лиц: {}".format(people.images.shape))
print("Количество классов: {}".format(len(people.target_names)))

counts = np.bincount(people.target)

#for i, (count, name) in enumerate(zip(counts, people.target_names)):
#    print("{0:25} {1:3}".format(name, count), end=' ')
#    if (i + 1) % 3 == 0:
#        print()

mask = np.zeros(people.target.shape, dtype=np.bool8)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Правильность на тестовом наборе: {:.2f}".format(
    knn.score(X_test, y_test)))

#mglearn.plots.plot_pca_whitening()

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Размерность обучающих данных после PCA: {}".format(X_train_pca.shape))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Правильность на тестовом наборе после PCA: {:.2f}".format(
    knn.score(X_test_pca, y_test)))

print("Форма pca.components_: {}".format(pca.components_.shape))

#fig, axes = plt.subplots(3, 5, figsize=(15, 12),
#subplot_kw={'xticks': (), 'yticks': ()})
#for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
#    ax.imshow(component.reshape(image_shape),
#    cmap='viridis')
#    ax.set_title('{}. component'.format((i+1)))

#mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

#mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
#plt.xlabel("Первая главная компонента")
#plt.ylabel("Вторая главная компонента")
# %%
