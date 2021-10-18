#%%
import mglearn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

c = load_breast_cancer()
scaler = StandardScaler()

new_data = scaler.fit_transform(c.data)

#fig, axes = plt.subplots(10, 3, figsize=(10, 20))
malignant = new_data[c.target == 1]
benign = new_data[c.target == 0]

#ax = axes.ravel()

#for i in range(30):
#    _, bins = np.histogram(new_data[:, i], bins=50)
#    ax[i].hist(malignant[:, i], bins = bins, color=mglearn.cm3(2), alpha=.5)
#    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#    ax[i].set_title(c.feature_names[i])
#    ax[i].set_yticks(())
#ax[0].set_xlabel("Значение признака")
#ax[0].set_ylabel("Частота")
#ax[0].legend(["Benign", "Malignant"], loc="best")
#fig.tight_layout()

pca = PCA(n_components=2)
pca.fit(new_data)
X_pca = pca.transform(new_data)

print("Форма исходного массива: {}".format(str(new_data.shape)))
print("Форма массива после сокращения размерности: {}".format(str(X_pca.shape)))

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], c.target)
plt.legend(c.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")

np.set_printoptions(precision=3, suppress=False, floatmode='fixed')

print("Форма главных компонент: {}".format(pca.components_.shape))
print("Компоненты PCA:\n{}".format(pca.components_))

plt.matshow(pca.components_, cmap='winter')
plt.yticks([0, 1], ["Первая компонента", "Вторая компонента"])
plt.colorbar()
plt.xticks(range(len(c.feature_names)), c.feature_names, rotation=60, ha='left')
plt.xlabel('Характеристика')
plt.ylabel('Главные компоненты')

#mglearn.plots.plot_pca_illustration()

# %%
