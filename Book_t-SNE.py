#%%
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5),
subplot_kw={'xticks':(), 'yticks':()})

for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)


# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

pca = PCA(n_components=2)
digits_pca = pca.fit_transform(digits.data)
colors = ['#476A2A', '#7851C9', '#BD3430', '#4A2D4E', '#875525',
'#A83683', '#4E655D', '#853541', '#3A3120', '#539D8D']

tsne = TSNE(random_state=0)
digits_tsne = tsne.fit_transform(digits.data)

#plt.figure(figsize=(10, 10))
#plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max() + 1)
#plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max() + 1)
#for i in range(len(digits.data)):
#    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
#    color=colors[digits.target[i]], fontdict={'weight':'bold', 'size': 9})
#    plt.xlabel("Первая главная компонента")
#    plt.ylabel("Вторая главная компонента")

#plt.figure(figsize=(10, 10))
#plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
#plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
#for i in range(len(digits.data)):
#    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
#    color=colors[digits.target[i]], fontdict={'weight':'bold', 'size': 11})
#    plt.xlabel("Первая компонента t-SNE")
#    plt.ylabel("Вторая компонента t-SNE")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i in range(len(digits.data)):
    axes[0].set_xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max() + 1)
    axes[0].set_ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max() + 1)
    axes[0].text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
    color=colors[digits.target[i]], fontdict={'weight':'bold', 'size': 9})
    axes[0].set_title('Разделение признаков PCA')
    axes[0].set_xlabel("Первая главная компонента")
    axes[0].set_ylabel("Вторая главная компонента")

    axes[1].set_xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
    axes[1].set_ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
    axes[1].text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
    color=colors[digits.target[i]], fontdict={'weight':'bold', 'size': 8})
    axes[1].set_title('Разделение признаков t-SNE')
    axes[1].set_xlabel("Первая компонента")
    axes[1].set_ylabel("Вторая компонента")
# %%
