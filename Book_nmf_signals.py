#%%
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF, PCA

S = mglearn.datasets.make_signals()
#plt.figure(figsize=(6, 1))
#plt.plot(S, '-')
#plt.xlabel("Время")
#plt.ylabel("Сигнал")

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Форма измерений: {}".format(X.shape))

nmf = NMF(n_components=3, random_state=46)
S_nmf = nmf.fit_transform(X)
print("Форма восстановленного сигнала: {}".format(S_nmf.shape))

pca = PCA(n_components=3)
S_pca = pca.fit_transform(X)

models = [X, S, S_nmf, S_pca]
names = ['Наблюдения (первые три измерения)',
'Фактические источники',
'Сигналы, восстановленные NMF',
'Сигналы, восстановленные PCA']

fig, axes = plt.subplots(4, figsize=(8,4), gridspec_kw={'hspace': .5},
subplot_kw={'xticks':(), 'yticks':()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
# %%
