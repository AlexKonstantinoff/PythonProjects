#%%
from matplotlib import pyplot as plt
from Book_faces import X_train
import mglearn
import numpy as np
from Book_faces import X_train, X_test, image_shape
from sklearn.decomposition import NMF

nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)

X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

#fig, axes = plt.subplots(3, 5, figsize=(15,12), 
#subplot_kw={'xticks': (), 'yticks': ()})
#for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
#    ax.imshow(component.reshape(image_shape))
#    ax.set_title("{}. component".format((i+1)))

comp = 7

inds = np.argsort(X_train_nmf[:, comp])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), 
subplot_kw={'xticks': (), 'yticks': ()})

for i, (ind, ax) in (enumerate(zip(inds, axes.ravel()))):
    ax.imshow(X_train[ind].reshape(image_shape))
    ax.set_title("{} component".format(comp+1))

#mglearn.plots.plot_nmf_illustration()
#mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
# %%
