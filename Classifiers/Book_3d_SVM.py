#%%
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
import mglearn as mg
import numpy as np

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mg.discrete_scatter(X[:, 0], X[:, 1], y)

linear = LinearSVC().fit(X, y)

mg.plots.plot_2d_separator(linear, X)
mg.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")

X_new = np.hstack([X, X[:, 1:] ** 2])

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

fig = plt.figure()
#ax = Axes3D(fig, elev=-145, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

mask = y == 0
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
#ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
#ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mg.cm2, s=60)
#ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker = '^', 
#cmap=mg.cm2, s=60)
#ax.set_xlabel("Признак 0")
#ax.set_ylabel("Признак 1")
#ax.set_zlabel("Признак 1 ** 2")

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], 
cmap=mg.cm2, alpha=0.5)
mg.discrete_scatter(X[:, 0], X[:, 1], y)
# %%
