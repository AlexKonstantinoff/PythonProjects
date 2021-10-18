#%%
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import mglearn

X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()

clusters = dbscan.fit_predict(X)
print('Принадлежность к кластерам:\n{}'.format(clusters))

for ms in range (1, 6):
    for e in np.arange(0.1, 4.2, 0.5):
        dbscan = DBSCAN(eps=e, min_samples=ms, n_jobs=-1)
        clusters = dbscan.fit_predict(X)
        print('Принадлежность для min_samples = {}; eps = {:.2f}:\n{}'.format(
            ms, e, clusters))

#mglearn.plots.plot_dbscan()

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
# %%
