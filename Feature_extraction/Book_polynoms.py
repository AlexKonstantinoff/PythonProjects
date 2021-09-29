#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=100)

line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)

encoder = OneHotEncoder(sparse=False)
X_binned = encoder.fit_transform(np.digitize(X, bins=bins))

line_binned = encoder.transform(np.digitize(line, bins=bins))

X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])

#plt.plot(
#    line, reg.predict(line_combined), 
#    label='Линейная регрессия после комбинирования')

#for bin in bins:
#    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.legend(loc='best')
plt.ylabel('Выход регрессии')
plt.xlabel('Входной признак')
plt.plot(X[:, 0], y, 'o', c='k')

X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(
    line, reg.predict(line_product), label='Линейная регрессия произведения')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

# %%
