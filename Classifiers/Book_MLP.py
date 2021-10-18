#%%
from IPython.display import display
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=4)

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10], 
activation='tanh', max_iter=500)
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")

#display(mglearn.plots.plot_single_hidden_layer_graph())
#display(mglearn.plots.plot_two_hidden_layer_graph())

#line = np.linspace(-3, 3, 100)
#plt.plot(line, np.tanh(line), label="tanh")
#plt.plot(line, np.maximum(line, 0), label="Rectified linear unit (relu)")
#plt.legend(loc="best")
#plt.xlabel("X")
#plt.ylabel("relu(x), tg(x)")
# %%
