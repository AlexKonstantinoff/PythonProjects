#%%
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from Book_MLP import X_train, y_train, mlp
import mglearn

c = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, 
random_state=0, stratify=c.target)

mlp.fit(X_train, y_train)
print("Точность предсказания на обучающем наборе: {:.2f}".format(mlp.score(X_train, y_train)))
print("Точность предсказания на тестовом наборе: {:.2f}".format(mlp.score(X_test, y_test)))

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp2 = MLPClassifier(random_state=0)

mlp2.fit(X_train_scaled, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(mlp2.score(X_train_scaled, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(mlp2.score(X_test_scaled, y_test)))

plt.figure(figsize=(20, 5))
plt.imshow(mlp2.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), c.feature_names)
plt.xlabel("Столбцы матрицы весов")
plt.ylabel("Входная характеристика")
plt.colorbar()

#print("Максимальные значения характеристик:\n{}".format(c.data.max(axis=0)))
#fig, axes = plt.subplots(2, 4, figsize=(20, 8))
#for axx, n_hidden_nodes in zip(axes, [10, 100]):
#    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
#        mlp = MLPClassifier(solver='lbfgs', random_state=0, 
#        hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
#        mlp.fit(X_train, y_train)
#        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
#        ax.set_title('n hidden nodes = [{}, {}]\nalpha={:.4f}'.format(
#            n_hidden_nodes, n_hidden_nodes, alpha))

#for i, ax in enumerate(axes.ravel()):
#    mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])
#    mlp.fit(X_train, y_train)
#    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.6, ax=ax)
#    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)

# %%
