#%%
import mglearn
import graphviz
import pydotplus
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

c = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, stratify=c.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

dot_data = export_graphviz(tree, out_file=None, class_names=["malignant", "benign"],
feature_names=c.feature_names, impurity=False, filled=True)

gr = pydotplus.graph_from_dot_data(dot_data)
gr.write_pdf("tree.pdf")
display(Image(gr.create_png()))

#with open("tree.dot") as f:
#    graph = f.read()
#graphviz.Source(graph)

print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))
for name, score in zip(c["feature_names"], tree.feature_importances_):
    print("Важность признака {} -- {}".format(name, score))

def plot_f_importance(model):
    n_features = c.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), c.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")

plot_f_importance(tree)

#mglearn.plots.plot_animal_tree()
#mglearn.plots.plot_tree_progressive()
# %%
import mglearn
from IPython.display import Image, display

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)
# %%
