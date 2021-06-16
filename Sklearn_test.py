from sklearn import datasets
import numpy

iris = datasets.load_iris()
digits = datasets.load_digits()

print(numpy.column_stack([iris.data, iris.target]))
