from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_parameter=1, second_parameter=2):
        self.__first_parameter = first_parameter
        self.__second_parameter = second_parameter

    def fit(self, X, y=None):

        print('Подгонка модели должна быть осуществлена здесь!')

        return self

    def transform(self, X):
        #   Для классификатора или кластеризатора вместо метода transform
        #   можно определить метод predict

        X_transformed = X + 1

        return X_transformed

X = 2
y = 3

transformer = MyTransformer(X, y)

transformer.fit(X, y)
transformer.transform(X)