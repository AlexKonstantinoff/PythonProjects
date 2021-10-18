from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

c = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    c.data, c.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Правильность на тестовом наборе: {:.3f}".format(
    svm.score(X_test, y_test)))

scaler = MinMaxScaler()
X_train_tr = scaler.fit_transform(X_train)
X_test_tr = scaler.fit(X_train).transform(X_test)

svm.fit(X_train_tr, y_train)
print("Правильность предсказания после масштабирования данных (Min-Max): {:.3f}".format(
    svm.score(X_test_tr, y_test)))

scalerR = RobustScaler()
scalerR.fit(X_train)
X_train_R = scalerR.transform(X_train)
X_test_R = scalerR.transform(X_test)
svm.fit(X_train_R, y_train)
print("Правильность предсказания после масштабирования данных (Robust): {:.3f}".format(
    svm.score(X_test_R, y_test)))

StdScaler = StandardScaler()
StdScaler.fit(X_train)
X_train_std = StdScaler.transform(X_train)
X_test_std = StdScaler.transform(X_test)
svm.fit(X_train_std, y_train)
print("Правильность предсказания после масштабирования \
данных (StandartScaler): {:.3f}".format(
    svm.score(X_test_std, y_test)))