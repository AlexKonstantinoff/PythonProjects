from Book_deep_SVM import X_test, X_train, y_train, y_test
from sklearn.svm import SVC

min_on_training = X_train.min(axis=0)

range_on_training = (X_train - min_on_training).max(axis=0)

X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC(C=10, gamma=2.8)
svc.fit(X_train_scaled, y_train)

print("Правильность на обучающем наборе: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#print("Минимальное значение для каждого признака\n{}".format(X_train_scaled.min(axis=0)))
#print("Максимальное значение для каждого признака\n{}".format(X_train_scaled.max(axis=0)))
