#%%
import mglearn
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt

c = load_breast_cancer()

X, y = make_blobs(n_samples=80, centers=5, random_state=3, cluster_std=2)

X_tr_bl, X_t_bl = train_test_split(X, random_state=5, test_size=.1)
X_train, X_test, y_train, y_test = train_test_split(c.data, c.target, random_state=1)

sM = MinMaxScaler()
sR = RobustScaler()
scaler = MinMaxScaler()
scalerR = RobustScaler()

scaler.fit(X_train)
scalerR.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_train_scaledR = scalerR.transform(X_train)

#print("Форма преобразованного массива: {}".format(X_train_scaled.shape))
#print("Min значение признака до масштабирования:\n{}".format(X_train.min(axis=0)))
#print("Max значение признака до масштабирования:\n{}".format(X_train.max(axis=0)))
#print("Min значение признака после масштабирования:\n{}".format(
#    X_train_scaled.min(axis=0)))
#print("Max значение признака после масштабирования:\n{}".format(
#    X_train_scaled.max(axis=0)))

X_test_scaled = scaler.transform(X_test)
X_test_scaledR = scalerR.transform(X_test)

print("Min значение признака после масштабирования (Min - max):\n{}".format(
    X_test_scaled.min(axis=0)))
print("Max значение признака после масштабирования (Min - max):\n{}".format(
    X_test_scaled.max(axis=0)))

print("Min значение признака после масштабирования (Robust):\n{}".format(
    X_test_scaledR.min(axis=0)))
print("Max значение признака после масштабирования:\n{}".format(
    X_test_scaledR.max(axis=0)))

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
axes[0].scatter(X_tr_bl[:, 0], X_tr_bl[:, 1],
    c=mglearn.cm2(0), label="Обучающий набор", s=60)
axes[0].scatter(X_t_bl[:, 0], X_t_bl[:, 1], marker='^',
    c=mglearn.cm2(1), label="Тестовый набор", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Исходные данные")

X_tr_scaled = sM.fit_transform(X_tr_bl)
X_t_scaled = sM.fit(X_tr_bl).transform(X_t_bl)

sR.fit(X_tr_bl)
X_tr_scaledR = sR.transform(X_tr_bl)
X_t_scaledR = sR.transform(X_t_bl)

#
# Ниже пример !!!НЕПРАВИЛЬНОГО!!! масштабирования данных! 
# Тестовые данные нельзя масштабировать отдельно!
#

test_scaler = MinMaxScaler()
test_scaler.fit(X_t_bl)
X_test_scaled_badly = test_scaler.transform(X_t_bl)

#
# Данные необходимо масштабировать ранее подогнанным скейлером!
#

axes[1].scatter(X_tr_scaled[:, 0], X_tr_scaled[:, 1],
    c=mglearn.cm2(0), label="Обучающий набор", s=60)
axes[1].scatter(X_t_scaled[:, 0], X_t_scaled[:, 1], marker='^',
    c=mglearn.cm2(1), label="Тестовый набор", s=60)
axes[1].set_title("Масштабированные данные (Min - max)")

axes[2].scatter(X_tr_scaledR[:, 0], X_tr_scaledR[:, 1],
    c=mglearn.cm2(0), label="Обучающий набор", s=60)
axes[2].scatter(X_t_scaledR[:, 0], X_t_scaledR[:, 1], marker='^',
    c=mglearn.cm2(1), label="Тестовый набор", s=60)
axes[2].set_title("Масштабированные данные (Robust)")

axes[3].scatter(X_tr_scaled[:, 0], X_tr_scaled[:, 1],
    c=mglearn.cm2(0), label="Обучающий набор", s=60)
axes[3].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^',
    c=mglearn.cm2(1), label="Тестовый набор", s=60)
axes[3].set_title("Неправильно масштабированные данные")

for ax in axes:
    ax.set_xlabel("Признак 0")
    ax.set_ylabel("Признак 1")

#mglearn.plots.plot_scaling()
# %%
