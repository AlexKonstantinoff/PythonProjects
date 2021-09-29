#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

def load_citibike():
    data_mine = pd.read_csv('NYC-BikeShare.csv', nrows=30000)
    data_mine['one'] = 1
    data_mine['Start Time'] = pd.to_datetime(data_mine['Start Time'])
    data_starttime = data_mine.set_index('Start Time')
    data_resampled = data_starttime.resample('3h').sum().fillna(0)

    return data_resampled.one

citibike = load_citibike()

# print('Данные Citi Bike: \n{}'.format(citibike.head(10)))

#plt.figure(figsize=(10, 3), dpi=600)
xticks = pd.date_range(
    start=citibike.index.min(), end=citibike.index.max(), freq='D')
#plt.xticks(xticks, xticks.strftime("%a %d-%m, %Y"), rotation=90, ha='left')
#plt.plot(citibike, linewidth=1)
#plt.xlabel('Дата')
#plt.ylabel('Частота проката')

y = citibike.values

X = citibike.index.astype('int64').values.reshape(-1, 1) // 10**9

X_hour = citibike.index.hour.values.reshape(-1, 1)

X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1), 
                        citibike.index.hour.values.reshape(-1, 1)])

n_train = 184

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print('R^2 для тестового набора: {:.2f}'.format(
        regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3), dpi=600)

    plt.xticks(
        range(0, len(X), 8), xticks.strftime('%a %d-%m, %Y'), rotation = 90, ha='left')

    plt.plot(range(n_train), y_train, label='Обучение')

    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='Тест')
    plt.plot(range(n_train), y_pred_train, '--', label='Прогноз обучения')

    plt.plot(range(
        n_train, len(y_test) + n_train), y_pred, '--', label='Прогноз тест')

    plt.legend(loc=(1.01, 0))
    plt.xlabel('Дата')
    plt.ylabel('Частота проката')

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
#plt.figure()
#eval_on_features(X, y, regressor)
#eval_on_features(X_hour, y, regressor)
#eval_on_features(X_hour_week, y, regressor)
#eval_on_features(X_hour_week, y, LinearRegression())

enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

#eval_on_features(X_hour_week_onehot, y, Ridge())

poly_transform = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transform.fit_transform(X_hour_week_onehot)
lr = Ridge().fit(X_hour_week_onehot_poly, y)
#eval_on_features(X_hour_week_onehot_poly, y, lr)

hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour

feautres_poly = poly_transform.get_feature_names(features)
features_nonzero = np.array(feautres_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize=(15, 2), dpi=600)
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel('Оценка коэффициента')
plt.ylabel('Признак')
# %%
