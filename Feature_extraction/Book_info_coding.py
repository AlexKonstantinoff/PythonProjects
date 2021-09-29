#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('adult_data.csv', header=None, index_col=False,
    names=['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num',
    'Marital-status', 'Occupation', 'Relationship', 'Race', 'Gender',
    'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country',
    'Income'])
data = data[['Age', 'Workclass', 'Education','Gender',
    'Hours-per-week', 'Occupation', 'Income']]

data.head(10)
#print(data.Gender.value_counts())
#print(data.Income.value_counts())

#print('Исходные признаки:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
#print('Признаки после get_dummies:\n', list(data_dummies.columns))
data_dummies.head(10)

features = data_dummies.to_numpy()[:, 0:44]
X = features
y = data_dummies['Income_ >50K'].values
print('Форма массива X: {}; форма массива y: {}'.format(
    X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2)
logreg = LogisticRegression(max_iter=300, n_jobs=-1)
logreg.fit(X_train, y_train)
print('Правильность на тестовом наборе: {:.2f}'.format(
    logreg.score(X_test, y_test)))

# %%

import pandas as pd
from IPython.display import display

demo_df = pd.DataFrame({'Целочисленный признак': [0, 1, 2, 1],
    'Категориальный признак': ['socks', 'fox', 'socks', 'box']})

display(demo_df)
pd.get_dummies(demo_df)

demo_df['Целочисленный признак'] = demo_df[
    'Целочисленный признак'].astype(str)
pd.get_dummies(
    demo_df, columns=['Целочисленный признак', 'Категориальный признак'])

# %%
