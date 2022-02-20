import pandas
from sklearn.tree import DecisionTreeClassifier

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data_raw = pandas.read_csv('../titanic.csv')

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
# возраст пассажира (Age) и его пол (Sex).
data = data_raw.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'])

# В данных есть пропущенные значения — например, для некоторых
# пассажиров неизвестен их возраст. Такие записи при чтении их в
# pandas принимают значение nan. Найдите все объекты, у которых
# есть пропущенные признаки, и удалите их из выборки.
data.dropna(inplace=True)

# Обратите внимание, что признак Sex имеет строковые значения
data = data.replace({'Sex': {'female': 1, 'male': 0}})

# Выделите целевую переменную — она записана в столбце Survived.
survived = data['Survived']
data = data.drop(columns='Survived')

# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию.
clf = DecisionTreeClassifier(random_state=421)
clf.fit(data, survived)

# Вычислите важности признаков и найдите два признака с наибольшей важностью.
importances = clf.feature_importances_

print(data.head())
print(importances)
