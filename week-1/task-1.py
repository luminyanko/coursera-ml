import pandas
import re

data = pandas.read_csv('./titanic.csv', index_col='PassengerId')

passenger_count = len(data.index)

# Какое количество мужчин и женщин ехало на корабле?
print(data['Sex'].value_counts())

# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
survived_count = data[data["Survived"] == 1].count().values[1]
print(round(survived_count / passenger_count * 100), 2)

# Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах.
first_class = data[data["Pclass"] == 1].count().values[1]
print(round(first_class / passenger_count * 100), 2)

# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
print(data['Age'].mean())
print(data['Age'].median())

# Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
print(round(data['SibSp'].corr(data['Parch'])), 2)

# Какое самое популярное женское имя на корабле?
data_female = data[data["Sex"] == 'female']


def first_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')
    return name


names = data[data['Sex'] == 'female']['Name'].map(first_name)
name_counts = names.value_counts()

print(name_counts.head(1).index.values[0])
