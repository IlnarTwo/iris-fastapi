from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import to_categorical
from keras.utils import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np

# Загрузка данных и разбиение на обучающую и тестовую выборки
iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# Определение модели
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=65, batch_size=10, validation_data=(X_test, y_test))

# Оценка модели
scores = model.evaluate(X_test, y_test, verbose=2)

# Прогнозирование классов для тестовых данных
classes = model.predict(X_test)

# Расчет точности прогнозирования
accuration = np.sum(np.around(classes) == y_test) / (len(classes)*3)*100
print("Точность прогнозирования: " + str(accuration) + '%')
print("Прогноз:")
print(np.around(classes))
print("На самом деле:")
print(y_test)
print(np.around(classes) == y_test)