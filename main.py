from fastapi import FastAPI, Form
import numpy as np
import keras

app = FastAPI()

# Загрузка обученной модели
model = keras.applications.fisherLab6()

# Определение конечной точки FastAPI для обработки запросов POST
@app.post("/classify")
async def classify_iris(iris: str = Form(...), sepal_length: float = Form(...), sepal_width: float = Form(...), petal_length: float = Form(...), petal_width: float = Form(...)):
    # Преобразование входных данных в массив NumPy
    sepal_length_value = float(sepal_length)
    sepal_width_value = float(sepal_width)
    petal_length_value = float(petal_length)
    petal_width_value = float(petal_width)
    sepal_length_vector = np.array([sepal_length_value])
    sepal_width_vector = np.array([sepal_width_value])
    petal_length_vector = np.array([petal_length_value])
    petal_width_vector = np.array([petal_width_value])
    
    # Классификация цветка с помощью обученной модели
    prediction = model.predict([sepal_length_vector, sepal_width_vector, petal_length_vector, petal_width_vector])
    predicted_class = np.argmax(prediction)
    
    # Возврат названия цветка
    if predicted_class == 0:
        return {"predicted_class": "setosa"}
    elif predicted_class == 1:
        return {"predicted_class": "versicolor"}
    elif predicted_class == 2:
        return {"predicted_class": "virginica"}